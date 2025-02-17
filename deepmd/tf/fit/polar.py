# SPDX-License-Identifier: LGPL-3.0-or-later
import warnings
from typing import (
    Optional,
)

import numpy as np

from deepmd.tf.common import (
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.tf.descriptor import (
    DescrptSeA,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.loss.tensor import (
    TensorLoss,
)
from deepmd.tf.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.tf.utils.graph import (
    get_fitting_net_variables_from_graph_def,
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.network import (
    one_layer,
    one_layer_rand_seed_shift,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@Fitting.register("polar")
class PolarFittingSeA(Fitting):
    r"""Fit the atomic polarizability with descriptor se_a.

    Parameters
    ----------
    ntypes
            The ntypes of the descriptor :math:`\mathcal{D}`
    dim_descrpt
            The dimension of the descriptor :math:`\mathcal{D}`
    embedding_width
            The rotation matrix dimension of the descriptor :math:`\mathcal{D}`
    neuron : list[int]
            Number of neurons in each hidden layer of the fitting net
    resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    numb_fparam
            Number of frame parameters
    numb_aparam
            Number of atomic parameters
    dim_case_embd
            Dimension of case specific embedding.
    sel_type : list[int]
            The atom types selected to have an atomic polarizability prediction. If is None, all atoms are selected.
    fit_diag : bool
            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.
    scale : list[float]
            The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
    diag_shift : list[float]
            The diagonal part of the polarizability matrix of type i will be shifted by diag_shift[i]. The shift operation is carried out after scale.
    seed : int
            Random seed for initializing the network parameters.
    activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        embedding_width: int,
        neuron: list[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        sel_type: Optional[list[int]] = None,
        fit_diag: bool = True,
        scale: Optional[list[float]] = None,
        shift_diag: bool = True,  # YWolfeee: will support the user to decide whether to use this function
        # diag_shift : list[float] = None, YWolfeee: will not support the user to assign a shift
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        mixed_types: bool = False,
        type_map: Optional[list[str]] = None,  # to be compat with input
        **kwargs,
    ) -> None:
        """Constructor."""
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.sel_type = sel_type
        self.fit_diag = fit_diag
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = one_layer_rand_seed_shift()
        # self.diag_shift = diag_shift
        self.shift_diag = shift_diag
        self.scale = scale
        self.activation_function_name = activation_function
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        if self.sel_type is None:
            self.sel_type = list(range(self.ntypes))
        self.sel_mask = np.array(
            [ii in self.sel_type for ii in range(self.ntypes)], dtype=bool
        )
        if self.scale is None:
            self.scale = np.array([1.0 for ii in range(self.ntypes)])
        else:
            if isinstance(self.scale, list):
                assert len(self.scale) == ntypes, (
                    "Scale should be a list of length ntypes."
                )
            elif isinstance(self.scale, float):
                self.scale = [self.scale for _ in range(ntypes)]
            else:
                raise ValueError(
                    "Scale must be a list of float of length ntypes or a float."
                )
            self.scale = np.array(self.scale)
        # if self.diag_shift is None:
        #    self.diag_shift = [0.0 for ii in range(self.ntypes)]
        if not isinstance(self.sel_type, list):
            self.sel_type = [self.sel_type]
        self.sel_type = sorted(self.sel_type)
        self.constant_matrix = np.zeros(  # pylint: disable=no-explicit-dtype
            self.ntypes
        )  # self.ntypes x 1, store the average diagonal value
        # if type(self.diag_shift) is not list:
        #    self.diag_shift = [self.diag_shift]
        self.dim_rot_mat_1 = embedding_width
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False
        self.fitting_net_variables = None
        self.mixed_prec = None
        self.mixed_types = mixed_types
        self.type_map = type_map
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.dim_case_embd = dim_case_embd
        if numb_fparam > 0:
            raise ValueError("numb_fparam is not supported in the dipole fitting")
        if numb_aparam > 0:
            raise ValueError("numb_aparam is not supported in the dipole fitting")
        if dim_case_embd > 0:
            raise ValueError("dim_case_embd is not supported in TensorFlow.")
        self.fparam_avg = None
        self.fparam_std = None
        self.fparam_inv_std = None
        self.aparam_avg = None
        self.aparam_std = None
        self.aparam_inv_std = None

    def get_sel_type(self) -> list[int]:
        """Get selected atom types."""
        return self.sel_type

    def get_out_size(self) -> int:
        """Get the output size. Should be 9."""
        return 9

    def compute_output_stats(self, all_stat) -> None:
        """Compute the output statistics.

        Parameters
        ----------
        all_stat
            Dictionary of inputs.
            can be prepared by model.make_stat_input
        """
        if "polarizability" not in all_stat.keys():
            self.avgeig = np.zeros([9])  # pylint: disable=no-explicit-dtype
            warnings.warn(
                "no polarizability data, cannot do data stat. use zeros as guess"
            )
            return
        data = all_stat["polarizability"]
        all_tmp = []
        for ss in range(len(data)):
            tmp = np.concatenate(data[ss], axis=0)
            tmp = np.reshape(tmp, [-1, 3, 3])
            tmp, _ = np.linalg.eig(tmp)
            tmp = np.absolute(tmp)
            tmp = np.sort(tmp, axis=1)
            all_tmp.append(tmp)
        all_tmp = np.concatenate(all_tmp, axis=1)
        self.avgeig = np.average(all_tmp, axis=0)

        # YWolfeee: support polar normalization, initialize to a more appropriate point
        if self.shift_diag:
            mean_polar = np.zeros([len(self.sel_type), 9])  # pylint: disable=no-explicit-dtype
            sys_matrix, polar_bias = [], []
            for ss in range(len(all_stat["type"])):
                nframes = all_stat["type"][ss].shape[0]
                atom_has_polar = [
                    w for w in all_stat["type"][ss][0] if (w in self.sel_type)
                ]  # select atom with polar
                if all_stat["find_atom_polarizability"][ss] > 0.0:
                    for itype in range(
                        len(self.sel_type)
                    ):  # Atomic polar mode, should specify the atoms
                        index_lis = [
                            index
                            for index, w in enumerate(atom_has_polar)
                            if w == self.sel_type[itype]
                        ]  # select index in this type

                        sys_matrix.append(np.zeros((1, len(self.sel_type))))  # pylint: disable=no-explicit-dtype
                        sys_matrix[-1][0, itype] = len(index_lis)

                        polar_bias.append(
                            np.sum(
                                all_stat["atom_polarizability"][ss].reshape(
                                    nframes, len(atom_has_polar), -1
                                )[:, index_lis, :]
                                / nframes,
                                axis=(0, 1),
                            ).reshape((1, 9))
                        )
                else:  # No atomic polar in this system, so it should have global polar
                    if (
                        not all_stat["find_polarizability"][ss] > 0.0
                    ):  # This system is just a joke?
                        continue
                    # Till here, we have global polar
                    sys_matrix.append(
                        np.zeros((1, len(self.sel_type)))  # pylint: disable=no-explicit-dtype
                    )  # add a line in the equations
                    for itype in range(
                        len(self.sel_type)
                    ):  # Atomic polar mode, should specify the atoms
                        index_lis = [
                            index
                            for index, w in enumerate(atom_has_polar)
                            if atom_has_polar[index] == self.sel_type[itype]
                        ]  # select index in this type

                        sys_matrix[-1][0, itype] = len(index_lis)

                    # add polar_bias
                    polar_bias.append(
                        np.mean(all_stat["polarizability"][ss], axis=0).reshape((1, 9))
                    )

            matrix, bias = (
                np.concatenate(sys_matrix, axis=0),
                np.concatenate(polar_bias, axis=0),
            )
            atom_polar, _, _, _ = np.linalg.lstsq(matrix, bias, rcond=None)
            for itype in range(len(self.sel_type)):
                self.constant_matrix[self.sel_type[itype]] = np.mean(
                    np.diagonal(atom_polar[itype].reshape((3, 3)))
                )

    @cast_precision
    def _build_lower(self, start_index, natoms, inputs, rot_mat, suffix="", reuse=None):
        # cut-out inputs
        inputs_i = tf.slice(
            inputs, [0, start_index * self.dim_descrpt], [-1, natoms * self.dim_descrpt]
        )
        inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
        rot_mat_i = tf.slice(
            rot_mat,
            [0, start_index * self.dim_rot_mat],
            [-1, natoms * self.dim_rot_mat],
        )
        rot_mat_i = tf.reshape(rot_mat_i, [-1, self.dim_rot_mat_1, 3])
        layer = inputs_i
        for ii in range(0, len(self.n_neuron)):
            if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
                layer += one_layer(
                    layer,
                    self.n_neuron[ii],
                    name="layer_" + str(ii) + suffix,
                    reuse=reuse,
                    seed=self.seed,
                    use_timestep=self.resnet_dt,
                    activation_fn=self.fitting_activation_fn,
                    precision=self.fitting_precision,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.fitting_net_variables,
                    mixed_prec=self.mixed_prec,
                )
            else:
                layer = one_layer(
                    layer,
                    self.n_neuron[ii],
                    name="layer_" + str(ii) + suffix,
                    reuse=reuse,
                    seed=self.seed,
                    activation_fn=self.fitting_activation_fn,
                    precision=self.fitting_precision,
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.fitting_net_variables,
                    mixed_prec=self.mixed_prec,
                )
            if (not self.uniform_seed) and (self.seed is not None):
                self.seed += self.seed_shift
        if self.fit_diag:
            bavg = np.zeros(self.dim_rot_mat_1)  # pylint: disable=no-explicit-dtype
            # bavg[0] = self.avgeig[0]
            # bavg[1] = self.avgeig[1]
            # bavg[2] = self.avgeig[2]
            # (nframes x natoms) x naxis
            final_layer = one_layer(
                layer,
                self.dim_rot_mat_1,
                activation_fn=None,
                name="final_layer" + suffix,
                reuse=reuse,
                seed=self.seed,
                bavg=bavg,
                precision=self.fitting_precision,
                uniform_seed=self.uniform_seed,
                initial_variables=self.fitting_net_variables,
                mixed_prec=self.mixed_prec,
                final_layer=True,
            )
            if (not self.uniform_seed) and (self.seed is not None):
                self.seed += self.seed_shift
            # (nframes x natoms) x naxis
            final_layer = tf.reshape(
                final_layer, [tf.shape(inputs)[0] * natoms, self.dim_rot_mat_1]
            )
            # (nframes x natoms) x naxis x naxis
            final_layer = tf.matrix_diag(final_layer)
        else:
            bavg = np.zeros(self.dim_rot_mat_1 * self.dim_rot_mat_1)  # pylint: disable=no-explicit-dtype
            # bavg[0*self.dim_rot_mat_1+0] = self.avgeig[0]
            # bavg[1*self.dim_rot_mat_1+1] = self.avgeig[1]
            # bavg[2*self.dim_rot_mat_1+2] = self.avgeig[2]
            # (nframes x natoms) x (naxis x naxis)
            final_layer = one_layer(
                layer,
                self.dim_rot_mat_1 * self.dim_rot_mat_1,
                activation_fn=None,
                name="final_layer" + suffix,
                reuse=reuse,
                seed=self.seed,
                bavg=bavg,
                precision=self.fitting_precision,
                uniform_seed=self.uniform_seed,
                initial_variables=self.fitting_net_variables,
                mixed_prec=self.mixed_prec,
                final_layer=True,
            )
            if (not self.uniform_seed) and (self.seed is not None):
                self.seed += self.seed_shift
            # (nframes x natoms) x naxis x naxis
            final_layer = tf.reshape(
                final_layer,
                [tf.shape(inputs)[0] * natoms, self.dim_rot_mat_1, self.dim_rot_mat_1],
            )
            # (nframes x natoms) x naxis x naxis
            final_layer = final_layer + tf.transpose(final_layer, perm=[0, 2, 1])
        # (nframes x natoms) x naxis x 3(coord)
        final_layer = tf.matmul(final_layer, rot_mat_i)
        # (nframes x natoms) x 3(coord) x 3(coord)
        final_layer = tf.matmul(rot_mat_i, final_layer, transpose_a=True)
        # nframes x natoms x 3 x 3
        final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms, 3, 3])
        return final_layer

    def build(
        self,
        input_d: tf.Tensor,
        rot_mat: tf.Tensor,
        natoms: tf.Tensor,
        input_dict: Optional[dict] = None,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ):
        """Build the computational graph for fitting net.

        Parameters
        ----------
        input_d
            The input descriptor
        rot_mat
            The rotation matrix from the descriptor.
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        input_dict
            Additional dict for inputs.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        atomic_polar
            The atomic polarizability
        """
        if input_dict is None:
            input_dict = {}
        type_embedding = input_dict.get("type_embedding", None)
        atype = input_dict.get("atype", None)
        nframes = input_dict.get("nframes")
        start_index = 0

        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            self.t_constant_matrix = tf.get_variable(
                "t_constant_matrix",
                self.constant_matrix.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(self.constant_matrix),
            )

        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        rot_mat = tf.reshape(rot_mat, [-1, self.dim_rot_mat * natoms[0]])
        if nframes is None:
            nframes = tf.shape(inputs)[0]

        if self.mixed_types or type_embedding is not None:
            # keep old behavior
            self.mixed_types = True
            # nframes x nloc
            nloc_mask = tf.reshape(
                tf.tile(tf.repeat(self.sel_mask, natoms[2:]), [nframes]), [nframes, -1]
            )
            # nframes x nloc_masked
            scale = tf.reshape(
                tf.reshape(
                    tf.tile(tf.repeat(self.scale, natoms[2:]), [nframes]), [nframes, -1]
                )[nloc_mask],
                [nframes, -1],
            )
            if self.shift_diag:
                # nframes x nloc_masked
                constant_matrix = tf.reshape(
                    tf.reshape(
                        tf.tile(
                            tf.repeat(self.t_constant_matrix, natoms[2:]), [nframes]
                        ),
                        [nframes, -1],
                    )[nloc_mask],
                    [nframes, -1],
                )
            atype_nall = tf.reshape(atype, [-1, natoms[1]])
            # (nframes x nloc_masked)
            self.atype_nloc_masked = tf.reshape(
                tf.slice(atype_nall, [0, 0], [-1, natoms[0]])[nloc_mask], [-1]
            )  ## lammps will make error
            self.nloc_masked = tf.shape(
                tf.reshape(self.atype_nloc_masked, [nframes, -1])
            )[1]

        if type_embedding is not None:
            atype_embed = tf.nn.embedding_lookup(type_embedding, self.atype_nloc_masked)
        else:
            atype_embed = None

        self.atype_embed = atype_embed
        if atype_embed is not None:
            inputs = tf.reshape(
                tf.reshape(inputs, [nframes, natoms[0], self.dim_descrpt])[nloc_mask],
                [-1, self.dim_descrpt],
            )
            rot_mat = tf.reshape(
                tf.reshape(rot_mat, [nframes, natoms[0], self.dim_rot_mat])[nloc_mask],
                [-1, self.dim_rot_mat * self.nloc_masked],
            )
            atype_embed = tf.cast(atype_embed, self.fitting_precision)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat([inputs, atype_embed], axis=1)
            self.dim_descrpt = self.dim_descrpt + type_shape[1]

        if not self.mixed_types:
            count = 0
            outs_list = []
            for type_i in range(self.ntypes):
                if type_i not in self.sel_type:
                    start_index += natoms[2 + type_i]
                    continue
                final_layer = self._build_lower(
                    start_index,
                    natoms[2 + type_i],
                    inputs,
                    rot_mat,
                    suffix="_type_" + str(type_i) + suffix,
                    reuse=reuse,
                )
                # shift and scale
                sel_type_idx = self.sel_type.index(type_i)
                final_layer = final_layer * self.scale[sel_type_idx]
                final_layer = final_layer + tf.slice(
                    self.t_constant_matrix, [sel_type_idx], [1]
                ) * tf.eye(
                    3,
                    batch_shape=[tf.shape(inputs)[0], natoms[2 + type_i]],
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                )
                start_index += natoms[2 + type_i]

                # concat the results
                outs_list.append(final_layer)
                count += 1
            outs = tf.concat(outs_list, axis=1)
        else:
            inputs = tf.reshape(inputs, [-1, self.dim_descrpt * self.nloc_masked])
            final_layer = self._build_lower(
                0, self.nloc_masked, inputs, rot_mat, suffix=suffix, reuse=reuse
            )
            # shift and scale
            final_layer *= tf.expand_dims(tf.expand_dims(scale, -1), -1)
            if self.shift_diag:
                final_layer += tf.expand_dims(
                    tf.expand_dims(constant_matrix, -1), -1
                ) * tf.eye(3, batch_shape=[1, 1], dtype=GLOBAL_TF_FLOAT_PRECISION)
            outs = final_layer

        tf.summary.histogram("fitting_net_output", outs)
        return tf.reshape(outs, [-1])

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the fitting net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        self.fitting_net_variables = get_fitting_net_variables_from_graph_def(
            graph_def, suffix=suffix
        )
        if self.shift_diag:
            try:
                self.constant_matrix = get_tensor_by_name_from_graph(
                    graph, f"fitting_attr{suffix}/t_constant_matrix"
                )
            except GraphWithoutTensorError:
                warnings.warn(
                    "You are trying to read a model trained with shift_diag=True, but the mean of the diagonal terms of the polarizability is not stored in the graph. This will lead to wrong inference results. You may train your model with the latest DeePMD-kit to avoid this issue.",
                    stacklevel=2,
                )

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """Receive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
            The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec["output_prec"])

    def get_loss(self, loss: dict, lr) -> Loss:
        """Get the loss function."""
        return TensorLoss(
            loss,
            model=self,
            tensor_name="polar",
            tensor_size=9,
            label_name="polarizability",
        )

    def serialize(self, suffix: str) -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Fitting",
            "type": "polar",
            "@version": 4,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "embedding_width": self.dim_rot_mat_1,
            "mixed_types": self.mixed_types,
            "dim_out": 3,
            "neuron": self.n_neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "activation_function": self.activation_function_name,
            "precision": self.fitting_precision.name,
            "exclude_types": [],
            "fit_diag": self.fit_diag,
            "scale": list(self.scale),
            "shift_diag": self.shift_diag,
            "nets": self.serialize_network(
                ntypes=self.ntypes,
                ndim=0 if self.mixed_types else 1,
                in_dim=self.dim_descrpt,
                out_dim=self.dim_rot_mat_1,
                neuron=self.n_neuron,
                activation_function=self.activation_function_name,
                resnet_dt=self.resnet_dt,
                variables=self.fitting_net_variables,
                suffix=suffix,
            ),
            "@variables": {
                "fparam_avg": None,
                "fparam_inv_std": None,
                "aparam_avg": None,
                "aparam_inv_std": None,
                "case_embd": None,
                "scale": self.scale.reshape(-1, 1),
                "constant_matrix": self.constant_matrix.reshape(-1),
            },
            "type_map": self.type_map,
        }
        return data

    @classmethod
    def deserialize(cls, data: dict, suffix: str):
        """Deserialize the model.

        Parameters
        ----------
        data : dict
            The serialized data

        Returns
        -------
        Model
            The deserialized model
        """
        data = data.copy()
        check_version_compatibility(
            data.pop("@version", 1), 4, 1
        )  # to allow PT version.
        fitting = cls(**data)
        fitting.fitting_net_variables = cls.deserialize_network(
            data["nets"],
            suffix=suffix,
        )
        fitting.constant_matrix = data["@variables"]["constant_matrix"].ravel()
        return fitting


class GlobalPolarFittingSeA:
    r"""Fit the system polarizability with descriptor se_a.

    Parameters
    ----------
    descrpt : tf.Tensor
            The descriptor
    neuron : list[int]
            Number of neurons in each hidden layer of the fitting net
    resnet_dt : bool
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    sel_type : list[int]
            The atom types selected to have an atomic polarizability prediction
    fit_diag : bool
            Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.
    scale : list[float]
            The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
    diag_shift : list[float]
            The diagonal part of the polarizability matrix of type i will be shifted by diag_shift[i]. The shift operation is carried out after scale.
    seed : int
            Random seed for initializing the network parameters.
    activation_function : str
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision : str
            The precision of the embedding net parameters. Supported options are |PRECISION|
    """

    def __init__(
        self,
        descrpt: tf.Tensor,
        neuron: list[int] = [120, 120, 120],
        resnet_dt: bool = True,
        sel_type: Optional[list[int]] = None,
        fit_diag: bool = True,
        scale: Optional[list[float]] = None,
        diag_shift: Optional[list[float]] = None,
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = "default",
    ) -> None:
        """Constructor."""
        if not isinstance(descrpt, DescrptSeA):
            raise RuntimeError("GlobalPolarFittingSeA only supports DescrptSeA")
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        self.polar_fitting = PolarFittingSeA(
            descrpt,
            neuron,
            resnet_dt,
            sel_type,
            fit_diag,
            scale,
            diag_shift,
            seed,
            activation_function,
            precision,
        )

    def get_sel_type(self) -> int:
        """Get selected atom types."""
        return self.polar_fitting.get_sel_type()

    def get_out_size(self) -> int:
        """Get the output size. Should be 9."""
        return self.polar_fitting.get_out_size()

    def build(
        self,
        input_d,
        rot_mat,
        natoms,
        input_dict: Optional[dict] = None,
        reuse=None,
        suffix="",
    ) -> tf.Tensor:
        """Build the computational graph for fitting net.

        Parameters
        ----------
        input_d
            The input descriptor
        rot_mat
            The rotation matrix from the descriptor.
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        input_dict
            Additional dict for inputs.
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        polar
            The system polarizability
        """
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        outs = self.polar_fitting.build(
            input_d, rot_mat, natoms, input_dict, reuse, suffix
        )
        # nframes x natoms x 9
        outs = tf.reshape(outs, [tf.shape(inputs)[0], -1, 9])
        outs = tf.reduce_sum(outs, axis=1)
        tf.summary.histogram("fitting_net_output", outs)
        return tf.reshape(outs, [-1])

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """Init the fitting net variables with the given dict.

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope
        """
        self.polar_fitting.init_variables(
            graph=graph, graph_def=graph_def, suffix=suffix
        )

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """Receive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
            The mixed precision setting used in the embedding net
        """
        self.polar_fitting.enable_mixed_precision(mixed_prec)

    def get_loss(self, loss: dict, lr) -> Loss:
        """Get the loss function.

        Parameters
        ----------
        loss : dict
            the loss dict
        lr : LearningRateExp
            the learning rate

        Returns
        -------
        Loss
            the loss function
        """
        return TensorLoss(
            loss,
            model=self,
            tensor_name="global_polar",
            tensor_size=9,
            atomic=False,
            label_name="polarizability",
        )

    @property
    def input_requirement(self) -> list[DataRequirementItem]:
        """Return data requirements needed for the model input."""
        data_requirement = []
        if self.numb_fparam > 0:
            data_requirement.append(
                DataRequirementItem(
                    "fparam", self.numb_fparam, atomic=False, must=True, high_prec=False
                )
            )
        if self.numb_aparam > 0:
            data_requirement.append(
                DataRequirementItem(
                    "aparam", self.numb_aparam, atomic=True, must=True, high_prec=False
                )
            )
        return data_requirement

    def get_numb_fparam(self) -> int:
        """Get the number of frame parameters."""
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """Get the number of atomic parameters."""
        return self.numb_aparam
