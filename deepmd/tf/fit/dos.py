# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
    Optional,
)

import numpy as np

from deepmd.tf.common import (
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.loss.dos import (
    DOSLoss,
)
from deepmd.tf.loss.loss import (
    Loss,
)
from deepmd.tf.nvnmd.fit.ener import (
    one_layer_nvnmd,
)
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.tf.utils.graph import (
    get_fitting_net_variables_from_graph_def,
    get_tensor_by_name_from_graph,
)
from deepmd.tf.utils.network import one_layer as one_layer_deepmd
from deepmd.tf.utils.network import (
    one_layer_rand_seed_shift,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)


@Fitting.register("dos")
class DOSFitting(Fitting):
    r"""Fitting the density of states (DOS) of the system.
    The energy should be shifted by the fermi level.

    Parameters
    ----------
    ntypes
            The ntypes of the descrptor :math:`\mathcal{D}`
    dim_descrpt
            The dimension of the descrptor :math:`\mathcal{D}`
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    resnet_dt
            Time-step `dt` in the resnet construction:
            :math:`y = x + dt * \phi (Wx + b)`
    numb_fparam
            Number of frame parameter
    numb_aparam
            Number of atomic parameter
    ! numb_dos (added)
            Number of gridpoints on which the DOS is evaluated (NEDOS in VASP)
    rcond
            The condition number for the regression of atomic energy.
    trainable
            If the weights of fitting net are trainable.
            Suppose that we have :math:`N_l` hidden layers in the fitting net,
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    seed
            Random seed for initializing the network parameters.
    activation_function
            The activation function :math:`\boldsymbol{\phi}` in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    layer_name : list[Optional[str]], optional
            The name of the each layer. If two layers, either in the same fitting or different fittings,
            have the same name, they will share the same neural network parameters.
    use_aparam_as_mask: bool, optional
            If True, the atomic parameters will be used as a mask that determines the atom is real/virtual.
            And the aparam will not be used as the atomic parameters for embedding.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types, otherwise use
        different fitting nets for different atom types.
    type_map: List[str], Optional
            A list of strings. Give the name to each type of atoms.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: List[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        numb_dos: int = 300,
        rcond: Optional[float] = None,
        trainable: Optional[List[bool]] = None,
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        layer_name: Optional[List[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        mixed_types: bool = False,
        type_map: Optional[List[str]] = None,  # to be compat with input
        **kwargs,
    ) -> None:
        """Constructor."""
        # model param
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.use_aparam_as_mask = use_aparam_as_mask

        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam

        self.numb_dos = numb_dos

        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.rcond = rcond
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = one_layer_rand_seed_shift()
        self.activation_function = activation_function
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.trainable = trainable
        if self.trainable is None:
            self.trainable = [True for ii in range(len(self.n_neuron) + 1)]
        if isinstance(self.trainable, bool):
            self.trainable = [self.trainable] * (len(self.n_neuron) + 1)
        assert (
            len(self.trainable) == len(self.n_neuron) + 1
        ), "length of trainable should be that of n_neuron + 1"

        self.useBN = False
        self.bias_dos = np.zeros((self.ntypes, self.numb_dos), dtype=np.float64)
        self.fparam_avg = None
        self.fparam_std = None
        self.fparam_inv_std = None
        self.aparam_avg = None
        self.aparam_std = None
        self.aparam_inv_std = None

        self.fitting_net_variables = None
        self.mixed_prec = None
        self.layer_name = layer_name
        if self.layer_name is not None:
            assert isinstance(self.layer_name, list), "layer_name should be a list"
            assert (
                len(self.layer_name) == len(self.n_neuron) + 1
            ), "length of layer_name should be that of n_neuron + 1"
        self.mixed_types = mixed_types
        self.type_map = type_map

    def get_numb_fparam(self) -> int:
        """Get the number of frame parameters."""
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """Get the number of atomic parameters."""
        return self.numb_aparam

    def get_numb_dos(self) -> int:
        """Get the number of gridpoints in energy space."""
        return self.numb_dos

    # not used
    def compute_output_stats(self, all_stat: dict, mixed_type: bool = False) -> None:
        """Compute the ouput statistics.

        Parameters
        ----------
        all_stat
            must have the following components:
            all_stat['dos'] of shape n_sys x n_batch x n_frame x numb_dos
            can be prepared by model.make_stat_input
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.
        """
        self.bias_dos = self._compute_output_stats(
            all_stat, rcond=self.rcond, mixed_type=mixed_type
        )

    def _compute_output_stats(self, all_stat, rcond=1e-3, mixed_type=False):
        data = all_stat["dos"]
        # data[sys_idx][batch_idx][frame_idx]
        sys_dos = []
        for ss in range(len(data)):
            sys_data = []
            for ii in range(len(data[ss])):
                for jj in range(len(data[ss][ii])):
                    sys_data.append(data[ss][ii][jj])
            sys_data = np.concatenate(sys_data).reshape(-1, self.numb_dos)
            sys_dos.append(np.average(sys_data, axis=0))
        sys_dos = np.array(sys_dos).reshape(-1, self.numb_dos)
        sys_tynatom = []
        if mixed_type:
            data = all_stat["real_natoms_vec"]
            nsys = len(data)
            for ss in range(len(data)):
                tmp_tynatom = []
                for ii in range(len(data[ss])):
                    for jj in range(len(data[ss][ii])):
                        tmp_tynatom.append(data[ss][ii][jj].astype(np.float64))
                tmp_tynatom = np.average(np.array(tmp_tynatom), axis=0)
                sys_tynatom.append(tmp_tynatom)
        else:
            data = all_stat["natoms_vec"]
            nsys = len(data)
            for ss in range(len(data)):
                sys_tynatom.append(data[ss][0].astype(np.float64))
        sys_tynatom = np.array(sys_tynatom)
        sys_tynatom = np.reshape(sys_tynatom, [nsys, -1])
        sys_tynatom = sys_tynatom[:, 2:]

        dos_shift, _ = compute_stats_from_redu(
            sys_dos,
            sys_tynatom,
            rcond=rcond,
        )

        return dos_shift

    def compute_input_stats(self, all_stat: dict, protection: float = 1e-2) -> None:
        """Compute the input statistics.

        Parameters
        ----------
        all_stat
            if numb_fparam > 0 must have all_stat['fparam']
            if numb_aparam > 0 must have all_stat['aparam']
            can be prepared by model.make_stat_input
        protection
            Divided-by-zero protection
        """
        # stat fparam
        if self.numb_fparam > 0:
            cat_data = np.concatenate(all_stat["fparam"], axis=0)
            cat_data = np.reshape(cat_data, [-1, self.numb_fparam])
            self.fparam_avg = np.average(cat_data, axis=0)
            self.fparam_std = np.std(cat_data, axis=0)
            for ii in range(self.fparam_std.size):
                if self.fparam_std[ii] < protection:
                    self.fparam_std[ii] = protection
            self.fparam_inv_std = 1.0 / self.fparam_std
        # stat aparam
        if self.numb_aparam > 0:
            sys_sumv = []
            sys_sumv2 = []
            sys_sumn = []
            for ss_ in all_stat["aparam"]:
                ss = np.reshape(ss_, [-1, self.numb_aparam])
                sys_sumv.append(np.sum(ss, axis=0))
                sys_sumv2.append(np.sum(np.multiply(ss, ss), axis=0))
                sys_sumn.append(ss.shape[0])
            sumv = np.sum(sys_sumv, axis=0)
            sumv2 = np.sum(sys_sumv2, axis=0)
            sumn = np.sum(sys_sumn)
            self.aparam_avg = (sumv) / sumn
            self.aparam_std = self._compute_std(sumv2, sumv, sumn)
            for ii in range(self.aparam_std.size):
                if self.aparam_std[ii] < protection:
                    self.aparam_std[ii] = protection
            self.aparam_inv_std = 1.0 / self.aparam_std

    def _compute_std(self, sumv2, sumv, sumn):
        return np.sqrt(sumv2 / sumn - np.multiply(sumv / sumn, sumv / sumn))

    @cast_precision
    def _build_lower(
        self,
        start_index,
        natoms,
        inputs,
        fparam=None,
        aparam=None,
        bias_dos=0.0,
        type_suffix="",
        suffix="",
        reuse=None,
    ):
        # cut-out inputs
        inputs_i = tf.slice(inputs, [0, start_index, 0], [-1, natoms, -1])
        inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
        layer = inputs_i
        if fparam is not None:
            ext_fparam = tf.tile(fparam, [1, natoms])
            ext_fparam = tf.reshape(ext_fparam, [-1, self.numb_fparam])
            ext_fparam = tf.cast(ext_fparam, self.fitting_precision)
            layer = tf.concat([layer, ext_fparam], axis=1)
        if aparam is not None:
            ext_aparam = tf.slice(
                aparam,
                [0, start_index * self.numb_aparam],
                [-1, natoms * self.numb_aparam],
            )
            ext_aparam = tf.reshape(ext_aparam, [-1, self.numb_aparam])
            ext_aparam = tf.cast(ext_aparam, self.fitting_precision)
            layer = tf.concat([layer, ext_aparam], axis=1)

        if nvnmd_cfg.enable:
            one_layer = one_layer_nvnmd
        else:
            one_layer = one_layer_deepmd
        for ii in range(0, len(self.n_neuron)):
            if self.layer_name is not None and self.layer_name[ii] is not None:
                layer_suffix = "share_" + self.layer_name[ii] + type_suffix
                layer_reuse = tf.AUTO_REUSE
            else:
                layer_suffix = "layer_" + str(ii) + type_suffix + suffix
                layer_reuse = reuse
            if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii - 1]:
                layer += one_layer(
                    layer,
                    self.n_neuron[ii],
                    name=layer_suffix,
                    reuse=layer_reuse,
                    seed=self.seed,
                    use_timestep=self.resnet_dt,
                    activation_fn=self.fitting_activation_fn,
                    precision=self.fitting_precision,
                    trainable=self.trainable[ii],
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.fitting_net_variables,
                    mixed_prec=self.mixed_prec,
                )
            else:
                layer = one_layer(
                    layer,
                    self.n_neuron[ii],
                    name=layer_suffix,
                    reuse=layer_reuse,
                    seed=self.seed,
                    activation_fn=self.fitting_activation_fn,
                    precision=self.fitting_precision,
                    trainable=self.trainable[ii],
                    uniform_seed=self.uniform_seed,
                    initial_variables=self.fitting_net_variables,
                    mixed_prec=self.mixed_prec,
                )
            if (not self.uniform_seed) and (self.seed is not None):
                self.seed += self.seed_shift
        if self.layer_name is not None and self.layer_name[-1] is not None:
            layer_suffix = "share_" + self.layer_name[-1] + type_suffix
            layer_reuse = tf.AUTO_REUSE
        else:
            layer_suffix = "final_layer" + type_suffix + suffix
            layer_reuse = reuse
        final_layer = one_layer(
            layer,
            self.numb_dos,  # TODO: output a vector
            activation_fn=None,
            bavg=bias_dos,
            name=layer_suffix,
            reuse=layer_reuse,
            seed=self.seed,
            precision=self.fitting_precision,
            trainable=self.trainable[-1],
            uniform_seed=self.uniform_seed,
            initial_variables=self.fitting_net_variables,
            mixed_prec=self.mixed_prec,
            final_layer=True,
        )
        if (not self.uniform_seed) and (self.seed is not None):
            self.seed += self.seed_shift

        return final_layer

    def build(
        self,
        inputs: tf.Tensor,
        natoms: tf.Tensor,
        input_dict: Optional[dict] = None,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
        """Build the computational graph for fitting net.

        Parameters
        ----------
        inputs
            The input descriptor
        input_dict
            Additional dict for inputs.
            if numb_fparam > 0, should have input_dict['fparam']
            if numb_aparam > 0, should have input_dict['aparam']
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        ener
            The system energy
        """
        if input_dict is None:
            input_dict = {}
        bias_dos = self.bias_dos
        type_embedding = input_dict.get("type_embedding", None)
        atype = input_dict.get("atype", None)
        if self.numb_fparam > 0:
            if self.fparam_avg is None:
                self.fparam_avg = 0.0
            if self.fparam_inv_std is None:
                self.fparam_inv_std = 1.0
        if self.numb_aparam > 0:
            if self.aparam_avg is None:
                self.aparam_avg = 0.0
            if self.aparam_inv_std is None:
                self.aparam_inv_std = 1.0

        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_dfparam = tf.constant(self.numb_fparam, name="dfparam", dtype=tf.int32)
            t_daparam = tf.constant(self.numb_aparam, name="daparam", dtype=tf.int32)
            t_numb_dos = tf.constant(self.numb_dos, name="numb_dos", dtype=tf.int32)

            self.t_bias_dos = tf.get_variable(
                "t_bias_dos",
                self.bias_dos.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(self.bias_dos),
            )
            if self.numb_fparam > 0:
                t_fparam_avg = tf.get_variable(
                    "t_fparam_avg",
                    self.numb_fparam,
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                    trainable=False,
                    initializer=tf.constant_initializer(self.fparam_avg),
                )
                t_fparam_istd = tf.get_variable(
                    "t_fparam_istd",
                    self.numb_fparam,
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                    trainable=False,
                    initializer=tf.constant_initializer(self.fparam_inv_std),
                )
            if self.numb_aparam > 0:
                t_aparam_avg = tf.get_variable(
                    "t_aparam_avg",
                    self.numb_aparam,
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                    trainable=False,
                    initializer=tf.constant_initializer(self.aparam_avg),
                )
                t_aparam_istd = tf.get_variable(
                    "t_aparam_istd",
                    self.numb_aparam,
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                    trainable=False,
                    initializer=tf.constant_initializer(self.aparam_inv_std),
                )

        inputs = tf.reshape(inputs, [-1, natoms[0], self.dim_descrpt])

        if bias_dos is not None:
            assert len(bias_dos) == self.ntypes

        fparam = None
        if self.numb_fparam > 0:
            fparam = input_dict["fparam"]
            fparam = tf.reshape(fparam, [-1, self.numb_fparam])
            fparam = (fparam - t_fparam_avg) * t_fparam_istd

        aparam = None
        if not self.use_aparam_as_mask:
            if self.numb_aparam > 0:
                aparam = input_dict["aparam"]
                aparam = tf.reshape(aparam, [-1, self.numb_aparam])
                aparam = (aparam - t_aparam_avg) * t_aparam_istd
                aparam = tf.reshape(aparam, [-1, self.numb_aparam * natoms[0]])

        atype_nall = tf.reshape(atype, [-1, natoms[1]])
        self.atype_nloc = tf.reshape(
            tf.slice(atype_nall, [0, 0], [-1, natoms[0]]), [-1]
        )  ## lammps will make error
        if type_embedding is not None:
            # keep old behavior
            self.mixed_types = True
            atype_embed = tf.nn.embedding_lookup(type_embedding, self.atype_nloc)
        else:
            atype_embed = None

        self.atype_embed = atype_embed
        if atype_embed is not None:
            atype_embed = tf.cast(atype_embed, GLOBAL_TF_FLOAT_PRECISION)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat(
                [tf.reshape(inputs, [-1, self.dim_descrpt]), atype_embed], axis=1
            )
            self.dim_descrpt = self.dim_descrpt + type_shape[1]

        if not self.mixed_types:
            start_index = 0
            outs_list = []
            for type_i in range(self.ntypes):
                final_layer = self._build_lower(
                    start_index,
                    natoms[2 + type_i],
                    inputs,
                    fparam,
                    aparam,
                    bias_dos=0.0,
                    type_suffix="_type_" + str(type_i),
                    suffix=suffix,
                    reuse=reuse,
                )

                final_layer = tf.reshape(
                    final_layer,
                    [
                        tf.shape(inputs)[0],
                        natoms[2 + type_i],
                        self.numb_dos,
                    ],
                )
                outs_list.append(final_layer)
                start_index += natoms[2 + type_i]
            # concat the results
            # concat once may be faster than multiple concat
            outs = tf.concat(outs_list, axis=1)
        # with type embedding
        else:
            inputs = tf.reshape(inputs, [-1, natoms[0], self.dim_descrpt])
            final_layer = self._build_lower(
                0,
                natoms[0],
                inputs,
                fparam,
                aparam,
                bias_dos=0.0,
                suffix=suffix,
                reuse=reuse,
            )

            outs = tf.reshape(
                final_layer,
                [tf.shape(inputs)[0], natoms[0], self.numb_dos],
            )
        # add bias
        # self.atom_ener_before = outs
        # self.add_type = tf.reshape(
        #     tf.nn.embedding_lookup(self.t_bias_dos, self.atype_nloc),
        #     [tf.shape(inputs)[0], natoms[0]],
        # )
        # outs = outs + self.add_type
        # self.atom_ener_after = outs

        tf.summary.histogram("fitting_net_output", outs)
        return outs

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
        if self.layer_name is not None:
            # shared variables have no suffix
            shared_variables = get_fitting_net_variables_from_graph_def(
                graph_def, suffix=""
            )
            self.fitting_net_variables.update(shared_variables)
        if self.numb_fparam > 0:
            self.fparam_avg = get_tensor_by_name_from_graph(
                graph, f"fitting_attr{suffix}/t_fparam_avg"
            )
            self.fparam_inv_std = get_tensor_by_name_from_graph(
                graph, f"fitting_attr{suffix}/t_fparam_istd"
            )
        if self.numb_aparam > 0:
            self.aparam_avg = get_tensor_by_name_from_graph(
                graph, f"fitting_attr{suffix}/t_aparam_avg"
            )
            self.aparam_inv_std = get_tensor_by_name_from_graph(
                graph, f"fitting_attr{suffix}/t_aparam_istd"
            )
        try:
            self.bias_dos = get_tensor_by_name_from_graph(
                graph, f"fitting_attr{suffix}/t_bias_dos"
            )
        except GraphWithoutTensorError:
            # for compatibility, old models has no t_bias_dos
            pass

    def enable_mixed_precision(self, mixed_prec: Optional[dict] = None) -> None:
        """Reveive the mixed precision setting.

        Parameters
        ----------
        mixed_prec
            The mixed precision setting used in the embedding net
        """
        self.mixed_prec = mixed_prec
        self.fitting_precision = get_precision(mixed_prec["output_prec"])

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
        return DOSLoss(
            **loss, starter_learning_rate=lr.start_lr(), numb_dos=self.get_numb_dos()
        )

    @classmethod
    def deserialize(cls, data: dict, suffix: str = ""):
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
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        data["numb_dos"] = data.pop("dim_out")
        fitting = cls(**data)
        fitting.fitting_net_variables = cls.deserialize_network(
            data["nets"],
            suffix=suffix,
        )
        fitting.bias_dos = data["@variables"]["bias_atom_e"]
        if fitting.numb_fparam > 0:
            fitting.fparam_avg = data["@variables"]["fparam_avg"]
            fitting.fparam_inv_std = data["@variables"]["fparam_inv_std"]
        if fitting.numb_aparam > 0:
            fitting.aparam_avg = data["@variables"]["aparam_avg"]
            fitting.aparam_inv_std = data["@variables"]["aparam_inv_std"]
        return fitting

    def serialize(self, suffix: str = "") -> dict:
        """Serialize the model.

        Returns
        -------
        dict
            The serialized data
        """
        data = {
            "@class": "Fitting",
            "type": "dos",
            "@version": 2,
            "var_name": "dos",
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "mixed_types": self.mixed_types,
            "dim_out": self.numb_dos,
            "neuron": self.n_neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "rcond": self.rcond,
            "trainable": self.trainable,
            "activation_function": self.activation_function,
            "precision": self.fitting_precision.name,
            "exclude_types": [],
            "nets": self.serialize_network(
                ntypes=self.ntypes,
                ndim=0 if self.mixed_types else 1,
                in_dim=self.dim_descrpt + self.numb_fparam + self.numb_aparam,
                out_dim=self.numb_dos,
                neuron=self.n_neuron,
                activation_function=self.activation_function,
                resnet_dt=self.resnet_dt,
                variables=self.fitting_net_variables,
                suffix=suffix,
            ),
            "@variables": {
                "bias_atom_e": self.bias_dos,
                "fparam_avg": self.fparam_avg,
                "fparam_inv_std": self.fparam_inv_std,
                "aparam_avg": self.aparam_avg,
                "aparam_inv_std": self.aparam_inv_std,
            },
            "type_map": self.type_map,
        }
        return data

    @property
    def input_requirement(self) -> List[DataRequirementItem]:
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
