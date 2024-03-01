# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    TYPE_CHECKING,
    List,
    Optional,
)

import numpy as np

from deepmd.tf.common import (
    add_data_requirement,
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    global_cvt_2_tf_float,
    tf,
)
from deepmd.tf.fit.fitting import (
    Fitting,
)
from deepmd.tf.infer import (
    DeepPotential,
)
from deepmd.tf.loss.ener import (
    EnerDipoleLoss,
    EnerSpinLoss,
    EnerStdLoss,
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
from deepmd.tf.utils.spin import (
    Spin,
)
from deepmd.utils.finetune import (
    change_energy_bias_lower,
)
from deepmd.utils.out_stat import (
    compute_stats_from_redu,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


@Fitting.register("ener")
class EnerFitting(Fitting):
    r"""Fitting the energy of the system. The force and the virial can also be trained.

    The potential energy :math:`E` is a fitting network function of the descriptor :math:`\mathcal{D}`:

    .. math::
        E(\mathcal{D}) = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)} \circ \mathcal{L}^{(0)}

    The first :math:`n` hidden layers :math:`\mathcal{L}^{(0)}, \cdots, \mathcal{L}^{(n-1)}` are given by

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b})

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}` are weights and biases, respectively,
    both of which are trainable if `trainable[i]` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    The output layer :math:`\mathcal{L}^{(n)}` is given by

    .. math::
        \mathbf{y}=\mathcal{L}^{(n)}(\mathbf{x};\mathbf{w},\mathbf{b})=
            \mathbf{x}^T\mathbf{w}+\mathbf{b}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_{n-1}}` is the input vector and :math:`\mathbf{y} \in \mathbb{R}`
    is the output scalar. :math:`\mathbf{w} \in \mathbb{R}^{N_{n-1}}` and
    :math:`\mathbf{b} \in \mathbb{R}` are weights and bias, respectively,
    both of which are trainable if `trainable[n]` is `True`.

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
    rcond
            The condition number for the regression of atomic energy.
    tot_ener_zero
            Force the total energy to zero. Useful for the charge fitting.
    trainable
            If the weights of fitting net are trainable.
            Suppose that we have :math:`N_l` hidden layers in the fitting net,
            this list is of length :math:`N_l + 1`, specifying if the hidden layers and the output layer are trainable.
    seed
            Random seed for initializing the network parameters.
    atom_ener
            Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.
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
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: List[int] = [120, 120, 120],
        resnet_dt: bool = True,
        numb_fparam: int = 0,
        numb_aparam: int = 0,
        rcond: Optional[float] = None,
        tot_ener_zero: bool = False,
        trainable: Optional[List[bool]] = None,
        seed: Optional[int] = None,
        atom_ener: List[float] = [],
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        layer_name: Optional[List[Optional[str]]] = None,
        use_aparam_as_mask: bool = False,
        spin: Optional[Spin] = None,
        **kwargs,
    ) -> None:
        """Constructor."""
        # model param
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.use_aparam_as_mask = use_aparam_as_mask
        # args = ()\
        #        .add('numb_fparam',      int,    default = 0)\
        #        .add('numb_aparam',      int,    default = 0)\
        #        .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
        #        .add('resnet_dt',        bool,   default = True)\
        #        .add('rcond',            float,  default = 1e-3) \
        #        .add('tot_ener_zero',    bool,   default = False) \
        #        .add('seed',             int)               \
        #        .add('atom_ener',        list,   default = [])\
        #        .add("activation_function", str,    default = "tanh")\
        #        .add("precision",           str, default = "default")\
        #        .add("trainable",        [list, bool], default = True)
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.rcond = rcond
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.spin = spin
        self.ntypes_spin = self.spin.get_ntypes_spin() if self.spin is not None else 0
        self.seed_shift = one_layer_rand_seed_shift()
        self.tot_ener_zero = tot_ener_zero
        self.activation_function_name = activation_function
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
        self.atom_ener = []
        self.atom_ener_v = atom_ener
        for at, ae in enumerate(atom_ener if atom_ener is not None else []):
            if ae is not None:
                self.atom_ener.append(
                    tf.constant(ae, GLOBAL_TF_FLOAT_PRECISION, name="atom_%d_ener" % at)
                )
            else:
                self.atom_ener.append(None)
        self.useBN = False
        self.bias_atom_e = np.zeros(self.ntypes, dtype=np.float64)
        # data requirement
        if self.numb_fparam > 0:
            add_data_requirement(
                "fparam", self.numb_fparam, atomic=False, must=True, high_prec=False
            )
        self.fparam_avg = None
        self.fparam_std = None
        self.fparam_inv_std = None
        if self.numb_aparam > 0:
            add_data_requirement(
                "aparam", self.numb_aparam, atomic=True, must=True, high_prec=False
            )
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

    def get_numb_fparam(self) -> int:
        """Get the number of frame parameters."""
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """Get the number of atomic parameters."""
        return self.numb_aparam

    def compute_output_stats(self, all_stat: dict, mixed_type: bool = False) -> None:
        """Compute the ouput statistics.

        Parameters
        ----------
        all_stat
            must have the following components:
            all_stat['energy'] of shape n_sys x n_batch x n_frame
            can be prepared by model.make_stat_input
        mixed_type
            Whether to perform the mixed_type mode.
            If True, the input data has the mixed_type format (see doc/model/train_se_atten.md),
            in which frames in a system may have different natoms_vec(s), with the same nloc.
        """
        self.bias_atom_e = self._compute_output_stats(
            all_stat, rcond=self.rcond, mixed_type=mixed_type
        )

    def _compute_output_stats(self, all_stat, rcond=1e-3, mixed_type=False):
        data = all_stat["energy"]
        # data[sys_idx][batch_idx][frame_idx]
        sys_ener = []
        for ss in range(len(data)):
            sys_data = []
            for ii in range(len(data[ss])):
                for jj in range(len(data[ss][ii])):
                    sys_data.append(data[ss][ii][jj])
            sys_data = np.concatenate(sys_data)
            sys_ener.append(np.average(sys_data))
        sys_ener = np.array(sys_ener)
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
        if len(self.atom_ener) > 0:
            # Atomic energies stats are incorrect if atomic energies are assigned.
            # In this situation, we directly use these assigned energies instead of computing stats.
            # This will make the loss decrease quickly
            assigned_atom_ener = np.array(
                [ee if ee is not None else np.nan for ee in self.atom_ener_v]
            )
        else:
            assigned_atom_ener = None
        energy_shift, _ = compute_stats_from_redu(
            sys_ener.reshape(-1, 1),
            sys_tynatom,
            assigned_bias=assigned_atom_ener,
            rcond=rcond,
        )
        return energy_shift.ravel()

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
        bias_atom_e=0.0,
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
            1,
            activation_fn=None,
            bavg=bias_atom_e,
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
        bias_atom_e = self.bias_atom_e
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

        ntypes_atom = self.ntypes - self.ntypes_spin
        if self.spin is not None:
            for type_i in range(ntypes_atom):
                if self.bias_atom_e.shape[0] != self.ntypes:
                    self.bias_atom_e = np.pad(
                        self.bias_atom_e,
                        (0, self.ntypes_spin),
                        "constant",
                        constant_values=(0, 0),
                    )
                    bias_atom_e = self.bias_atom_e
                if self.spin.use_spin[type_i]:
                    self.bias_atom_e[type_i] = (
                        self.bias_atom_e[type_i]
                        + self.bias_atom_e[type_i + ntypes_atom]
                    )
                else:
                    self.bias_atom_e[type_i] = self.bias_atom_e[type_i]
            self.bias_atom_e = self.bias_atom_e[:ntypes_atom]

        if nvnmd_cfg.enable:
            # fix the bug: CNN and QNN have different t_bias_atom_e.
            if "t_bias_atom_e" in nvnmd_cfg.weight.keys():
                self.bias_atom_e = nvnmd_cfg.weight["t_bias_atom_e"]

        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_dfparam = tf.constant(self.numb_fparam, name="dfparam", dtype=tf.int32)
            t_daparam = tf.constant(self.numb_aparam, name="daparam", dtype=tf.int32)
            self.t_bias_atom_e = tf.get_variable(
                "t_bias_atom_e",
                self.bias_atom_e.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(self.bias_atom_e),
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
        if len(self.atom_ener):
            # only for atom_ener
            nframes = input_dict.get("nframes")
            if nframes is not None:
                # like inputs, but we don't want to add a dependency on inputs
                inputs_zero = tf.zeros(
                    (nframes, natoms[0], self.dim_descrpt),
                    dtype=GLOBAL_TF_FLOAT_PRECISION,
                )
            else:
                inputs_zero = tf.zeros_like(inputs, dtype=GLOBAL_TF_FLOAT_PRECISION)

        if bias_atom_e is not None:
            assert len(bias_atom_e) == self.ntypes

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
        self.atype_nloc = tf.slice(
            atype_nall, [0, 0], [-1, natoms[0]]
        )  ## lammps will make error
        atype_filter = tf.cast(self.atype_nloc >= 0, GLOBAL_TF_FLOAT_PRECISION)
        self.atype_nloc = tf.reshape(self.atype_nloc, [-1])
        # prevent embedding_lookup error,
        # but the filter will be applied anyway
        self.atype_nloc = tf.clip_by_value(self.atype_nloc, 0, self.ntypes - 1)

        ## if spin is used
        if self.spin is not None:
            self.atype_nloc = tf.slice(
                atype_nall, [0, 0], [-1, tf.reduce_sum(natoms[2 : 2 + ntypes_atom])]
            )
            atype_filter = tf.cast(self.atype_nloc >= 0, GLOBAL_TF_FLOAT_PRECISION)
            self.atype_nloc = tf.reshape(self.atype_nloc, [-1])
        if (
            nvnmd_cfg.enable
            and nvnmd_cfg.quantize_descriptor
            and nvnmd_cfg.restore_descriptor
            and (nvnmd_cfg.version == 1)
        ):
            type_embedding = nvnmd_cfg.map["t_ebd"]
        if type_embedding is not None:
            atype_embed = tf.nn.embedding_lookup(type_embedding, self.atype_nloc)
        else:
            atype_embed = None

        self.atype_embed = atype_embed

        if atype_embed is None:
            start_index = 0
            outs_list = []
            for type_i in range(ntypes_atom):
                final_layer = self._build_lower(
                    start_index,
                    natoms[2 + type_i],
                    inputs,
                    fparam,
                    aparam,
                    bias_atom_e=0.0,
                    type_suffix="_type_" + str(type_i),
                    suffix=suffix,
                    reuse=reuse,
                )
                # concat the results
                if type_i < len(self.atom_ener) and self.atom_ener[type_i] is not None:
                    zero_layer = self._build_lower(
                        start_index,
                        natoms[2 + type_i],
                        inputs_zero,
                        fparam,
                        aparam,
                        bias_atom_e=0.0,
                        type_suffix="_type_" + str(type_i),
                        suffix=suffix,
                        reuse=True,
                    )
                    final_layer -= zero_layer
                final_layer = tf.reshape(
                    final_layer, [tf.shape(inputs)[0], natoms[2 + type_i]]
                )
                outs_list.append(final_layer)
                start_index += natoms[2 + type_i]
            # concat the results
            # concat once may be faster than multiple concat
            outs = tf.concat(outs_list, axis=1)
        # with type embedding
        else:
            atype_embed = tf.cast(atype_embed, GLOBAL_TF_FLOAT_PRECISION)
            type_shape = atype_embed.get_shape().as_list()
            inputs = tf.concat(
                [tf.reshape(inputs, [-1, self.dim_descrpt]), atype_embed], axis=1
            )
            original_dim_descrpt = self.dim_descrpt
            self.dim_descrpt = self.dim_descrpt + type_shape[1]
            inputs = tf.reshape(inputs, [-1, natoms[0], self.dim_descrpt])
            final_layer = self._build_lower(
                0,
                natoms[0],
                inputs,
                fparam,
                aparam,
                bias_atom_e=0.0,
                suffix=suffix,
                reuse=reuse,
            )
            if len(self.atom_ener):
                # remove contribution in vacuum
                inputs_zero = tf.concat(
                    [tf.reshape(inputs_zero, [-1, original_dim_descrpt]), atype_embed],
                    axis=1,
                )
                inputs_zero = tf.reshape(inputs_zero, [-1, natoms[0], self.dim_descrpt])
                zero_layer = self._build_lower(
                    0,
                    natoms[0],
                    inputs_zero,
                    fparam,
                    aparam,
                    bias_atom_e=0.0,
                    suffix=suffix,
                    reuse=True,
                )
                # atomic energy will be stored in `self.t_bias_atom_e` which is not trainable
                final_layer -= zero_layer
            outs = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[0]])
        # add bias
        self.atom_ener_before = outs * atype_filter
        # atomic bias energy from data statistics
        self.atom_bias_ener = tf.reshape(
            tf.nn.embedding_lookup(self.t_bias_atom_e, self.atype_nloc),
            [tf.shape(inputs)[0], tf.reduce_sum(natoms[2 : 2 + ntypes_atom])],
        )
        outs = outs + self.atom_bias_ener
        outs *= atype_filter
        self.atom_bias_ener *= atype_filter
        self.atom_ener_after = outs

        if self.tot_ener_zero:
            force_tot_ener = 0.0
            outs = tf.reshape(outs, [-1, tf.reduce_sum(natoms[2 : 2 + ntypes_atom])])
            outs_mean = tf.reshape(tf.reduce_mean(outs, axis=1), [-1, 1])
            outs_mean = outs_mean - tf.ones_like(
                outs_mean, dtype=GLOBAL_TF_FLOAT_PRECISION
            ) * (
                force_tot_ener
                / global_cvt_2_tf_float(tf.reduce_sum(natoms[2 : 2 + ntypes_atom]))
            )
            outs = outs - outs_mean
            outs = tf.reshape(outs, [-1])

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
        if self.layer_name is not None:
            # shared variables have no suffix
            shared_variables = get_fitting_net_variables_from_graph_def(
                graph_def, suffix=""
            )
            self.fitting_net_variables.update(shared_variables)
        if self.numb_fparam > 0:
            self.fparam_avg = get_tensor_by_name_from_graph(
                graph, "fitting_attr%s/t_fparam_avg" % suffix
            )
            self.fparam_inv_std = get_tensor_by_name_from_graph(
                graph, "fitting_attr%s/t_fparam_istd" % suffix
            )
        if self.numb_aparam > 0:
            self.aparam_avg = get_tensor_by_name_from_graph(
                graph, "fitting_attr%s/t_aparam_avg" % suffix
            )
            self.aparam_inv_std = get_tensor_by_name_from_graph(
                graph, "fitting_attr%s/t_aparam_istd" % suffix
            )
        try:
            self.bias_atom_e = get_tensor_by_name_from_graph(
                graph, "fitting_attr%s/t_bias_atom_e" % suffix
            )
        except GraphWithoutTensorError:
            # for compatibility, old models has no t_bias_atom_e
            pass

    def change_energy_bias(
        self,
        data,
        frozen_model,
        origin_type_map,
        full_type_map,
        bias_shift="delta",
        ntest=10,
    ) -> None:
        dp = None
        if bias_shift == "delta":
            # init model
            dp = DeepPotential(frozen_model)
        self.bias_atom_e = change_energy_bias_lower(
            data,
            dp,
            origin_type_map,
            full_type_map,
            self.bias_atom_e,
            bias_shift=bias_shift,
            ntest=ntest,
        )

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
            The loss function parameters.
        lr : LearningRateExp
            The learning rate.

        Returns
        -------
        Loss
            The loss function.
        """
        _loss_type = loss.pop("type", "ener")
        loss["starter_learning_rate"] = lr.start_lr()
        if _loss_type == "ener":
            return EnerStdLoss(**loss)
        elif _loss_type == "ener_dipole":
            return EnerDipoleLoss(**loss)
        elif _loss_type == "ener_spin":
            return EnerSpinLoss(**loss, use_spin=self.spin.use_spin)
        else:
            raise RuntimeError("unknown loss type")

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
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        fitting = cls(**data)
        fitting.fitting_net_variables = cls.deserialize_network(
            data["nets"],
            suffix=suffix,
        )
        fitting.bias_atom_e = data["@variables"]["bias_atom_e"]
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
            "type": "ener",
            "@version": 1,
            "var_name": "energy",
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            # very bad design: type embedding is not passed to the class
            # TODO: refactor the class
            "mixed_types": False,
            "dim_out": 1,
            "neuron": self.n_neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "rcond": self.rcond,
            "tot_ener_zero": self.tot_ener_zero,
            "trainable": self.trainable,
            "atom_ener": self.atom_ener_v,
            "activation_function": self.activation_function_name,
            "precision": self.fitting_precision.name,
            "layer_name": self.layer_name,
            "use_aparam_as_mask": self.use_aparam_as_mask,
            "spin": self.spin,
            "exclude_types": [],
            "nets": self.serialize_network(
                ntypes=self.ntypes,
                # TODO: consider type embeddings
                ndim=1,
                in_dim=self.dim_descrpt + self.numb_fparam + self.numb_aparam,
                neuron=self.n_neuron,
                activation_function=self.activation_function_name,
                resnet_dt=self.resnet_dt,
                variables=self.fitting_net_variables,
                suffix=suffix,
            ),
            "@variables": {
                "bias_atom_e": self.bias_atom_e,
                "fparam_avg": self.fparam_avg,
                "fparam_inv_std": self.fparam_inv_std,
                "aparam_avg": self.aparam_avg,
                "aparam_inv_std": self.aparam_inv_std,
            },
        }
        return data
