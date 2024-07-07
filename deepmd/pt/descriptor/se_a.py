# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.utils.env_mat import (
    EnvMat,
)

from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)

from deepmd.pt.utils.graph import (
    get_extra_embedding_net_suffix,
)

from deepmd.pt.utils.tabulate import (
    DPTabulate,
)

from deepmd.pt.utils.compress import (
    make_data,
)

from deepmd.pt.utils.spin import (
    Spin,
)

from deepmd.pt.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_PT_FLOAT_PRECISION,
)

from deepmd.pt.common import (
    get_activation_func,
    get_np_precision,
    get_precision,
)

import torch
import torch.nn.functional as F


class DescrptSeA():
    r"""DeepPot-SE constructed from all information (both angular and radial) of
    atomic configurations. The embedding takes the distance between atoms as input.

    The descriptor :math:`\mathcal{D}^i \in \mathcal{R}^{M_1 \times M_2}` is given by [1]_

    .. math::
        \mathcal{D}^i = (\mathcal{G}^i)^T \mathcal{R}^i (\mathcal{R}^i)^T \mathcal{G}^i_<

    where :math:`\mathcal{R}^i \in \mathbb{R}^{N \times 4}` is the coordinate
    matrix, and each row of :math:`\mathcal{R}^i` can be constructed as follows

    .. math::
        (\mathcal{R}^i)_j = [
        \begin{array}{c}
            s(r_{ji}) & \frac{s(r_{ji})x_{ji}}{r_{ji}} & \frac{s(r_{ji})y_{ji}}{r_{ji}} & \frac{s(r_{ji})z_{ji}}{r_{ji}}
        \end{array}
        ]

    where :math:`\mathbf{R}_{ji}=\mathbf{R}_j-\mathbf{R}_i = (x_{ji}, y_{ji}, z_{ji})` is
    the relative coordinate and :math:`r_{ji}=\lVert \mathbf{R}_{ji} \lVert` is its norm.
    The switching function :math:`s(r)` is defined as:

    .. math::
        s(r)=
        \begin{cases}
        \frac{1}{r}, & r<r_s \\
        \frac{1}{r} \{ {(\frac{r - r_s}{ r_c - r_s})}^3 (-6 {(\frac{r - r_s}{ r_c - r_s})}^2 +15 \frac{r - r_s}{ r_c - r_s} -10) +1 \}, & r_s \leq r<r_c \\
        0, & r \geq r_c
        \end{cases}

    Each row of the embedding matrix  :math:`\mathcal{G}^i \in \mathbb{R}^{N \times M_1}` consists of outputs
    of a embedding network :math:`\mathcal{N}` of :math:`s(r_{ji})`:

    .. math::
        (\mathcal{G}^i)_j = \mathcal{N}(s(r_{ji}))

    :math:`\mathcal{G}^i_< \in \mathbb{R}^{N \times M_2}` takes first :math:`M_2` columns of
    :math:`\mathcal{G}^i`. The equation of embedding network :math:`\mathcal{N}` can be found at
    :meth:`deepmd.tf.utils.network.embedding_net`.

    Parameters
    ----------
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[int]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    neuron : list[int]
            Number of neurons in each hidden layers of the embedding net :math:`\mathcal{N}`
    axis_neuron
            Number of the axis neuron :math:`M_2` (number of columns of the sub-matrix of the embedding matrix)
    resnet_dt
            Time-step `dt` in the resnet construction:
            y = x + dt * \phi (Wx + b)
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    type_one_side
            Try to build N_types embedding nets. Otherwise, building N_types^2 embedding nets
    exclude_types : List[List[int]]
            The excluded pairs of types which have no interaction with each other.
            For example, `[[0, 1]]` means no interaction between type 0 and type 1.
    set_davg_zero
            Set the shift of embedding net input to zero.
    activation_function
            The activation function in the embedding net. Supported options are |ACTIVATION_FN|
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    env_protection: float
            Protection parameter to prevent division by zero errors during environment matrix calculations.
    type_map: List[str], Optional
            A list of strings. Give the name to each type of atoms.

    References
    ----------
    .. [1] Linfeng Zhang, Jiequn Han, Han Wang, Wissam A. Saidi, Roberto Car, and E. Weinan. 2018.
       End-to-end symmetry preserving inter-atomic potential energy model for finite and extended
       systems. In Proceedings of the 32nd International Conference on Neural Information Processing
       Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4441-4451.
    """
    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: List[int],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        seed: Optional[int] = None,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
        spin: Optional[Spin] = None,
        tebd_input_mode: str = "concat",
        type_map: Optional[List[str]] = None,  # to be compat with input
        env_protection: float = 0.0,  # not implement!!
        **kwargs,
    ) -> None:
        """Constructor."""
        if rcut < rcut_smth:
            raise RuntimeError(
                f"rcut_smth ({rcut_smth:f}) should be no more than rcut ({rcut:f})!"
            )
        if env_protection != 0.0:
            raise NotImplementedError("env_protection != 0.0 is not supported.")
        # to be compat with old option of `stripped_type_embedding`
        stripped_type_embedding = tebd_input_mode == "strip"
        self.sel_a = sel
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.filter_neuron = neuron
        self.n_axis_neuron = axis_neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.uniform_seed = uniform_seed
        # self.seed_shift = embedding_net_rand_seed_shift(self.filter_neuron)
        self.trainable = trainable
        self.compress_activation_fn = get_activation_func(activation_function)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.activation_function_name = activation_function
        self.filter_precision = get_precision(precision)
        self.filter_np_precision = get_np_precision(precision)
        self.orig_exclude_types = exclude_types
        self.exclude_types = set()
        self.env_protection = env_protection
        self.type_map = type_map
        for tt in exclude_types:
            assert len(tt) == 2
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero
        self.type_one_side = type_one_side
        self.spin = spin
        self.stripped_type_embedding = stripped_type_embedding
        self.extra_embedding_net_variables = None
        self.layer_size = len(neuron)

        # extend sel_a for spin system
        if self.spin is not None:
            self.ntypes_spin = self.spin.get_ntypes_spin()
            self.sel_a_spin = self.sel_a[: self.ntypes_spin]
            self.sel_a.extend(self.sel_a_spin)
        else:
            self.ntypes_spin = 0

        # descrpt config
        self.sel_r = [0 for ii in range(len(self.sel_a))]
        self.ntypes = len(self.sel_a)
        assert self.ntypes == len(self.sel_r)
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        self.useBN = False
        self.dstd = None
        self.davg = None
        self.compress = False
        self.embedding_net_variables = None
        self.mixed_prec = None
        self.place_holders = {}
        self.nei_type = np.repeat(np.arange(self.ntypes), self.sel_a)  # like a mask

        avg_zero = np.zeros([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        std_ones = np.ones([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )

    def enable_compression(
            self,
            min_nbor_dist: float,
            graph,
            graph_def,  
            table_extrapolate: float = 5,
            table_stride_1: float = 0.01,
            table_stride_2: float = 0.1,
            check_frequency: int = -1,
            suffix: str = "",
        ) -> None:
            """Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.

            Parameters
            ----------
            min_nbor_dist
                The nearest distance between atoms
            graph : torch.nn.Module
                The PyTorch model
            graph_def : Any
                The graph_def of the model (not directly used in PyTorch)
            table_extrapolate
                The scale of model extrapolation
            table_stride_1
                The uniform stride of the first table
            table_stride_2
                The uniform stride of the second table
            check_frequency
                The overflow check frequency
            suffix : str, optional
                The suffix of the scope
            """
            # do some checks before the model compression process
            assert (
                not self.filter_resnet_dt
            ), "Model compression error: descriptor resnet_dt must be false!"
            for tt in self.exclude_types:
                if (tt[0] not in range(self.ntypes)) or (tt[1] not in range(self.ntypes)):
                    raise RuntimeError(
                        f"exclude types {tt} must be within the number of atomic types {self.ntypes}!"
                    )
            if self.ntypes * self.ntypes - len(self.exclude_types) == 0:
                raise RuntimeError(
                    "empty embedding-net are not supported in model compression!"
                )

            if self.stripped_type_embedding:
                one_side_suffix = get_extra_embedding_net_suffix(type_one_side=True)
                two_side_suffix = get_extra_embedding_net_suffix(type_one_side=False)
                # Replace graph_def operations with PyTorch model-specific operations
                # Assuming get_pattern_nodes_from_graph_def and get_extra_embedding_net_variable functions are similarly implemented in PyTorch
                ret_two_side = get_pattern_nodes_from_graph_def(
                    graph_def, f"filter_type_all{suffix}/.+{two_side_suffix}"
                )
                ret_one_side = get_pattern_nodes_from_graph_def(
                    graph_def, f"filter_type_all{suffix}/.+{one_side_suffix}"
                )
                if len(ret_two_side) == 0 and len(ret_one_side) == 0:
                    raise RuntimeError(
                        "can not find variables of embedding net from graph_def, maybe it is not a compressible model."
                    )
                elif len(ret_one_side) != 0 and len(ret_two_side) != 0:
                    raise RuntimeError(
                        "both one side and two side embedding net varaibles are detected, it is a wrong model."
                    )
                elif len(ret_two_side) != 0:
                    self.final_type_embedding = get_two_side_type_embedding(self, graph)
                    self.matrix = get_extra_side_embedding_net_variable(
                        self, graph_def, two_side_suffix, "matrix", suffix
                    )
                    self.bias = get_extra_side_embedding_net_variable(
                        self, graph_def, two_side_suffix, "bias", suffix
                    )
                    self.extra_embedding = make_data(self, self.final_type_embedding)
                else:
                    self.final_type_embedding = get_type_embedding(self, graph)
                    self.matrix = get_extra_side_embedding_net_variable(
                        self, graph_def, one_side_suffix, "matrix", suffix
                    )
                    self.bias = get_extra_side_embedding_net_variable(
                        self, graph_def, one_side_suffix, "bias", suffix
                    )
                    self.extra_embedding = make_data(self, self.final_type_embedding)

            self.compress = True
            self.table = DPTabulate(
                self,
                self.filter_neuron,
                graph,
                self.type_one_side,
                self.exclude_types,
                self.compress_activation_fn,
                suffix=suffix,
            )
            self.table_config = [
                table_extrapolate,
                table_stride_1,
                table_stride_2,
                check_frequency,
            ]
            self.lower, self.upper = self.table.build(
                min_nbor_dist, table_extrapolate, table_stride_1, table_stride_2
            )

            # Assume these functions are implemented to get tensors by name from PyTorch model
            # self.davg = get_tensor_by_name_from_graph(graph, f"descrpt_attr{suffix}/t_avg")
            # self.dstd = get_tensor_by_name_from_graph(graph, f"descrpt_attr{suffix}/t_std")



    def build(
        self,
        coord_: torch.Tensor,
        atype_: torch.Tensor,
        natoms: torch.Tensor,
        box_: torch.Tensor,
        mesh: torch.Tensor,
        input_dict: dict,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> torch.Tensor:
        """Build the computational graph for the descriptor.

        Parameters
        ----------
        coord_
            The coordinate of atoms
        atype_
            The type of atoms
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        box_ : torch.Tensor
            The box of the system
        mesh
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        input_dict
            Dictionary for additional inputs
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor
            The output descriptor
        """
        davg = self.davg
        dstd = self.dstd

        if davg is None:
            davg = np.zeros([self.ntypes, self.ndescrpt])
        if dstd is None:
            dstd = np.ones([self.ntypes, self.ndescrpt])

        t_rcut = torch.tensor(
            np.max([self.rcut_r, self.rcut_a]),
            dtype=GLOBAL_PT_FLOAT_PRECISION,
        )
        t_ntypes = torch.tensor(self.ntypes, dtype=torch.int32)
        t_ndescrpt = torch.tensor(self.ndescrpt, dtype=torch.int32)
        t_sel = torch.tensor(self.sel_a, dtype=torch.int32)
        t_original_sel = torch.tensor(
            self.original_sel if self.original_sel is not None else self.sel_a,
            dtype=torch.int32,
        )

        self.t_avg = torch.nn.Parameter(
            torch.tensor(davg, dtype=GLOBAL_PT_FLOAT_PRECISION),
            requires_grad=False,
        )
        self.t_std = torch.nn.Parameter(
            torch.tensor(dstd, dtype=GLOBAL_PT_FLOAT_PRECISION),
            requires_grad=False,
        )

        with torch.no_grad():
            coord = coord_.view(-1, natoms[1] * 3)
            box = box_.view(-1, 9)
            atype = atype_.view(-1, natoms[1])
        self.atype = atype

        # Assuming build_op_descriptor and other methods are defined elsewhere
        op_descriptor = op_module.prod_env_mat_a
        self.descrpt, self.descrpt_deriv, self.rij, self.nlist = op_descriptor(
            coord,
            atype,
            natoms,
            box,
            mesh,
            self.t_avg,
            self.t_std,
            rcut_a=self.rcut_a,
            rcut_r=self.rcut_r,
            rcut_r_smth=self.rcut_r_smth,
            sel_a=self.sel_a,
            sel_r=self.sel_r,
        )

        nlist_t = self.nlist.view(-1) + 1
        atype_t = torch.cat([torch.tensor([self.ntypes]), self.atype.view(-1)], dim=0)
        self.nei_type_vec = torch.nn.functional.embedding(nlist_t, atype_t)

        # Assuming _identity_tensors and _pass_filter are defined elsewhere
        self.descrpt_reshape = self.descrpt.view(-1, self.ndescrpt)
        self._identity_tensors(suffix=suffix)

        self.dout, self.qmat = self._pass_filter(
            self.descrpt_reshape,
            atype,
            natoms,
            input_dict,
            suffix=suffix,
            reuse=reuse,
            trainable=self.trainable,
        )

        return self.dout


    def _pass_filter(
        self, inputs, atype, natoms, input_dict, reuse=None, suffix="", trainable=True
    ):
        if input_dict is not None:
            type_embedding = input_dict.get("type_embedding", None)
            if type_embedding is not None:
                self.use_tebd = True
        else:
            type_embedding = None
        if self.stripped_type_embedding and type_embedding is None:
            raise RuntimeError("type_embedding is required for se_a_tebd_v2 model.")
        
        start_index = 0
        inputs = inputs.view(-1, natoms[0], self.ndescrpt)
        output = []
        output_qmat = []
        
        if not self.type_one_side and type_embedding is None:
            for type_i in range(self.ntypes):
                inputs_i = inputs[:, start_index:start_index + natoms[2 + type_i], :]
                inputs_i = inputs_i.view(-1, self.ndescrpt)
                filter_name = "filtseler_type_" + str(type_i) + suffix
                layer, qmat = self._filter(
                    inputs_i,
                    type_i,
                    name=filter_name,
                    natoms=natoms,
                    reuse=reuse,
                    trainable=trainable,
                    activation_fn=self.filter_activation_fn,
                )
                layer = layer.view(inputs.size(0), natoms[2 + type_i], self.get_dim_out())
                qmat = qmat.view(inputs.size(0), natoms[2 + type_i], self.get_dim_rot_mat_1() * 3)
                output.append(layer)
                output_qmat.append(qmat)
                start_index += natoms[2 + type_i]
        else:
            inputs_i = inputs.view(-1, self.ndescrpt)
            type_i = -1
            # if nvnmd_cfg.enable and nvnmd_cfg.quantize_descriptor:
            #     inputs_i = descrpt2r4(inputs_i, natoms)
            
            self.atype_nloc = atype[:, :natoms[0]].view(-1)
            
            if len(self.exclude_types):
                mask = self.build_type_exclude_mask(
                    self.exclude_types,
                    self.ntypes,
                    self.sel_a,
                    self.ndescrpt,
                    self.atype_nloc,
                    inputs_i.size(0),
                )
                inputs_i *= mask

            layer, qmat = self._filter(
                inputs_i,
                type_i,
                name="filter_type_all" + suffix,
                natoms=natoms,
                reuse=reuse,
                trainable=trainable,
                activation_fn=self.filter_activation_fn,
                type_embedding=type_embedding,
            )
            layer = layer.view(inputs.size(0), natoms[0], self.get_dim_out())
            qmat = qmat.view(inputs.size(0), natoms[0], self.get_dim_rot_mat_1() * 3)
            output.append(layer)
            output_qmat.append(qmat)
        
        output = torch.cat(output, dim=1)
        output_qmat = torch.cat(output_qmat, dim=1)
        
        return output, output_qmat


    def _filter_lower(
            self,
            type_i,
            type_input,
            start_index,
            incrs_index,
            inputs,
            nframes,
            natoms,
            type_embedding=None,
            is_exclude=False,
            activation_fn=None,
            bavg=0.0,
            stddev=1.0,
            trainable=True,
            suffix="",
        ):
        """Input env matrix, returns R.G."""
        outputs_size = [1, *self.filter_neuron]
        # cut-out inputs
        # with natom x (nei_type_i x 4)
        inputs_i = inputs[:, start_index * 4:start_index * 4 + incrs_index * 4]
        shape_i = inputs_i.shape
        natom = inputs_i.shape[0]
        # reshape inputs
        # with (natom x nei_type_i) x 4
        inputs_reshape = inputs_i.view(-1, 4)
        # with (natom x nei_type_i) x 1
        xyz_scatter = inputs_reshape[:, :1]

        if type_embedding is not None:
            if self.stripped_type_embedding:
                if self.type_one_side:
                    extra_embedding_index = self.nei_type_vec
                else:
                    padding_ntypes = type_embedding.shape[0]
                    atype_expand = self.atype_nloc.view(-1, 1)
                    idx_i = atype_expand * padding_ntypes
                    idx_j = self.nei_type_vec.view(-1, self.nnei)
                    idx = idx_i + idx_j
                    index_of_two_side = idx.view(-1)
                    extra_embedding_index = index_of_two_side
            else:
                xyz_scatter = self._concat_type_embedding(xyz_scatter, nframes, natoms, type_embedding)
                if self.compress:
                    raise RuntimeError(
                        "compression of type embedded descriptor is not supported when tebd_input_mode is not set to 'strip'"
                    )
        
        if self.compress and (not is_exclude):
            if self.stripped_type_embedding:
                net_output = F.embedding(extra_embedding_index, self.extra_embedding)
                net = "filter_net"
                info = [
                    self.lower[net],
                    self.upper[net],
                    self.upper[net] * self.table_config[0],
                    self.table_config[1],
                    self.table_config[2],
                    self.table_config[3],
                ]
                return torch.ops.deepmd.tabulate_fusion_se_atten(
                    self.table.data[net].to(dtype=self.filter_precision),
                    info,
                    xyz_scatter,
                    inputs_i.view(natom, shape_i[1] // 4, 4),
                    net_output,
                    last_layer_size=outputs_size[-1],
                    is_sorted=False,
                )
            else:
                net = f"filter_{'-1' if self.type_one_side else str(type_input)}_net_{type_i}"
                info = [
                    self.lower[net],
                    self.upper[net],
                    self.upper[net] * self.table_config[0],
                    self.table_config[1],
                    self.table_config[2],
                    self.table_config[3],
                ]
                return torch.ops.deepmd.tabulate_fusion_se_a(
                    self.table.data[net].to(dtype=self.filter_precision),
                    info,
                    xyz_scatter,
                    inputs_i.view(natom, shape_i[1] // 4, 4),
                    last_layer_size=outputs_size[-1],
                )


    def _filter(
            self,
            inputs,
            type_input,
            natoms,
            type_embedding=None,
            activation_fn=F.tanh,
            stddev=1.0,
            bavg=0.0,
            name="linear",
            reuse=None,
            trainable=True,
        ):
        nframes = inputs.view(-1, natoms[0], self.ndescrpt).shape[0]
        # natom x (nei x 4)
        shape = list(inputs.shape)
        outputs_size = [1, *self.filter_neuron]
        outputs_size_2 = self.n_axis_neuron
        all_excluded = all(
            (type_input, type_i) in self.exclude_types for type_i in range(self.ntypes)
        )
        if all_excluded:
            # all types are excluded so result and qmat should be zeros
            # we can safaly return a zero matrix...
            # See also https://stackoverflow.com/a/34725458/9567349
            # result: natom x outputs_size x outputs_size_2
            # qmat: natom x outputs_size x 3
            natom = inputs.shape[0]
            result = torch.zeros((natom, outputs_size_2, outputs_size[-1]), dtype=GLOBAL_PT_FLOAT_PRECISION)
            qmat = torch.zeros((natom, outputs_size[-1], 3), dtype=GLOBAL_PT_FLOAT_PRECISION)
            return result, qmat

        start_index = 0
        type_i = 0
        # natom x 4 x outputs_size
        if type_embedding is None:
            rets = []
            for type_i in range(self.ntypes):
                ret = self._filter_lower(
                    type_i,
                    type_input,
                    start_index,
                    self.sel_a[type_i],
                    inputs,
                    nframes,
                    natoms,
                    type_embedding=type_embedding,
                    is_exclude=(type_input, type_i) in self.exclude_types,
                    activation_fn=activation_fn,
                    stddev=stddev,
                    bavg=bavg,
                    trainable=trainable,
                    suffix="_" + str(type_i),
                )
                if (type_input, type_i) not in self.exclude_types:
                    # add zero is meaningless; skip
                    rets.append(ret)
                start_index += self.sel_a[type_i]
            # faster to use add_n than multiple add
            xyz_scatter_1 = torch.stack(rets).sum(0)
        else:
            xyz_scatter_1 = self._filter_lower(
                type_i,
                type_input,
                start_index,
                np.cumsum(self.sel_a)[-1],
                inputs,
                nframes,
                natoms,
                type_embedding=type_embedding,
                is_exclude=False,
                activation_fn=activation_fn,
                stddev=stddev,
                bavg=bavg,
                trainable=trainable,
            )

        if self.original_sel is None:
            # shape[1] = nnei * 4
            nnei = shape[1] // 4
        else:
            nnei = torch.tensor(sum(self.original_sel), dtype=torch.int32)

        xyz_scatter_1 = xyz_scatter_1 / nnei
        # natom x 4 x outputs_size_2
        xyz_scatter_2 = xyz_scatter_1[:, :, :outputs_size_2]
        # # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = xyz_scatter_1[:, 1:4, :]
        # natom x outputs_size_1 x 3
        qmat = qmat.permute(0, 2, 1)
        # natom x outputs_size x outputs_size_2
        result = torch.matmul(xyz_scatter_1.permute(0, 2, 1), xyz_scatter_2)
        # natom x (outputs_size x outputs_size_2)
        result = result.view(-1, outputs_size_2 * outputs_size[-1])

        return result, qmat
