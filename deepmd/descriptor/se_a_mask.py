import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.common import (
    get_activation_func,
    get_precision,
)
from deepmd.env import (
    GLOBAL_PD_FLOAT_PRECISION,
    op_module,
    paddle,
)
from deepmd.utils.network import (
    EmbeddingNet,
    embedding_net_rand_seed_shift,
)


class DescrptSeAMask(paddle.nn.Layer):
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
    :meth:`deepmd.utils.network.embedding_net`.
    Specially for descriptor se_a_mask is a concise implementation of se_a.
    The difference is that se_a_mask only considered a non-pbc system.
    And accept a mask matrix to indicate the atom i in frame j is a real atom or not.
    (1 means real atom, 0 means ghost atom)
    Thus se_a_mask can accept a variable number of atoms in a frame.

    Parameters
    ----------
    sel : list[str]
            sel[i] specifies the maxmum number of type i atoms in the neighbor list.
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
    activation_function
            The activation function in the embedding net. Supported options are {0}
    precision
            The precision of the embedding net parameters. Supported options are {1}
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    References
    ----------
    .. [1] Linfeng Zhang, Jiequn Han, Han Wang, Wissam A. Saidi, Roberto Car, and E. Weinan. 2018.
       End-to-end symmetry preserving inter-atomic potential energy model for finite and extended
       systems. In Proceedings of the 32nd International Conference on Neural Information Processing
       Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4441-4451.
    """

    def __init__(
        self,
        sel: List[str],
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        trainable: bool = True,
        type_one_side: bool = False,
        exclude_types: List[List[int]] = [],
        seed: Optional[int] = None,
        activation_function: str = "tanh",
        precision: str = "default",
        uniform_seed: bool = False,
    ) -> None:
        super().__init__()
        """Constructor."""
        self.sel_a = sel
        self.total_atom_num = np.cumsum(self.sel_a)[-1]
        self.ntypes = len(self.sel_a)
        self.filter_neuron = neuron
        self.n_axis_neuron = axis_neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = embedding_net_rand_seed_shift(self.filter_neuron)
        self.trainable = trainable
        self.compress_activation_fn = get_activation_func(activation_function)
        self.filter_activation_fn = get_activation_func(activation_function)
        self.filter_precision = get_precision(precision)
        self.exclude_types = set()
        for tt in exclude_types:
            assert len(tt) == 2
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = False
        self.type_one_side = type_one_side
        # descrpt config. Not used in se_a_mask
        self.sel_r = [0 for ii in range(len(self.sel_a))]
        self.ntypes = len(self.sel_a)
        assert self.ntypes == len(self.sel_r)
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei = self.nnei_a

        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt = self.ndescrpt_a
        self.useBN = False
        self.dstd = None
        self.davg = None
        self.rcut = -1.0  # Not used in se_a_mask
        self.compress = False
        self.embedding_net_variables = None
        self.mixed_prec = None
        # self.place_holders = {}
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii]))  # like a mask
        # self.nei_type = tf.constant(nei_type, dtype=tf.int32)
        # self.nei_type = paddle.to_tensor(nei_type, dtype="int32")
        self.register_buffer(
            "buffer_ntypes_spin", paddle.to_tensor(nei_type, dtype="int32")
        )

        nets = []
        for type_input in range(self.ntypes):
            layer = []
            for type_i in range(self.ntypes):
                layer.append(
                    EmbeddingNet(
                        self.filter_neuron,
                        self.filter_precision,
                        self.filter_activation_fn,
                        self.filter_resnet_dt,
                        self.seed,
                        self.trainable,
                        name="filter_type_" + str(type_input) + str(type_i),
                    )
                )
            nets.append(paddle.nn.LayerList(layer))

        self.embedding_nets = paddle.nn.LayerList(nets)
        self.original_sel = None

    def get_rcut(self) -> float:
        """Returns the cutoff radius."""
        warnings.warn("The cutoff radius is not used for this descriptor")
        return -1.0

    def get_ntypes(self) -> int:
        """Returns the number of atom types."""
        return self.ntypes

    def get_dim_out(self) -> int:
        """Returns the output dimension of this descriptor."""
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_dim_rot_mat_1(self) -> int:
        """Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3."""
        return self.filter_neuron[-1]

    def compute_input_stats(
        self,
        data_coord: list,
        data_box: list,
        data_atype: list,
        natoms_vec: list,
        mesh: list,
        input_dict: dict,
    ) -> None:
        """Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.

        Parameters
        ----------
        data_coord
            The coordinates. Can be generated by deepmd.model.make_stat_input
        data_box
            The box. Can be generated by deepmd.model.make_stat_input
        data_atype
            The atom types. Can be generated by deepmd.model.make_stat_input
        natoms_vec
            The vector for the number of atoms of the system and different types of atoms. Can be generated by deepmd.model.make_stat_input
        mesh
            The mesh for neighbor searching. Can be generated by deepmd.model.make_stat_input
        input_dict
            Dictionary for additional input
        """
        """
        TODO: Since not all input atoms are real in se_a_mask,
        statistics should be reimplemented for se_a_mask descriptor.
        """

        self.davg = None
        self.dstd = None

    def forward(
        self,
        coord_: paddle.Tensor,
        atype_: paddle.Tensor,
        natoms: paddle.Tensor,
        box_: paddle.Tensor,
        mesh: paddle.Tensor,
        input_dict: Dict[str, Any],
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> paddle.Tensor:
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
        box_ : tf.Tensor
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

        """
        ``aparam'' shape is [nframes, natoms]
        aparam[:, :] is the real/virtual sign for each atom.
        """
        aparam = input_dict["aparam"]

        self.mask = paddle.cast(aparam, paddle.int32)
        self.mask = paddle.reshape(self.mask, [-1, natoms[1]])
        # with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
        if davg is None:
            davg = np.zeros([self.ntypes, self.ndescrpt])
        if dstd is None:
            dstd = np.ones([self.ntypes, self.ndescrpt])
            # t_rcut = tf.constant(
            #     self.rcut,
            #     name="rcut",
            #     dtype=GLOBAL_TF_FLOAT_PRECISION,
            # )
            # t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            # t_ndescrpt = tf.constant(self.ndescrpt, name="ndescrpt", dtype=tf.int32)
            # t_sel = tf.constant(self.sel_a, name="sel", dtype=tf.int32)
            # """
            # self.t_avg = tf.get_variable('t_avg',
            #                              davg.shape,
            #                              dtype = GLOBAL_TF_FLOAT_PRECISION,
            #                              trainable = False,
            #                              initializer = tf.constant_initializer(davg))
            # self.t_std = tf.get_variable('t_std',
            #                              dstd.shape,
            #                              dtype = GLOBAL_TF_FLOAT_PRECISION,
            #                              trainable = False,
            #                              initializer = tf.constant_initializer(dstd))
            # """

        coord = paddle.reshape(coord_, [-1, natoms[1] * 3])

        box_ = paddle.reshape(
            box_, [-1, 9]
        )  # Not used in se_a_mask descriptor. For compatibility in c++ inference.

        atype = paddle.reshape(atype_, [-1, natoms[1]])

        coord = paddle.to_tensor(coord, place="cpu")
        atype = paddle.to_tensor(atype, place="cpu")
        self.mask = paddle.to_tensor(self.mask, place="cpu")
        box_ = paddle.to_tensor(box_, place="cpu")
        natoms = paddle.to_tensor(natoms, place="cpu")
        mesh = paddle.to_tensor(mesh, place="cpu")

        (
            self.descrpt,
            self.descrpt_deriv,
            self.rij,
            self.nlist,
        ) = op_module.descrpt_se_a_mask(coord, atype, self.mask, box_, natoms, mesh)
        # only used when tensorboard was set as true
        # tf.summary.histogram("descrpt", self.descrpt)
        # tf.summary.histogram("rij", self.rij)
        # tf.summary.histogram("nlist", self.nlist)

        self.descrpt_reshape = paddle.reshape(self.descrpt, [-1, self.ndescrpt])
        # self._identity_tensors(suffix=suffix)
        self.descrpt_reshape.stop_gradient = False

        self.dout, self.qmat = self._pass_filter(
            self.descrpt_reshape,
            atype,
            natoms,
            input_dict,
            suffix=suffix,
            reuse=reuse,
            trainable=self.trainable,
        )

        # only used when tensorboard was set as true
        # tf.summary.histogram("embedding_net_output", self.dout)
        return self.dout

    def prod_force_virial(
        self,
        atom_ener: paddle.Tensor,
        natoms: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        """Compute force and virial.

        Parameters
        ----------
        atom_ener
            The atomic energy
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms

        Returns
        -------
        force
            The force on atoms
        virial
            None for se_a_mask op
        atom_virial
            None for se_a_mask op
        """
        net_deriv = paddle.grad(atom_ener, self.descrpt_reshape, create_graph=True)[0]
        # tf.summary.histogram("net_derivative", net_deriv)
        net_deriv_reshape = paddle.reshape(net_deriv, [-1, natoms[0] * self.ndescrpt])
        net_deriv_reshape = paddle.to_tensor(net_deriv_reshape, place="cpu")
        force = op_module.prod_force_se_a_mask(
            net_deriv_reshape,
            self.descrpt_deriv,
            self.mask,
            self.nlist,
            total_atom_num=self.total_atom_num,
        )

        # tf.summary.histogram("force", force)

        # Construct virial and atom virial tensors to avoid reshape errors in model/ener.py
        # They are not used in se_a_mask op
        virial = paddle.zeros([1, 9], dtype=force.dtype)
        atom_virial = paddle.zeros([1, natoms[1], 9], dtype=force.dtype)

        return force, virial, atom_virial

    def _pass_filter(
        self, inputs, atype, natoms, input_dict, reuse=None, suffix="", trainable=True
    ):
        """pass_filter.

        Parameters
        ----------
        inputs : paddle.Tensor
            Inputs tensor.
        atype : paddle.Tensor
            Atom type Tensor.
        natoms : paddle.Tensor
            Number of atoms vector
        input_dict : Dict[str, paddle.Tensor]
            Input data dict.
        reuse : bool, optional
            Whether reuse variables. Defaults to None.
        suffix : str, optional
            Variable suffix. Defaults to "".
        trainable : bool, optional
            Whether make subnetwork traninable. Defaults to True.

        Returns
        -------
        Tuple[Tensor, Tensor]: output: [1, all_atom, M1*M2], output_qmat: [1, all_atom, M1*3]
        """
        if input_dict is not None:
            type_embedding = input_dict.get("type_embedding", None)
        else:
            type_embedding = None
        start_index = 0
        inputs = paddle.reshape(inputs, [-1, int(natoms[0].item()), int(self.ndescrpt)])
        output = []
        output_qmat = []
        if not self.type_one_side and type_embedding is None:
            for type_i in range(self.ntypes):
                inputs_i = paddle.slice(
                    inputs,
                    [0, 1, 2],
                    [0, start_index, 0],
                    [
                        inputs.shape[0],
                        start_index + natoms[2 + type_i].item(),
                        inputs.shape[2],
                    ],
                )
                inputs_i = paddle.reshape(inputs_i, [-1, self.ndescrpt])
                filter_name = "filter_type_" + str(type_i) + suffix
                layer, qmat = self._filter(
                    inputs_i,
                    type_i,
                    name=filter_name,
                    natoms=natoms,
                    reuse=reuse,
                    trainable=trainable,
                    activation_fn=self.filter_activation_fn,
                )
                layer = paddle.reshape(
                    layer, [inputs.shape[0], natoms[2 + type_i], self.get_dim_out()]
                )
                qmat = paddle.reshape(
                    qmat,
                    [
                        inputs.shape[0],
                        natoms[2 + type_i],
                        self.get_dim_rot_mat_1() * 3,
                    ],
                )
                output.append(layer)
                output_qmat.append(qmat)
                start_index += natoms[2 + type_i].item()
        else:
            raise NotImplementedError()
            # This branch will not be excecuted at current
            # inputs_i = inputs
            # inputs_i = paddle.reshape(inputs_i, [-1, self.ndescrpt])
            # type_i = -1
            # # if nvnmd_cfg.enable and nvnmd_cfg.quantize_descriptor:
            # #     inputs_i = descrpt2r4(inputs_i, natoms)
            # if len(self.exclude_types):
            #     atype_nloc = paddle.reshape(
            #         paddle.slice(atype, [0, 1], [0, 0], [atype.shape[0], natoms[0]]),
            #         [-1],
            #     )  # when nloc != nall, pass nloc to mask
            #     mask = self.build_type_exclude_mask(
            #         self.exclude_types,
            #         self.ntypes,
            #         self.sel_a,
            #         self.ndescrpt,
            #         atype_nloc,
            #         paddle.shape(inputs_i)[0],
            #     )
            #     inputs_i *= mask

            # layer, qmat = self._filter(
            #     inputs_i,
            #     type_i,
            #     name="filter_type_all" + suffix,
            #     natoms=natoms,
            #     reuse=reuse,
            #     trainable=trainable,
            #     activation_fn=self.filter_activation_fn,
            #     type_embedding=type_embedding,
            # )
            # layer = paddle.reshape(
            #     layer, [inputs.shape[0], natoms[0], self.get_dim_out()]
            # )
            # qmat = paddle.reshape(
            #     qmat, [inputs.shape[0], natoms[0], self.get_dim_rot_mat_1() * 3]
            # )
            # output.append(layer)
            # output_qmat.append(qmat)
        output = paddle.concat(output, axis=1)
        output_qmat = paddle.concat(output_qmat, axis=1)
        return output, output_qmat

    def _filter_lower(
        self,
        type_i: int,  # inner-loop
        type_input: int,  # outer-loop
        start_index: int,
        incrs_index: int,
        inputs: paddle.Tensor,
        nframes: int,
        natoms: int,
        type_embedding=None,
        is_exclude=False,
    ):
        """Input env matrix, returns R.G."""
        outputs_size = [1, *self.filter_neuron]
        # cut-out inputs
        # with natom x (nei_type_i x 4)
        inputs_i = paddle.slice(
            inputs,
            [0, 1],
            [0, start_index * 4],
            [inputs.shape[0], start_index * 4 + incrs_index * 4],
        )

        shape_i = inputs_i.shape
        natom = inputs_i.shape[0]

        # with (natom x nei_type_i) x 4
        inputs_reshape = paddle.reshape(inputs_i, [-1, 4])
        # with (natom x nei_type_i) x 1
        xyz_scatter = paddle.reshape(
            paddle.slice(inputs_reshape, [0, 1], [0, 0], [inputs_reshape.shape[0], 1]),
            [-1, 1],
        )

        if type_embedding is not None:
            xyz_scatter = self._concat_type_embedding(
                xyz_scatter, nframes, natoms, type_embedding
            )  #
            if self.compress:
                raise RuntimeError(
                    "compression of type embedded descriptor is not supported at the moment"
                )
        # natom x 4 x outputs_size
        if self.compress and (not is_exclude):
            if self.type_one_side:
                net = "filter_-1_net_" + str(type_i)
            else:
                net = "filter_" + str(type_input) + "_net_" + str(type_i)
            info = [
                self.lower[net],
                self.upper[net],
                self.upper[net] * self.table_config[0],
                self.table_config[1],
                self.table_config[2],
                self.table_config[3],
            ]
            return op_module.tabulate_fusion_se_a(
                paddle.cast(self.table.data[net], self.filter_precision),
                info,
                xyz_scatter,
                paddle.reshape(inputs_i, [natom, shape_i[1] // 4, 4]),
                last_layer_size=outputs_size[-1],
            )
        else:
            if not is_exclude:
                # excuted this branch
                xyz_scatter_out = self.embedding_nets[type_input][type_i](xyz_scatter)
                if (not self.uniform_seed) and (self.seed is not None):
                    self.seed += self.seed_shift
            else:
                # we can safely return the final xyz_scatter filled with zero directly
                return paddle.cast(
                    paddle.fill((natom, 4, outputs_size[-1]), 0.0),
                    self.filter_precision,
                )
            # natom x nei_type_i x out_size
            xyz_scatter_out = paddle.reshape(
                xyz_scatter_out, (-1, shape_i[1] // 4, outputs_size[-1])
            )  # (natom x nei_type_i) x 100 ==> natom x nei_type_i x 100
            # When using paddle.reshape(inputs_i, [-1, shape_i[1]//4, 4]) below
            # [588 24] -> [588 6 4] correct
            # but if sel is zero
            # [588 0] -> [147 0 4] incorrect; the correct one is [588 0 4]
            # So we need to explicitly assign the shape to paddle.shape(inputs_i)[0] instead of -1
            # natom x 4 x outputs_size

            return paddle.matmul(
                paddle.reshape(inputs_i, [natom, shape_i[1] // 4, 4]),
                xyz_scatter_out,
                transpose_x=True,
            )

    def _filter(
        self,
        inputs: paddle.Tensor,
        type_input: int,
        natoms,
        type_embedding=None,
        activation_fn=paddle.nn.functional.tanh,
        stddev=1.0,
        bavg=0.0,
        name="linear",
        reuse=None,
        trainable=True,
    ):
        """_filter.

        Parameters
        ----------
        inputs : paddle.Tensor
            Inputs tensor.
        type_input : int
            Type of input.
        natoms : paddle.Tensor
            Number of atoms, a vector.
        type_embedding : paddle.Tensor
            Type embedding. Defaults to None.
        activation_fn : Callable
            Activation function. Defaults to paddle.nn.functional.tanh.
        stddev : float, optional
            Stddev for parameters initialization. Defaults to 1.0.
        bavg : float, optional
            Bavg for parameters initialization . Defaults to 0.0.
        name : str, optional
            Name for subnetwork. Defaults to "linear".
        reuse : bool, optional
            Whether reuse variables. Defaults to None.
        trainable : bool, optional
            Whether make subnetwork trainable. Defaults to True.

        Returns
        -------
        Tuple[Tensor, Tensor]: result: [64/128, M1*M2], qmat: [64/128, M1, 3]
        """
        # NOTE: code below is annotated as nframes computation is wrong
        # nframes = paddle.shape(paddle.reshape(inputs, [-1, natoms[0], self.ndescrpt]))[0]

        nframes = 1
        # natom x (nei x 4)
        shape = inputs.shape
        outputs_size = [1, *self.filter_neuron]
        outputs_size_2 = self.n_axis_neuron  # 16
        all_excluded = all(
            # FIXME: the bracket '[]' is needed when convert to static model, will be
            # removed when fixed.
            [  # noqa
                (type_input, type_i) in self.exclude_types  #  set()
                for type_i in range(self.ntypes)
            ]
        )
        if all_excluded:
            # all types are excluded so result and qmat should be zeros
            # we can safaly return a zero matrix...
            # See also https://stackoverflow.com/a/34725458/9567349
            # result: natom x outputs_size x outputs_size_2
            # qmat: natom x outputs_size x 3
            natom = paddle.shape(inputs)[0]
            result = paddle.cast(
                paddle.full((natom, outputs_size_2, outputs_size[-1]), 0.0),
                GLOBAL_PD_FLOAT_PRECISION,
            )
            qmat = paddle.cast(
                paddle.full((natom, outputs_size[-1], 3), 0.0),
                GLOBAL_PD_FLOAT_PRECISION,
            )
            return result, qmat

        # with tf.variable_scope(name, reuse=reuse):
        start_index = 0
        type_i = 0
        # natom x 4 x outputs_size
        if type_embedding is None:
            rets = []
            # execute this branch
            for type_i in range(self.ntypes):
                ret = self._filter_lower(
                    type_i,
                    type_input,
                    start_index,
                    self.sel_a[type_i],  # 46(O)/92(H)
                    inputs,
                    nframes,
                    natoms,
                    type_embedding=type_embedding,
                    is_exclude=(type_input, type_i) in self.exclude_types,
                )
                if (type_input, type_i) not in self.exclude_types:
                    # add zero is meaningless; skip
                    rets.append(ret)
                start_index += self.sel_a[type_i]
            # faster to use accumulate_n than multiple add
            xyz_scatter_1 = paddle.add_n(rets)
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
            )
        # natom x nei x outputs_size
        # xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
        # natom x nei x 4
        # inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
        # natom x 4 x outputs_size
        # xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
        if self.original_sel is None:
            # shape[1] = nnei * 4
            nnei = shape[1] / 4
        else:
            nnei = paddle.cast(
                paddle.to_tensor(
                    np.sum(self.original_sel),
                    dtype=paddle.int32,
                    stop_gradient=True,
                ),
                self.filter_precision,
            )
        xyz_scatter_1 = xyz_scatter_1 / nnei
        # natom x 4 x outputs_size_2
        xyz_scatter_2 = paddle.slice(
            xyz_scatter_1,
            [0, 1, 2],
            [0, 0, 0],
            [xyz_scatter_1.shape[0], xyz_scatter_1.shape[1], outputs_size_2],
        )
        # natom x 3 x outputs_size_2
        # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
        # natom x 3 x outputs_size_1
        qmat = paddle.slice(
            xyz_scatter_1,
            [0, 1, 2],
            [0, 1, 0],
            [xyz_scatter_1.shape[0], 1 + 3, xyz_scatter_1.shape[2]],
        )
        # natom x outputs_size_1 x 3
        qmat = paddle.transpose(qmat, perm=[0, 2, 1])  # [64/128, M1, 3]
        # natom x outputs_size x outputs_size_2
        result = paddle.matmul(xyz_scatter_1, xyz_scatter_2, transpose_x=True)
        # natom x (outputs_size x outputs_size_2)
        result = paddle.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])

        return result, qmat
