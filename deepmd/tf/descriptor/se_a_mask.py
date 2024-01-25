# SPDX-License-Identifier: LGPL-3.0-or-later
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.tf.common import (
    get_activation_func,
    get_precision,
)
from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.tf.utils.network import (
    embedding_net_rand_seed_shift,
)

from .descriptor import (
    Descriptor,
)
from .se_a import (
    DescrptSeA,
)


@Descriptor.register("se_a_mask")
class DescrptSeAMask(DescrptSeA):
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
        stripped_type_embedding: bool = False,
        **kwargs,
    ) -> None:
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
        self.stripped_type_embedding = stripped_type_embedding

        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt = self.ndescrpt_a
        self.useBN = False
        self.dstd = None
        self.davg = None
        self.rcut = -1.0  # Not used in se_a_mask
        self.compress = False
        self.embedding_net_variables = None
        self.mixed_prec = None
        self.place_holders = {}
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii]))  # like a mask
        self.nei_type = tf.constant(nei_type, dtype=tf.int32)

        avg_zero = np.zeros([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        std_ones = np.ones([self.ntypes, self.ndescrpt]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = "d_sea_mask_"
            for ii in ["coord", "box"]:
                self.place_holders[ii] = tf.placeholder(
                    GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_" + ii
                )
            self.place_holders["type"] = tf.placeholder(
                tf.int32, [None, None], name=name_pfx + "t_type"
            )
            self.place_holders["mask"] = tf.placeholder(
                tf.int32, [None, None], name=name_pfx + "t_aparam"
            )  # named aparam for inference compatibility in c++ interface.

            self.place_holders["natoms_vec"] = tf.placeholder(
                tf.int32, [self.ntypes + 2], name=name_pfx + "t_natoms"
            )  # Not used in se_a_mask. For compatibility with c++ interface.
            self.place_holders["default_mesh"] = tf.placeholder(
                tf.int32, [None], name=name_pfx + "t_mesh"
            )  # Not used in se_a_mask. For compatibility with c++ interface.

            self.stat_descrpt, descrpt_deriv, rij, nlist = op_module.descrpt_se_a_mask(
                self.place_holders["coord"],
                self.place_holders["type"],
                self.place_holders["mask"],
                self.place_holders["box"],
                self.place_holders["natoms_vec"],
                self.place_holders["default_mesh"],
            )
        self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)
        self.original_sel = None

    def get_rcut(self) -> float:
        """Returns the cutoff radius."""
        warnings.warn("The cutoff radius is not used for this descriptor")
        return -1.0

    def compute_input_stats(
        self,
        data_coord: list,
        data_box: list,
        data_atype: list,
        natoms_vec: list,
        mesh: list,
        input_dict: dict,
        **kwargs,
    ) -> None:
        """Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.

        Parameters
        ----------
        data_coord
            The coordinates. Can be generated by deepmd.tf.model.make_stat_input
        data_box
            The box. Can be generated by deepmd.tf.model.make_stat_input
        data_atype
            The atom types. Can be generated by deepmd.tf.model.make_stat_input
        natoms_vec
            The vector for the number of atoms of the system and different types of atoms. Can be generated by deepmd.tf.model.make_stat_input
        mesh
            The mesh for neighbor searching. Can be generated by deepmd.tf.model.make_stat_input
        input_dict
            Dictionary for additional input
        **kwargs
            Additional keyword arguments.
        """
        """
        TODO: Since not all input atoms are real in se_a_mask,
        statistics should be reimplemented for se_a_mask descriptor.
        """

        self.davg = None
        self.dstd = None

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: Dict[str, Any],
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> tf.Tensor:
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
        ``aparam'' shape is [nframes, nall]
        aparam[:, :] is the real/virtual sign for each atom.
        """
        aparam = input_dict["aparam"]
        with tf.variable_scope("fitting_attr" + suffix, reuse=reuse):
            t_aparam_nall = tf.constant(True, name="aparam_nall", dtype=tf.bool)
        self.mask = tf.cast(aparam, tf.int32)
        self.mask = tf.reshape(self.mask, [-1, natoms[1]])

        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt])
            if dstd is None:
                dstd = np.ones([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(
                self.rcut,
                name="rcut",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
            t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, name="ndescrpt", dtype=tf.int32)
            t_sel = tf.constant(self.sel_a, name="sel", dtype=tf.int32)
            """
            self.t_avg = tf.get_variable('t_avg',
                                         davg.shape,
                                         dtype = GLOBAL_TF_FLOAT_PRECISION,
                                         trainable = False,
                                         initializer = tf.constant_initializer(davg))
            self.t_std = tf.get_variable('t_std',
                                         dstd.shape,
                                         dtype = GLOBAL_TF_FLOAT_PRECISION,
                                         trainable = False,
                                         initializer = tf.constant_initializer(dstd))
            """

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box_ = tf.reshape(
            box_, [-1, 9]
        )  # Not used in se_a_mask descriptor. For compatibility in c++ inference.
        atype = tf.reshape(atype_, [-1, natoms[1]])

        (
            self.descrpt,
            self.descrpt_deriv,
            self.rij,
            self.nlist,
        ) = op_module.descrpt_se_a_mask(coord, atype, self.mask, box_, natoms, mesh)
        # only used when tensorboard was set as true
        tf.summary.histogram("descrpt", self.descrpt)
        tf.summary.histogram("rij", self.rij)
        tf.summary.histogram("nlist", self.nlist)

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
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

        # only used when tensorboard was set as true
        tf.summary.histogram("embedding_net_output", self.dout)
        return self.dout

    def prod_force_virial(
        self,
        atom_ener: tf.Tensor,
        natoms: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
        [net_deriv] = tf.gradients(atom_ener, self.descrpt_reshape)
        tf.summary.histogram("net_derivative", net_deriv)
        net_deriv_reshape = tf.reshape(net_deriv, [-1, natoms[0] * self.ndescrpt])
        force = op_module.prod_force_se_a_mask(
            net_deriv_reshape,
            self.descrpt_deriv,
            self.mask,
            self.nlist,
            total_atom_num=self.total_atom_num,
        )

        tf.summary.histogram("force", force)

        # Construct virial and atom virial tensors to avoid reshape errors in model/ener.py
        # They are not used in se_a_mask op
        virial = tf.zeros([1, 9], dtype=force.dtype)
        atom_virial = tf.zeros([1, natoms[1], 9], dtype=force.dtype)

        return force, virial, atom_virial

    @classmethod
    def update_sel(cls, global_jdata: dict, local_jdata: dict):
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        global_jdata : dict
            The global data, containing the training section
        local_jdata : dict
            The local data refer to the current class
        """
        return local_jdata
