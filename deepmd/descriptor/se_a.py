import math
import numpy as np
from typing import Tuple, List, Dict, Any

from deepmd.env import tf
from deepmd.common import get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter, get_np_precision
from deepmd.utils.argcheck import list_to_doc
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.utils.network import embedding_net, embedding_net_rand_seed_shift
from deepmd.utils.tabulate import DPTabulate
from deepmd.utils.type_embed import embed_atom_type
from deepmd.utils.sess import run_sess
from deepmd.utils.graph import load_graph_def, get_tensor_by_name_from_graph, get_embedding_net_variables
from .descriptor import Descriptor
from .se import DescrptSe

@Descriptor.register("se_e2_a")
@Descriptor.register("se_a")
class DescrptSeA (DescrptSe):
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

    :math:`\mathcal{G}^i_< \in \mathbb{R}^{N \times M_2}` takes first :math:`M_2`$` columns of
    :math:`\mathcal{G}^i`$`. The equation of embedding network :math:`\mathcal{N}` can be found at
    :meth:`deepmd.utils.network.embedding_net`.

    Parameters
    ----------
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[str]
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
       Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 4441–4451.
    """
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  rcut: float,
                  rcut_smth: float,
                  sel: List[str],
                  neuron: List[int] = [24,48,96],
                  axis_neuron: int = 8,
                  resnet_dt: bool = False,
                  trainable: bool = True,
                  seed: int = None,
                  type_one_side: bool = True,
                  exclude_types: List[List[int]] = [],
                  set_davg_zero: bool = False,
                  activation_function: str = 'tanh',
                  precision: str = 'default',
                  uniform_seed: bool = False
    ) -> None:
        """
        Constructor
        """
        self.sel_a = sel
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
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
        self.filter_np_precision = get_np_precision(precision)
        self.exclude_types = set()
        for tt in exclude_types:
            assert(len(tt) == 2)
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero
        self.type_one_side = type_one_side
        if self.type_one_side and len(exclude_types) != 0:
            raise RuntimeError('"type_one_side" is not compatible with "exclude_types"')

        # descrpt config
        self.sel_r = [ 0 for ii in range(len(self.sel_a)) ]
        self.ntypes = len(self.sel_a)
        assert(self.ntypes == len(self.sel_r))
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
        self.place_holders = {}
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii])) # like a mask 
        self.nei_type = tf.constant(nei_type, dtype = tf.int32)

        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = 'd_sea_'
            for ii in ['coord', 'box']:
                self.place_holders[ii] = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name = name_pfx+'t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name=name_pfx+'t_type')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name=name_pfx+'t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name=name_pfx+'t_mesh')
            self.stat_descrpt, descrpt_deriv, rij, nlist \
                = op_module.prod_env_mat_a(self.place_holders['coord'],
                                         self.place_holders['type'],
                                         self.place_holders['natoms_vec'],
                                         self.place_holders['box'],
                                         self.place_holders['default_mesh'],
                                         tf.constant(avg_zero),
                                         tf.constant(std_ones),
                                         rcut_a = self.rcut_a,
                                         rcut_r = self.rcut_r,
                                         rcut_r_smth = self.rcut_r_smth,
                                         sel_a = self.sel_a,
                                         sel_r = self.sel_r)
        self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)


    def get_rcut (self) -> float:
        """
        Returns the cut-off radius
        """
        return self.rcut_r

    def get_ntypes (self) -> int:
        """
        Returns the number of atom types
        """
        return self.ntypes

    def get_dim_out (self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_dim_rot_mat_1 (self) -> int:
        """
        Returns the first dimension of the rotation matrix. The rotation is of shape dim_1 x 3
        """
        return self.filter_neuron[-1]

    def get_nlist (self) -> Tuple[tf.Tensor, tf.Tensor, List[int], List[int]]:
        """
        Returns
        -------
        nlist
                Neighbor list
        rij
                The relative distance between the neighbor and the center atom.
        sel_a
                The number of neighbors with full information
        sel_r
                The number of neighbors with only radial information
        """
        return self.nlist, self.rij, self.sel_a, self.sel_r

    def compute_input_stats (self,
                             data_coord : list, 
                             data_box : list, 
                             data_atype : list, 
                             natoms_vec : list,
                             mesh : list, 
                             input_dict : dict
    ) -> None :
        """
        Compute the statisitcs (avg and std) of the training data. The input will be normalized by the statistics.
        
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
        all_davg = []
        all_dstd = []
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc,bb,tt,nn,mm in zip(data_coord,data_box,data_atype,natoms_vec,mesh) :
                sysr,sysr2,sysa,sysa2,sysn \
                    = self._compute_dstats_sys_smth(cc,bb,tt,nn,mm)
                sumr.append(sysr)
                suma.append(sysa)
                sumn.append(sysn)
                sumr2.append(sysr2)
                suma2.append(sysa2)
            sumr = np.sum(sumr, axis = 0)
            suma = np.sum(suma, axis = 0)
            sumn = np.sum(sumn, axis = 0)
            sumr2 = np.sum(sumr2, axis = 0)
            suma2 = np.sum(suma2, axis = 0)
            for type_i in range(self.ntypes) :
                davgunit = [sumr[type_i]/(sumn[type_i]+1e-15), 0, 0, 0]
                dstdunit = [self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i])
                            ]
                davg = np.tile(davgunit, self.ndescrpt // 4)
                dstd = np.tile(dstdunit, self.ndescrpt // 4)
                all_davg.append(davg)
                all_dstd.append(dstd)

        if not self.set_davg_zero:
            self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)

    def enable_compression(self,
                           min_nbor_dist : float,
                           model_file : str = 'frozon_model.pb',
                           table_extrapolate : float = 5,
                           table_stride_1 : float = 0.01,
                           table_stride_2 : float = 0.1,
                           check_frequency : int = -1,
                           suffix : str = "",
    ) -> None:
        """
        Reveive the statisitcs (distance, max_nbor_size and env_mat_range) of the training data.
        
        Parameters
        ----------
        min_nbor_dist
                The nearest distance between atoms
        model_file
                The original frozen model, which will be compressed by the program
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
        assert (
            not self.filter_resnet_dt
        ), "Model compression error: descriptor resnet_dt must be false!"
        self.compress = True
        self.table = DPTabulate(
            model_file, self.type_one_side, self.exclude_types, self.compress_activation_fn, suffix=suffix)
        self.table_config = [table_extrapolate, table_stride_1, table_stride_2, check_frequency]
        self.lower, self.upper \
            = self.table.build(min_nbor_dist, 
                               table_extrapolate, 
                               table_stride_1, 
                               table_stride_2)
        
        graph, _ = load_graph_def(model_file)
        self.davg = get_tensor_by_name_from_graph(graph, 'descrpt_attr%s/t_avg' % suffix)
        self.dstd = get_tensor_by_name_from_graph(graph, 'descrpt_attr%s/t_std' % suffix)



    def build (self, 
               coord_ : tf.Tensor, 
               atype_ : tf.Tensor,
               natoms : tf.Tensor,
               box_ : tf.Tensor, 
               mesh : tf.Tensor,
               input_dict : dict, 
               reuse : bool = None,
               suffix : str = ''
    ) -> tf.Tensor:
        """
        Build the computational graph for the descriptor

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
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt]) 
            if dstd is None:
                dstd = np.ones ([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]), 
                                 name = 'rcut', 
                                 dtype = GLOBAL_TF_FLOAT_PRECISION)
            t_ntypes = tf.constant(self.ntypes, 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, 
                                     name = 'ndescrpt', 
                                     dtype = tf.int32)            
            t_sel = tf.constant(self.sel_a, 
                                name = 'sel', 
                                dtype = tf.int32)            
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

        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        box   = tf.reshape (box_, [-1, 9])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = op_module.prod_env_mat_a (coord,
                                       atype,
                                       natoms,
                                       box,
                                       mesh,
                                       self.t_avg,
                                       self.t_std,
                                       rcut_a = self.rcut_a,
                                       rcut_r = self.rcut_r,
                                       rcut_r_smth = self.rcut_r_smth,
                                       sel_a = self.sel_a,
                                       sel_r = self.sel_r)
        # only used when tensorboard was set as true
        tf.summary.histogram('descrpt', self.descrpt)
        tf.summary.histogram('rij', self.rij)
        tf.summary.histogram('nlist', self.nlist)

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self._identity_tensors(suffix=suffix)

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, 
                                                 atype,
                                                 natoms, 
                                                 input_dict,
                                                 suffix = suffix, 
                                                 reuse = reuse, 
                                                 trainable = self.trainable)

        # only used when tensorboard was set as true
        tf.summary.histogram('embedding_net_output', self.dout)
        return self.dout
    
    def get_rot_mat(self) -> tf.Tensor:
        """
        Get rotational matrix
        """
        return self.qmat

    def prod_force_virial(self, 
                          atom_ener : tf.Tensor, 
                          natoms : tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Compute force and virial

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
                The total virial
        atom_virial
                The atomic virial
        """
        [net_deriv] = tf.gradients (atom_ener, self.descrpt_reshape)
        tf.summary.histogram('net_derivative', net_deriv)
        net_deriv_reshape = tf.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])        
        force \
            = op_module.prod_force_se_a (net_deriv_reshape,
                                          self.descrpt_deriv,
                                          self.nlist,
                                          natoms,
                                          n_a_sel = self.nnei_a,
                                          n_r_sel = self.nnei_r)
        virial, atom_virial \
            = op_module.prod_virial_se_a (net_deriv_reshape,
                                           self.descrpt_deriv,
                                           self.rij,
                                           self.nlist,
                                           natoms,
                                           n_a_sel = self.nnei_a,
                                           n_r_sel = self.nnei_r)
        tf.summary.histogram('force', force)
        tf.summary.histogram('virial', virial)
        tf.summary.histogram('atom_virial', atom_virial)
        
        return force, virial, atom_virial
        

    def _pass_filter(self, 
                     inputs,
                     atype,
                     natoms,
                     input_dict,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        if input_dict is not None:
            type_embedding = input_dict.get('type_embedding', None)
        else:
            type_embedding = None
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        output = []
        output_qmat = []
        if not self.type_one_side and type_embedding is None:
            for type_i in range(self.ntypes):
                inputs_i = tf.slice (inputs,
                                     [ 0, start_index*      self.ndescrpt],
                                     [-1, natoms[2+type_i]* self.ndescrpt] )
                inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
                layer, qmat = self._filter(tf.cast(inputs_i, self.filter_precision), type_i, name='filter_type_'+str(type_i)+suffix, natoms=natoms, reuse=reuse, trainable = trainable, activation_fn = self.filter_activation_fn)
                layer = tf.reshape(layer, [tf.shape(inputs)[0], natoms[2+type_i] * self.get_dim_out()])
                qmat  = tf.reshape(qmat,  [tf.shape(inputs)[0], natoms[2+type_i] * self.get_dim_rot_mat_1() * 3])
                output.append(layer)
                output_qmat.append(qmat)
                start_index += natoms[2+type_i]
        else :
            inputs_i = inputs
            inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
            type_i = -1
            layer, qmat = self._filter(tf.cast(inputs_i, self.filter_precision), type_i, name='filter_type_all'+suffix, natoms=natoms, reuse=reuse, trainable = trainable, activation_fn = self.filter_activation_fn, type_embedding=type_embedding)
            layer = tf.reshape(layer, [tf.shape(inputs)[0], natoms[0] * self.get_dim_out()])
            qmat  = tf.reshape(qmat,  [tf.shape(inputs)[0], natoms[0] * self.get_dim_rot_mat_1() * 3])
            output.append(layer)
            output_qmat.append(qmat)
        output = tf.concat(output, axis = 1)
        output_qmat = tf.concat(output_qmat, axis = 1)
        return output, output_qmat


    def _compute_dstats_sys_smth (self,
                                 data_coord, 
                                 data_box, 
                                 data_atype,                             
                                 natoms_vec,
                                 mesh) :    
        dd_all \
            = run_sess(self.sub_sess, self.stat_descrpt, 
                                feed_dict = {
                                    self.place_holders['coord']: data_coord,
                                    self.place_holders['type']: data_atype,
                                    self.place_holders['natoms_vec']: natoms_vec,
                                    self.place_holders['box']: data_box,
                                    self.place_holders['default_mesh']: mesh,
                                })
        natoms = natoms_vec
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysr = []
        sysa = []
        sysn = []
        sysr2 = []
        sysa2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index        
            # compute
            dd = np.reshape (dd, [-1, 4])
            ddr = dd[:,:1]
            dda = dd[:,1:]
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.
            sumn = dd.shape[0]
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn


    def _compute_std (self,sumv2, sumv, sumn) :
        if sumn == 0:
            return 1e-2
        val = np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val


    def _concat_type_embedding(
            self,
            xyz_scatter,
            nframes,
            natoms,
            type_embedding,
    ):
        '''Concatenate `type_embedding` of neighbors and `xyz_scatter`.
        If not self.type_one_side, concatenate `type_embedding` of center atoms as well.

        Parameters
        ----------
        xyz_scatter:
                shape is [nframes*natoms[0]*self.nnei, 1]
        nframes:
                shape is []
        natoms:
                shape is [1+1+self.ntypes]
        type_embedding:
                shape is [self.ntypes, Y] where Y=jdata['type_embedding']['neuron'][-1]

        Returns
        -------
            embedding:
                environment of each atom represented by embedding.
        '''
        te_out_dim = type_embedding.get_shape().as_list()[-1]        
        nei_embed = tf.nn.embedding_lookup(type_embedding,tf.cast(self.nei_type,dtype=tf.int32))  # shape is [self.nnei, 1+te_out_dim]
        nei_embed = tf.tile(nei_embed,(nframes*natoms[0],1))  # shape is [nframes*natoms[0]*self.nnei, te_out_dim]
        nei_embed = tf.reshape(nei_embed,[-1,te_out_dim])
        embedding_input = tf.concat([xyz_scatter,nei_embed],1)  # shape is [nframes*natoms[0]*self.nnei, 1+te_out_dim]
        if not self.type_one_side:
            atm_embed = embed_atom_type(self.ntypes, natoms, type_embedding)  # shape is [natoms[0], te_out_dim]
            atm_embed = tf.tile(atm_embed,(nframes,self.nnei))  # shape is [nframes*natoms[0], self.nnei*te_out_dim]
            atm_embed = tf.reshape(atm_embed,[-1,te_out_dim])  # shape is [nframes*natoms[0]*self.nnei, te_out_dim]
            embedding_input = tf.concat([embedding_input,atm_embed],1)  # shape is [nframes*natoms[0]*self.nnei, 1+te_out_dim+te_out_dim]
        return embedding_input


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
            is_exclude = False,
            activation_fn = None,
            bavg = 0.0,
            stddev = 1.0,
            trainable = True,
            suffix = '',
    ):
        """
        input env matrix, returns R.G
        """
        outputs_size = [1] + self.filter_neuron
        # cut-out inputs
        # with natom x (nei_type_i x 4)  
        inputs_i = tf.slice (inputs,
                             [ 0, start_index* 4],
                             [-1, incrs_index* 4] )
        shape_i = inputs_i.get_shape().as_list()
        natom = tf.shape(inputs_i)[0]
        # with (natom x nei_type_i) x 4
        inputs_reshape = tf.reshape(inputs_i, [-1, 4])
        # with (natom x nei_type_i) x 1
        xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
        if type_embedding is not None:
            type_embedding = tf.cast(type_embedding, self.filter_precision)
            xyz_scatter = self._concat_type_embedding(
                xyz_scatter, nframes, natoms, type_embedding)
            if self.compress:
                raise RuntimeError('compression of type embedded descriptor is not supported at the moment')
        # with (natom x nei_type_i) x out_size
        if self.compress and (not is_exclude):
          info = [self.lower, self.upper, self.upper * self.table_config[0], self.table_config[1], self.table_config[2], self.table_config[3]]
          if self.type_one_side:
            net = 'filter_-1_net_' + str(type_i)
          else:
            net = 'filter_' + str(type_input) + '_net_' + str(type_i)
          return op_module.tabulate_fusion(self.table.data[net].astype(self.filter_np_precision), info, xyz_scatter, tf.reshape(inputs_i, [natom, shape_i[1]//4, 4]), last_layer_size = outputs_size[-1])  
        else:
          if (not is_exclude):
              xyz_scatter = embedding_net(
                  xyz_scatter, 
                  self.filter_neuron, 
                  self.filter_precision, 
                  activation_fn = activation_fn, 
                  resnet_dt = self.filter_resnet_dt,
                  name_suffix = suffix,
                  stddev = stddev,
                  bavg = bavg,
                  seed = self.seed,
                  trainable = trainable, 
                  uniform_seed = self.uniform_seed,
                  initial_variables = self.embedding_net_variables)
              if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
          else:
            # we can safely return the final xyz_scatter filled with zero directly
            return tf.cast(tf.fill((natom, 4, outputs_size[-1]), 0.), GLOBAL_TF_FLOAT_PRECISION)
          # natom x nei_type_i x out_size
          xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))  
          # When using tf.reshape(inputs_i, [-1, shape_i[1]//4, 4]) below
          # [588 24] -> [588 6 4] correct
          # but if sel is zero
          # [588 0] -> [147 0 4] incorrect; the correct one is [588 0 4]
          # So we need to explicitly assign the shape to tf.shape(inputs_i)[0] instead of -1
          return tf.matmul(tf.reshape(inputs_i, [natom, shape_i[1]//4, 4]), xyz_scatter, transpose_a = True)


    def _filter(
            self, 
            inputs, 
            type_input,
            natoms,
            type_embedding = None,
            activation_fn=tf.nn.tanh, 
            stddev=1.0,
            bavg=0.0,
            name='linear', 
            reuse=None,
            trainable = True):
        nframes = tf.shape(tf.reshape(inputs, [-1, natoms[0], self.ndescrpt]))[0]
        # natom x (nei x 4)
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron
        all_excluded = all([(type_input, type_i) in self.exclude_types for type_i in range(self.ntypes)])
        if all_excluded:
            # all types are excluded so result and qmat should be zeros
            # we can safaly return a zero matrix...
            # See also https://stackoverflow.com/a/34725458/9567349
            # result: natom x outputs_size x outputs_size_2
            # qmat: natom x outputs_size x 3
            natom = tf.shape(inputs)[0]
            result = tf.cast(tf.fill((natom, outputs_size_2, outputs_size[-1]), 0.), GLOBAL_TF_FLOAT_PRECISION)
            qmat = tf.cast(tf.fill((natom, outputs_size[-1], 3), 0.), GLOBAL_TF_FLOAT_PRECISION)
            return result, qmat
            
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          type_i = 0
          # natom x 4 x outputs_size
          if type_embedding is None:
              for type_i in range(self.ntypes):
                  ret = self._filter_lower(
                      type_i, type_input,
                      start_index, self.sel_a[type_i],
                      inputs,
                      nframes,
                      natoms,
                      type_embedding = type_embedding,
                      is_exclude = (type_input, type_i) in self.exclude_types,
                      activation_fn = activation_fn,
                      stddev = stddev,
                      bavg = bavg,
                      trainable = trainable,
                      suffix = "_"+str(type_i))
                  if type_i == 0:
                      xyz_scatter_1 = ret
                  elif (type_input, type_i) not in self.exclude_types:
                      # add zero is meaningless; skip
                      xyz_scatter_1+= ret
                  start_index += self.sel_a[type_i]
          else :
              xyz_scatter_1 = self._filter_lower(
                  type_i, type_input,
                  start_index, np.cumsum(self.sel_a)[-1],
                  inputs,
                  nframes,
                  natoms,
                  type_embedding = type_embedding,
                  is_exclude = False,
                  activation_fn = activation_fn,
                  stddev = stddev,
                  bavg = bavg,
                  trainable = trainable)
          # natom x nei x outputs_size
          # xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
          # natom x nei x 4
          # inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
          # natom x 4 x outputs_size
          # xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
          xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
          # natom x 4 x outputs_size_2
          xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
          # # natom x 3 x outputs_size_2
          # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
          # natom x 3 x outputs_size_1
          qmat = tf.slice(xyz_scatter_1, [0,1,0], [-1, 3, -1])
          # natom x outputs_size_1 x 3
          qmat = tf.transpose(qmat, perm = [0, 2, 1])
          # natom x outputs_size x outputs_size_2
          result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
          # natom x (outputs_size x outputs_size_2)
          result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])

        return result, qmat
