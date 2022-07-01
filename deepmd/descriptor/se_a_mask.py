import math
import numpy as np
from typing import Tuple, List, Dict, Any

from deepmd.env import tf
from deepmd.common import get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT
from deepmd.utils.argcheck import list_to_doc
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.utils.network import embedding_net, embedding_net_rand_seed_shift
from deepmd.utils.tabulate import DPTabulate
from deepmd.utils.type_embed import embed_atom_type
from deepmd.utils.sess import run_sess
from deepmd.utils.graph import load_graph_def, get_tensor_by_name_from_graph
from .descriptor import Descriptor
from .se_a import DescrptSeA


@Descriptor.register("se_e2_a_mask")
class DescrptSeAMask (DescrptSeA):
    r"""DeepPot-SE constructed from all information (both angular and radial) of
    atomic configurations. The embedding takes the distance between atoms as input.

    DescrptSeAMask is a subclass of DescrptSeA.
    Specifically, DescrptSeAMask is a concise version of DescrptSeA.
    The initialization of DescrptSeAMask is similar to DescrptSeA.
    In the running phase (training or testing and inference), DescrptSeAMask required an mask_matrix to be provided.
    
    `mask_matrix` [#frames, #total_atoms] is matrix of 0,1 to indicate the atoms that are real particles or not.
    Thus, this op can accept the frames with different number of atoms for each type without splitting dataset.
    The DescrptSeAMask op is designed for non-pbc systems.
    All the atoms in the system is the neighbors of each other.
    Attention that `rcut` and `rcut_smth` values are needed since the inheritance from DescrptSeA.
    But these two values will not be used in descriptor calculation.
    The most affordable atoms for each type are defined in `sel`. 
    For example, `sel: [20, 10]` for `type_map:["H", "O"]` means the dataset for the first type `H` has 20 atoms at most and the second type `O` has 10 atoms at most. 
    
    The returned descrptor values are zero for the vritual atoms.
    Both the angular and radial information are included in descrptor matrix.
    
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
    
    """
    #@docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
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
                  set_davg_zero: bool = True,
                  activation_function: str = 'tanh',
                  precision: str = 'default',
                  uniform_seed: bool = False
    ) -> None:
        """
        Constructor
        """
        self.sel_a = sel
        self.cum_sel_a = np.concatenate((np.array([0]), np.cumsum(self.sel_a)))
        self.total_atom_num = np.cumsum(self.sel_a)[-1]
        self.ntypes = len(self.sel_a)
        self.descrpt_type = "se_a_mask"
        # rcut and rcut_smth will not be used in se_a_mask op.
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.type_one_side = False
        self.type_embedding = None
        
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
            assert(len(tt) == 2)
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero

        # descrpt config

        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei = self.nnei_a

        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt = self.ndescrpt_a
        self.useBN = False
        self.dstd = None
        self.davg = None
        self.compress = False
        self.embedding_net_variables = None
        self.mixed_prec = None
        self.place_holders = {}
        nei_type = np.array([])
        for ii in range(self.ntypes):
            nei_type = np.append(nei_type, ii * np.ones(self.sel_a[ii])) # like a mask 
        self.nei_type = tf.constant(nei_type, dtype = tf.int32)

        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = 'd_sea_mask_'
            for ii in ['coord']:
                self.place_holders[ii] = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name = name_pfx+'t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name=name_pfx+'t_type')
            self.place_holders['mask_matrix'] = tf.placeholder(tf.bool, [None, None], name=name_pfx+'t_mask')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name=name_pfx+'t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name=name_pfx+'t_mesh')
            self.stat_descrpt, descrpt_deriv, rij, nlist \
                = op_module.descrpt_se_a_mask(self.place_holders['coord'],
                                         self.place_holders['type'],
                                         self.place_holders['mask_matrix'],
                                         total_atom_num = self.total_atom_num)
        self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)
        self.original_sel = None


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
        mask_matrix = input_dict['mask_matrix']
        all_davg = []
        all_dstd = []
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc,tt,mm,nn in zip(data_coord,data_atype,mask_matrix, natoms_vec) :
                sysr,sysr2,sysa,sysa2,sysn \
                    = self._compute_dstats_sys_smth(cc,tt,mm, nn)
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
                #davgunit = [sumr[type_i]/(sumn[type_i]+1e-15), 0, 0, 0]
                #dstdunit = [self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]), 
                #            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                #            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                #            self._compute_std(suma2[type_i], suma[type_i], sumn[type_i])
                #            ]
                
                davg = np.tile(0., self.ndescrpt // 4)
                dstd = np.tile(1., self.ndescrpt // 4)
                all_davg.append(davg)
                all_dstd.append(dstd)

        if not self.set_davg_zero:
            self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)


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
        #davg = self.davg
        #dstd = self.dstd
        davg = None
        dstd = None
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt]) 
            if dstd is None:
                dstd = np.ones ([self.ntypes, self.ndescrpt])
            #t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]), 
            #                     name = 'rcut', 
            #                     dtype = GLOBAL_TF_FLOAT_PRECISION)
            t_ntypes = tf.constant(self.ntypes, 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, 
                                     name = 'ndescrpt', 
                                     dtype = tf.int32)            
            t_sel = tf.constant(self.sel_a, 
                                name = 'sel', 
                                dtype = tf.int32)            
            '''
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
            '''
            
        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        box   = tf.reshape (box_, [-1, 9])
        atype = tf.reshape (atype_, [-1, natoms[1]])
        mask_matrix = input_dict["mask_matrix"]
        mask_matrix = tf.reshape (mask_matrix, [-1, natoms[1]])
        mask_matrix = tf.cast(mask_matrix, tf.bool)
        
        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = op_module.descrpt_se_a_mask (coord,
                                       atype,
                                       mask_matrix,
                                       total_atom_num = self.total_atom_num)
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

    def prod_force_virial(self, 
                          atom_ener : tf.Tensor, 
                          natoms : tf.Tensor,
                          mask_matrix : tf.Tensor
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
                None for se_a_mask op
        atom_virial
                None for se_a_mask op
        """
        [net_deriv] = tf.gradients (atom_ener, self.descrpt_reshape)
        tf.summary.histogram('net_derivative', net_deriv)
        net_deriv_reshape = tf.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])
        mask_matrix = tf.reshape(mask_matrix, [-1, natoms[0]])
        mask_matrix = tf.cast(mask_matrix, tf.bool)        
        force \
            = op_module.prod_force_se_a_mask (net_deriv_reshape,
                                          self.descrpt_deriv,
                                          mask_matrix,
                                          self.nlist,
                                          total_atom_num = self.total_atom_num)
        
        tf.summary.histogram('force', force)
        #tf.summary.histogram('virial', virial)
        #tf.summary.histogram('atom_virial', atom_virial)
        
        return force, None, None
        

    def _compute_dstats_sys_smth (self,
                                 data_coord, 
                                 data_atype,                             
                                 data_mask_matrix,
                                 natoms) :    
        dd_all \
            = run_sess(self.sub_sess, self.stat_descrpt, 
                                feed_dict = {
                                    self.place_holders['coord']: data_coord,
                                    self.place_holders['type']: data_atype,
                                    self.place_holders['mask_matrix']: data_mask_matrix
                                    #self.place_holders['box']: data_box,
                                })
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        num4mask = np.sum(data_mask_matrix, axis = 1)
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
            ## Calculate the sumn with real atoms.
            mask4type_i = data_mask_matrix[:, self.cum_sel_a[type_i]:self.cum_sel_a[type_i+1]]
            num4type_i = np.sum(mask4type_i, axis = 1)                        
            
            # compute
            dd = np.reshape (dd, [-1, 4])
            ddr = dd[:,:1]
            dda = dd[:,1:]
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.
            sumn = np.sum(num4type_i * num4mask)
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn

