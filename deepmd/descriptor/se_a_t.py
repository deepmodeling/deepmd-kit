import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter
from deepmd.utils.argcheck import list_to_doc
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.utils.network import embedding_net

class DescrptSeAT ():
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  rcut: float,
                  rcut_smth: float,
                  sel: List[str],
                  neuron: List[int] = [24,48,96],
                  resnet_dt: bool = False,
                  trainable: bool = True,
                  seed: int = 1,
                  set_davg_zero: bool = False,
                  activation_function: str = 'tanh',
                  precision: str = 'default'
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        rcut
                The cut-off radius
        rcut_smth
                From where the environment matrix should be smoothed
        sel : list[str]
                sel[i] specifies the maxmum number of type i atoms in the cut-off radius
        neuron : list[int]
                Number of neurons in each hidden layers of the embedding net
        resnet_dt
                Time-step `dt` in the resnet construction:
                y = x + dt * \phi (Wx + b)
        trainable
                If the weights of embedding net are trainable.
        seed
                Random seed for initializing the network parameters.
        set_davg_zero
                Set the shift of embedding net input to zero.
        activation_function
                The activation function in the embedding net. Supported options are {0}
        precision
                The precision of the embedding net parameters. Supported options are {1}
        """
        self.sel_a = sel
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.filter_neuron = neuron
        self.filter_resnet_dt = resnet_dt
        self.seed = seed
        self.trainable = trainable
        self.filter_activation_fn = get_activation_func(activation_function)
        self.filter_precision = get_precision(precision)
        # self.exclude_types = set()
        # for tt in exclude_types:
        #     assert(len(tt) == 2)
        #     self.exclude_types.add((tt[0], tt[1]))
        #     self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = set_davg_zero

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

        self.place_holders = {}
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
                = op_module.descrpt_se_a(self.place_holders['coord'],
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
        Returns the cut-off radisu
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
                davgunit = [sumr[type_i]/sumn[type_i], 0, 0, 0]
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
            = op_module.descrpt_se_a (coord,
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

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name = 'o_rmat')
        self.descrpt_deriv = tf.identity(self.descrpt_deriv, name = 'o_rmat_deriv')
        self.rij = tf.identity(self.rij, name = 'o_rij')
        self.nlist = tf.identity(self.nlist, name = 'o_nlist')

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, 
                                                 atype,
                                                 natoms, 
                                                 input_dict,
                                                 suffix = suffix, 
                                                 reuse = reuse, 
                                                 trainable = self.trainable)

        return self.dout


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
        Return
        ------
        force
                The force on atoms
        virial
                The total virial
        atom_virial
                The atomic virial
        """
        [net_deriv] = tf.gradients (atom_ener, self.descrpt_reshape)
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
        return force, virial, atom_virial
        

    def _pass_filter(self, 
                     inputs,
                     atype,
                     natoms,
                     input_dict,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        output = []
        output_qmat = []
        inputs_i = inputs
        inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
        type_i = -1
        layer, qmat = self._filter(tf.cast(inputs_i, self.filter_precision), type_i, name='filter_type_all'+suffix, natoms=natoms, reuse=reuse, seed = self.seed, trainable = trainable, activation_fn = self.filter_activation_fn)
        layer = tf.reshape(layer, [tf.shape(inputs)[0], natoms[0] * self.get_dim_out()])
        # qmat  = tf.reshape(qmat,  [tf.shape(inputs)[0], natoms[0] * self.get_dim_rot_mat_1() * 3])
        output.append(layer)
        # output_qmat.append(qmat)
        output = tf.concat(output, axis = 1)
        # output_qmat = tf.concat(output_qmat, axis = 1)
        return output, None


    def _compute_dstats_sys_smth (self,
                                 data_coord, 
                                 data_box, 
                                 data_atype,                             
                                 natoms_vec,
                                 mesh) :    
        dd_all \
            = self.sub_sess.run(self.stat_descrpt, 
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
        val = np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val


    def _filter(self, 
                inputs, 
                type_input,
                natoms,
                activation_fn=tf.nn.tanh, 
                stddev=1.0,
                bavg=0.0,
                name='linear', 
                reuse=None,
                seed=None, 
                trainable = True):
        # natom x (nei x 4)
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + self.filter_neuron
        with tf.variable_scope(name, reuse=reuse):
            start_index_i = 0
            result = None
            for type_i in range(self.ntypes):
                # cut-out inputs
                # with natom x (nei_type_i x 4)  
                inputs_i = tf.slice (inputs,
                                     [ 0, start_index_i      *4],
                                     [-1, self.sel_a[type_i] *4] )
                start_index_i += self.sel_a[type_i]
                nei_type_i = self.sel_a[type_i]
                shape_i = inputs_i.get_shape().as_list()
                assert(shape_i[1] == nei_type_i * 4)
                # with natom x nei_type_i x 4
                env_i = tf.reshape(inputs_i, [-1, nei_type_i, 4])
                # with natom x nei_type_i x 3
                env_i = tf.slice(env_i, [0, 0, 1], [-1, -1, -1])
                start_index_j = 0
                for type_j in range(type_i, self.ntypes):
                    # with natom x (nei_type_j x 4)  
                    inputs_j = tf.slice (inputs,
                                         [ 0, start_index_j      *4],
                                         [-1, self.sel_a[type_j] *4] )
                    start_index_j += self.sel_a[type_j]
                    nei_type_j = self.sel_a[type_j]
                    shape_j = inputs_j.get_shape().as_list()
                    assert(shape_j[1] == nei_type_j * 4)
                    # with natom x nei_type_j x 4
                    env_j = tf.reshape(inputs_j, [-1, nei_type_j, 4])
                    # with natom x nei_type_i x 3
                    env_j = tf.slice(env_j, [0, 0, 1], [-1, -1, -1])
                    # with natom x nei_type_i x nei_type_j
                    env_ij = tf.einsum('ijm,ikm->ijk', env_i, env_j)
                    # with (natom x nei_type_i x nei_type_j)
                    ebd_env_ij = tf.reshape(env_ij, [-1, 1])
                    # with (natom x nei_type_i x nei_type_j) x out_size
                    ebd_env_ij = embedding_net(ebd_env_ij, 
                                               self.filter_neuron, 
                                               self.filter_precision, 
                                               activation_fn = activation_fn, 
                                               resnet_dt = self.filter_resnet_dt,
                                               name_suffix = f"_{type_i}_{type_j}",
                                               stddev = stddev,
                                               bavg = bavg,
                                               seed = seed,
                                               trainable = trainable)
                    # with natom x nei_type_i x nei_type_j x out_size
                    ebd_env_ij = tf.reshape(ebd_env_ij, [-1, nei_type_i, nei_type_j, outputs_size[-1]])
                    # with natom x out_size
                    res_ij = tf.einsum('ijk,ijkm->im', env_ij, ebd_env_ij)
                    res_ij = res_ij * (1.0 / float(nei_type_i) / float(nei_type_j))
                    if result is None:
                        result = res_ij
                    else:
                        result += res_ij
        return result, None

