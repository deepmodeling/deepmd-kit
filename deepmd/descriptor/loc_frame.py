import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import op_module
from deepmd.env import default_tf_session_config

class DescrptLocFrame () :
    def __init__(self, 
                 rcut: float,
                 sel_a : List[int],
                 sel_r : List[int],
                 axis_rule : List[int]
    ) -> None:
        """
        Constructor

        Parameters
        rcut
                The cut-off radius
        sel_a : list[str]
                The length of the list should be the same as the number of atom types in the system. 
                `sel_a[i]` gives the selected number of type-i neighbors. 
                The full relative coordinates of the neighbors are used by the descriptor.
        sel_r : list[str]
                The length of the list should be the same as the number of atom types in the system. 
                `sel_r[i]` gives the selected number of type-i neighbors. 
                Only relative distance of the neighbors are used by the descriptor.
                sel_a[i] + sel_r[i] is recommended to be larger than the maximally possible number of type-i neighbors in the cut-off radius.        
        axis_rule: list[int]
                The length should be 6 times of the number of types. 
                - axis_rule[i*6+0]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.\n\n\
                - axis_rule[i*6+1]: type of the atom defining the first axis of type-i atom.\n\n\
                - axis_rule[i*6+2]: index of the axis atom defining the first axis. Note that the neighbors with the same class and type are sorted according to their relative distance.\n\n\
                - axis_rule[i*6+3]: class of the atom defining the first axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.\n\n\
                - axis_rule[i*6+4]: type of the atom defining the second axis of type-i atom.\n\n\
                - axis_rule[i*6+5]: class of the atom defining the second axis of type-i atom. 0 for neighbors with full coordinates and 1 for neighbors only with relative distance.        
        """
        # args = ClassArg()\
        #        .add('sel_a',    list,   must = True) \
        #        .add('sel_r',    list,   must = True) \
        #        .add('rcut',     float,  default = 6.0) \
        #        .add('axis_rule',list,   must = True)
        # class_data = args.parse(jdata)
        self.sel_a = sel_a
        self.sel_r = sel_r
        self.axis_rule = axis_rule
        self.rcut_r = rcut
        # ntypes and rcut_a === -1
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
        self.davg = None
        self.dstd = None

        self.place_holders = {}
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(GLOBAL_NP_FLOAT_PRECISION)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = 'd_lf_'
            for ii in ['coord', 'box']:
                self.place_holders[ii] = tf.placeholder(GLOBAL_NP_FLOAT_PRECISION, [None, None], name = name_pfx+'t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name=name_pfx+'t_type')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name=name_pfx+'t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name=name_pfx+'t_mesh')
            self.stat_descrpt, descrpt_deriv, rij, nlist, axis, rot_mat \
                = op_module.descrpt (self.place_holders['coord'],
                                     self.place_holders['type'],
                                     self.place_holders['natoms_vec'],
                                     self.place_holders['box'],
                                     self.place_holders['default_mesh'],
                                     tf.constant(avg_zero),
                                     tf.constant(std_ones),
                                     rcut_a = self.rcut_a,
                                     rcut_r = self.rcut_r,
                                     sel_a = self.sel_a,
                                     sel_r = self.sel_r,
                                     axis_rule = self.axis_rule)
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
        return self.ndescrpt

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
            sumv = []
            sumn = []
            sumv2 = []
            for cc,bb,tt,nn,mm in zip(data_coord,data_box,data_atype,natoms_vec,mesh) :
                sysv,sysv2,sysn \
                    = self._compute_dstats_sys_nonsmth(cc,bb,tt,nn,mm)
                sumv.append(sysv)
                sumn.append(sysn)
                sumv2.append(sysv2)
            sumv = np.sum(sumv, axis = 0)
            sumn = np.sum(sumn, axis = 0)
            sumv2 = np.sum(sumv2, axis = 0)
            for type_i in range(self.ntypes) :
                davg = sumv[type_i] /  sumn[type_i]
                dstd = self._compute_std(sumv2[type_i], sumv[type_i], sumn[type_i])
                for ii in range (len(dstd)) :
                    if (np.abs(dstd[ii]) < 1e-2) :
                        dstd[ii] = 1e-2            
                all_davg.append(davg)
                all_dstd.append(dstd)
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

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist, self.axis, self.rot_mat \
            = op_module.descrpt (coord,
                                 atype,
                                 natoms,
                                 box,                                    
                                 mesh,
                                 self.t_avg,
                                 self.t_std,
                                 rcut_a = self.rcut_a,
                                 rcut_r = self.rcut_r,
                                 sel_a = self.sel_a,
                                 sel_r = self.sel_r,
                                 axis_rule = self.axis_rule)
        self.descrpt = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        tf.summary.histogram('descrpt', self.descrpt)
        tf.summary.histogram('rij', self.rij)
        tf.summary.histogram('nlist', self.nlist)

        return self.descrpt

    def get_rot_mat(self) -> tf.Tensor:
        """
        Get rotational matrix
        """
        return self.rot_mat

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
        [net_deriv] = tf.gradients (atom_ener, self.descrpt)
        tf.summary.histogram('net_derivative', net_deriv)
        net_deriv_reshape = tf.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])
        force = op_module.prod_force (net_deriv_reshape,
                                      self.descrpt_deriv,
                                      self.nlist,
                                      self.axis,
                                      natoms,
                                      n_a_sel = self.nnei_a,
                                      n_r_sel = self.nnei_r)
        virial, atom_virial \
            = op_module.prod_virial (net_deriv_reshape,
                                     self.descrpt_deriv,
                                     self.rij,
                                     self.nlist,
                                     self.axis,
                                     natoms,
                                     n_a_sel = self.nnei_a,
                                     n_r_sel = self.nnei_r)
        tf.summary.histogram('force', force)
        tf.summary.histogram('virial', virial)
        tf.summary.histogram('atom_virial', atom_virial)

        return force, virial, atom_virial


    def _compute_dstats_sys_nonsmth (self,
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
        sysv = []
        sysn = []
        sysv2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index        
            # compute
            sumv = np.sum(dd, axis = 0)
            sumn = dd.shape[0]
            sumv2 = np.sum(np.multiply(dd,dd), axis = 0)            
            sysv.append(sumv)
            sysn.append(sumn)
            sysv2.append(sumv2)
        return sysv, sysv2, sysn


    def _compute_std (self,sumv2, sumv, sumn) :
        return np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))

    
