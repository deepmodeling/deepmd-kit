import platform
import os
import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.{}".format(ext) )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.{}".format(ext))

class DescrptLocFrame () :
    def __init__(self, jdata):
        args = ClassArg()\
               .add('sel_a',    list,   must = True) \
               .add('sel_r',    list,   must = True) \
               .add('rcut',     float,  default = 6.0) \
               .add('axis_rule',list,   must = True)
        class_data = args.parse(jdata)
        self.sel_a = class_data['sel_a']
        self.sel_r = class_data['sel_r']
        self.axis_rule = class_data['axis_rule']
        self.rcut_r = class_data['rcut']
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


    def get_rcut (self) :
        return self.rcut_r

    def get_ntypes (self) :
        return self.ntypes

    def get_dim_out (self) :
        return self.ndescrpt

    def get_nlist (self) :
        return self.nlist, self.rij, self.sel_a, self.sel_r

    def compute_dstats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                        mesh) :
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
        davg = np.array(all_davg)
        dstd = np.array(all_dstd)
        return davg, dstd
        
        
    def build (self, 
               coord_, 
               atype_,
               natoms,
               box_, 
               mesh,
               davg = None, 
               dstd = None,
               suffix = '', 
               reuse = None):
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt]) 
            if dstd is None:
                dstd = np.ones ([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]), 
                                 name = 'rcut', 
                                 dtype = global_tf_float_precision)
            t_ntypes = tf.constant(self.ntypes, 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
            self.t_avg = tf.get_variable('t_avg', 
                                         davg.shape, 
                                         dtype = global_tf_float_precision,
                                         trainable = False,
                                         initializer = tf.constant_initializer(davg))
            self.t_std = tf.get_variable('t_std', 
                                         dstd.shape, 
                                         dtype = global_tf_float_precision,
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
        return self.descrpt

    def get_rot_mat(self) :
        return self.rot_mat

    def prod_force_virial(self, atom_ener, natoms) :
        [net_deriv] = tf.gradients (atom_ener, self.descrpt)
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

        return force, virial, atom_virial


    def _compute_dstats_sys_nonsmth (self,
                                    data_coord, 
                                    data_box, 
                                    data_atype, 
                                    natoms_vec,
                                    mesh) :    
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            descrpt, descrpt_deriv, rij, nlist, axis, rot_mat \
                = op_module.descrpt (tf.constant(data_coord),
                                     tf.constant(data_atype),
                                     tf.constant(natoms_vec, dtype = tf.int32),
                                     tf.constant(data_box),
                                     tf.constant(mesh),
                                     tf.constant(avg_zero),
                                     tf.constant(std_ones),
                                     rcut_a = self.rcut_a,
                                     rcut_r = self.rcut_r,
                                     sel_a = self.sel_a,
                                     sel_r = self.sel_r,
                                     axis_rule = self.axis_rule)
        # self.sess.run(tf.global_variables_initializer())
        # sub_sess = tf.Session(graph = sub_graph, 
        #                       config=tf.ConfigProto(intra_op_parallelism_threads=self.run_opt.num_intra_threads, 
        #                                             inter_op_parallelism_threads=self.run_opt.num_inter_threads
        #                       ))
        sub_sess = tf.Session(graph = sub_graph)
        dd_all = sub_sess.run(descrpt)
        sub_sess.close()
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

    
