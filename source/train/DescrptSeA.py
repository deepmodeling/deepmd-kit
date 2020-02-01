import platform
import os,sys,warnings
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

class DescrptSeA ():
    def __init__ (self, jdata):
        args = ClassArg()\
               .add('sel',      list,   must = True) \
               .add('rcut',     float,  default = 6.0) \
               .add('rcut_smth',float,  default = 5.5) \
               .add('neuron',   list,   default = [10, 20, 40]) \
               .add('axis_neuron', int, default = 4, alias = 'n_axis_neuron') \
               .add('resnet_dt',bool,   default = False) \
               .add('trainable',bool,   default = True) \
               .add('seed',     int) 
        class_data = args.parse(jdata)
        self.sel_a = class_data['sel']
        self.rcut_r = class_data['rcut']
        self.rcut_r_smth = class_data['rcut_smth']
        self.filter_neuron = class_data['neuron']
        self.n_axis_neuron = class_data['axis_neuron']
        self.filter_resnet_dt = class_data['resnet_dt']
        self.seed = class_data['seed']
        self.trainable = class_data['trainable']

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


    def get_rcut (self) :
        return self.rcut_r

    def get_ntypes (self) :
        return self.ntypes

    def get_dim_out (self) :
        return self.filter_neuron[-1] * self.n_axis_neuron

    def get_dim_rot_mat_1 (self) :
        return self.filter_neuron[-1]

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

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, natoms, suffix = suffix, reuse = reuse, trainable = self.trainable)

        return self.dout

    
    def get_rot_mat(self) :
        return self.qmat


    def prod_force_virial(self, atom_ener, natoms) :
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
                     natoms,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        shape = inputs.get_shape().as_list()
        output = []
        output_qmat = []
        for type_i in range(self.ntypes):
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.ndescrpt],
                                 [-1, natoms[2+type_i]* self.ndescrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
            layer, qmat = self._filter(inputs_i, name='filter_type_'+str(type_i)+suffix, natoms=natoms, reuse=reuse, seed = self.seed, trainable = trainable)
            layer = tf.reshape(layer, [tf.shape(inputs)[0], natoms[2+type_i] * self.get_dim_out()])
            qmat  = tf.reshape(qmat,  [tf.shape(inputs)[0], natoms[2+type_i] * self.get_dim_rot_mat_1() * 3])
            output.append(layer)
            output_qmat.append(qmat)
            start_index += natoms[2+type_i]
        output = tf.concat(output, axis = 1)
        output_qmat = tf.concat(output_qmat, axis = 1)
        return output, output_qmat


    def _compute_dstats_sys_smth (self,
                                 data_coord, 
                                 data_box, 
                                 data_atype,                             
                                 natoms_vec,
                                 mesh) :    
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            descrpt, descrpt_deriv, rij, nlist \
                = op_module.descrpt_se_a (tf.constant(data_coord),
                                           tf.constant(data_atype),
                                           tf.constant(natoms_vec, dtype = tf.int32),
                                           tf.constant(data_box),
                                           tf.constant(mesh),
                                           tf.constant(avg_zero),
                                           tf.constant(std_ones),
                                           rcut_a = self.rcut_a,
                                           rcut_r = self.rcut_r,
                                           rcut_r_smth = self.rcut_r_smth,
                                           sel_a = self.sel_a,
                                           sel_r = self.sel_r)
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
        outputs_size_2 = self.n_axis_neuron
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          xyz_scatter_total = []
          for type_i in range(self.ntypes):
            # cut-out inputs
            # with natom x (nei_type_i x 4)  
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      4],
                                 [-1, self.sel_a[type_i]* 4] )
            start_index += self.sel_a[type_i]
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei_type_i) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4])
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
            for ii in range(1, len(outputs_size)):
              w = tf.get_variable('matrix_'+str(ii)+'_'+str(type_i), 
                                [outputs_size[ii - 1], outputs_size[ii]], 
                                global_tf_float_precision,
                                  tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed), 
                                  trainable = trainable)
              b = tf.get_variable('bias_'+str(ii)+'_'+str(type_i), 
                                [1, outputs_size[ii]], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                                  trainable = trainable)
              if self.filter_resnet_dt :
                  idt = tf.get_variable('idt_'+str(ii)+'_'+str(type_i), 
                                        [1, outputs_size[ii]], 
                                        global_tf_float_precision,
                                        tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed), 
                                        trainable = trainable)
              if outputs_size[ii] == outputs_size[ii-1]:
                  if self.filter_resnet_dt :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
              elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                  if self.filter_resnet_dt :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
              else:
                  xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
            # natom x nei_type_i x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))
            xyz_scatter_total.append(xyz_scatter)

          # natom x nei x outputs_size
          xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
          # natom x nei x 4
          inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
          # natom x 4 x outputs_size
          xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
          xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
          # natom x 4 x outputs_size_2
          xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
          # # natom x 3 x outputs_size_2
          # qmat = tf.slice(xyz_scatter_2, [0,1,0], [-1, 3, -1])
          # natom x 3 x outputs_size_1
          qmat = tf.slice(xyz_scatter_1, [0,1,0], [-1, 3, -1])
          # natom x outputs_size_2 x 3
          qmat = tf.transpose(qmat, perm = [0, 2, 1])
          # natom x outputs_size x outputs_size_2
          result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
          # natom x (outputs_size x outputs_size_2)
          result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])

        return result, qmat

    def _filter_type_ext(self, 
                           inputs, 
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
        outputs_size_2 = self.n_axis_neuron
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          result_all = []
          xyz_scatter_1_all = []
          xyz_scatter_2_all = []
          for type_i in range(self.ntypes):
            # cut-out inputs
            # with natom x (nei_type_i x 4)  
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      4],
                                 [-1, self.sel_a[type_i]* 4] )
            start_index += self.sel_a[type_i]
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei_type_i) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4])
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
            for ii in range(1, len(outputs_size)):
              w = tf.get_variable('matrix_'+str(ii)+'_'+str(type_i), 
                                [outputs_size[ii - 1], outputs_size[ii]], 
                                global_tf_float_precision,
                                  tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed),
                                  trainable = trainable)
              b = tf.get_variable('bias_'+str(ii)+'_'+str(type_i), 
                                [1, outputs_size[ii]], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed),
                                  trainable = trainable)
              if self.filter_resnet_dt :
                  idt = tf.get_variable('idt_'+str(ii)+'_'+str(type_i), 
                                        [1, outputs_size[ii]], 
                                        global_tf_float_precision,
                                        tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed),
                                        trainable = trainable)
              if outputs_size[ii] == outputs_size[ii-1]:
                  if self.filter_resnet_dt :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
              elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                  if self.filter_resnet_dt :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
              else:
                  xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
            # natom x nei_type_i x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))
            # natom x nei_type_i x 4  
            inputs_i_reshape = tf.reshape(inputs_i, [-1, shape_i[1]//4, 4])
            # natom x 4 x outputs_size
            xyz_scatter_1 = tf.matmul(inputs_i_reshape, xyz_scatter, transpose_a = True)
            xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape_i[1])
            # natom x 4 x outputs_size_2
            xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
            xyz_scatter_1_all.append(xyz_scatter_1)
            xyz_scatter_2_all.append(xyz_scatter_2)

          # for type_i in range(self.ntypes):
          #   for type_j in range(type_i, self.ntypes):
          #     # natom x outputs_size x outputs_size_2
          #     result = tf.matmul(xyz_scatter_1_all[type_i], xyz_scatter_2_all[type_j], transpose_a = True)
          #     # natom x (outputs_size x outputs_size_2)
          #     result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])
          #     result_all.append(tf.identity(result))
          xyz_scatter_2_coll = tf.concat(xyz_scatter_2_all, axis = 2)
          for type_i in range(self.ntypes) :
              # natom x outputs_size x (outputs_size_2 x ntypes)
              result = tf.matmul(xyz_scatter_1_all[type_i], xyz_scatter_2_coll, transpose_a = True)
              # natom x (outputs_size x outputs_size_2 x ntypes)
              result = tf.reshape(result, [-1, outputs_size_2 * self.ntypes * outputs_size[-1]])
              result_all.append(tf.identity(result))              

          # natom x (ntypes x outputs_size x outputs_size_2 x ntypes)
          result_all = tf.concat(result_all, axis = 1)

        return result_all
