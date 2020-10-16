import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, get_activation_func, get_precision
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.env import op_module
from deepmd.env import default_tf_session_config

class DescrptSeR ():
    def __init__ (self, jdata):
        args = ClassArg()\
               .add('sel',      list,   must = True) \
               .add('rcut',     float,  default = 6.0) \
               .add('rcut_smth',float,  default = 0.5) \
               .add('neuron',   list,   default = [10, 20, 40]) \
               .add('resnet_dt',bool,   default = False) \
               .add('trainable',bool,   default = True) \
               .add('seed',     int) \
               .add('type_one_side', bool, default = False) \
               .add('exclude_types', list, default = []) \
               .add('set_davg_zero', bool, default = False) \
               .add("activation_function", str, default = "tanh") \
               .add("precision",           str, default = "default")
        class_data = args.parse(jdata)
        self.sel_r = class_data['sel']
        self.rcut = class_data['rcut']
        self.rcut_smth = class_data['rcut_smth']
        self.filter_neuron = class_data['neuron']
        self.filter_resnet_dt = class_data['resnet_dt']
        self.seed = class_data['seed']        
        self.trainable = class_data['trainable']
        self.filter_activation_fn = get_activation_func(class_data["activation_function"]) 
        self.filter_precision = get_precision(class_data['precision'])  
        exclude_types = class_data['exclude_types']
        self.exclude_types = set()
        for tt in exclude_types:
            assert(len(tt) == 2)
            self.exclude_types.add((tt[0], tt[1]))
            self.exclude_types.add((tt[1], tt[0]))
        self.set_davg_zero = class_data['set_davg_zero']
        self.type_one_side = class_data['type_one_side']

        # descrpt config
        self.sel_a = [ 0 for ii in range(len(self.sel_r)) ]
        self.ntypes = len(self.sel_r)
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.nnei_r
        self.useBN = False
        self.davg = None
        self.dstd = None

        self.place_holders = {}
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = 'd_ser_'
            for ii in ['coord', 'box']:
                self.place_holders[ii] = tf.placeholder(global_np_float_precision, [None, None], name = name_pfx+'t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name=name_pfx+'t_type')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name=name_pfx+'t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name=name_pfx+'t_mesh')
            self.stat_descrpt, descrpt_deriv, rij, nlist \
                = op_module.descrpt_se_r(self.place_holders['coord'],
                                         self.place_holders['type'],
                                         self.place_holders['natoms_vec'],
                                         self.place_holders['box'],
                                         self.place_holders['default_mesh'],
                                         tf.constant(avg_zero),
                                         tf.constant(std_ones),
                                         rcut = self.rcut,
                                         rcut_smth = self.rcut_smth,
                                         sel = self.sel_r)
            self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)


    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_dim_out (self) :
        return self.filter_neuron[-1]

    def get_nlist (self) :
        return self.nlist, self.rij, self.sel_a, self.sel_r

    def compute_input_stats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                        mesh) :    
        all_davg = []
        all_dstd = []
        sumr = []
        sumn = []
        sumr2 = []
        for cc,bb,tt,nn,mm in zip(data_coord,data_box,data_atype,natoms_vec,mesh) :
            sysr,sysr2,sysn \
                = self._compute_dstats_sys_se_r(cc,bb,tt,nn,mm)
            sumr.append(sysr)
            sumn.append(sysn)
            sumr2.append(sysr2)
        sumr = np.sum(sumr, axis = 0)
        sumn = np.sum(sumn, axis = 0)
        sumr2 = np.sum(sumr2, axis = 0)
        for type_i in range(self.ntypes) :
            davgunit = [sumr[type_i]/sumn[type_i]]
            dstdunit = [self._compute_std(sumr2[type_i], sumr[type_i], sumn[type_i])]
            davg = np.tile(davgunit, self.ndescrpt // 1)
            dstd = np.tile(dstdunit, self.ndescrpt // 1)
            all_davg.append(davg)
            all_dstd.append(dstd)

        if not self.set_davg_zero:
            self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)


    def build (self, 
               coord_, 
               atype_,
               natoms,
               box_, 
               mesh,
               suffix = '', 
               reuse = None):
        davg = self.davg
        dstd = self.dstd
        with tf.variable_scope('descrpt_attr' + suffix, reuse = reuse) :
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt]) 
            if dstd is None:
                dstd = np.ones ([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(self.rcut, 
                                 name = 'rcut', 
                                 dtype = global_tf_float_precision)
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
            = op_module.descrpt_se_r (coord,
                                      atype,
                                      natoms,
                                      box,
                                      mesh,
                                      self.t_avg,
                                      self.t_std,
                                      rcut = self.rcut,
                                      rcut_smth = self.rcut_smth,
                                      sel = self.sel_r)

        self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
        self.descrpt_reshape = tf.identity(self.descrpt_reshape, name = 'o_rmat')
        self.descrpt_deriv = tf.identity(self.descrpt_deriv, name = 'o_rmat_deriv')
        self.rij = tf.identity(self.rij, name = 'o_rij')
        self.nlist = tf.identity(self.nlist, name = 'o_nlist')

        self.dout = self._pass_filter(self.descrpt_reshape, natoms, suffix = suffix, reuse = reuse, trainable = self.trainable)

        return self.dout


    def prod_force_virial(self, atom_ener, natoms) :
        [net_deriv] = tf.gradients (atom_ener, self.descrpt_reshape)
        net_deriv_reshape = tf.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])        
        force \
            = op_module.prod_force_se_r (net_deriv_reshape,
                                         self.descrpt_deriv,
                                         self.nlist,
                                         natoms)
        virial, atom_virial \
            = op_module.prod_virial_se_r (net_deriv_reshape,
                                          self.descrpt_deriv,
                                          self.rij,
                                          self.nlist,
                                          natoms)
        return force, virial, atom_virial
    

    def _pass_filter(self, 
                     inputs,
                     natoms,
                     reuse = None,
                     suffix = '', 
                     trainable = True) :
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        output = []
        if not self.type_one_side:
            for type_i in range(self.ntypes):
                inputs_i = tf.slice (inputs,
                                     [ 0, start_index*      self.ndescrpt],
                                     [-1, natoms[2+type_i]* self.ndescrpt] )
                inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
                layer = self._filter_r(tf.cast(inputs_i, self.filter_precision), type_i, name='filter_type_'+str(type_i)+suffix, natoms=natoms, reuse=reuse, seed = self.seed, trainable = trainable, activation_fn = self.filter_activation_fn)
                layer = tf.reshape(layer, [tf.shape(inputs)[0], natoms[2+type_i] * self.get_dim_out()])
                output.append(layer)
                start_index += natoms[2+type_i]
        else :
            inputs_i = inputs
            inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
            type_i = -1
            layer = self._filter_r(tf.cast(inputs_i, self.filter_precision), type_i, name='filter_type_all'+suffix, natoms=natoms, reuse=reuse, seed = self.seed, trainable = trainable, activation_fn = self.filter_activation_fn)
            layer = tf.reshape(layer, [tf.shape(inputs)[0], natoms[0] * self.get_dim_out()])
            output.append(layer)
        output = tf.concat(output, axis = 1)
        return output

    def _compute_dstats_sys_se_r (self,
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
        sysn = []
        sysr2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index        
            # compute
            dd = np.reshape (dd, [-1, 1])
            ddr = dd[:,:1]
            sumr = np.sum(ddr)
            sumn = dd.shape[0]
            sumr2 = np.sum(np.multiply(ddr, ddr))
            sysr.append(sumr)
            sysn.append(sumn)
            sysr2.append(sumr2)
        return sysr, sysr2, sysn


    def _compute_std (self,sumv2, sumv, sumn) :
        val = np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val


    def _filter_r(self, 
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
        # natom x nei
        outputs_size = [1] + self.filter_neuron
        with tf.variable_scope(name, reuse=reuse):
            start_index = 0
            xyz_scatter_total = []
            for type_i in range(self.ntypes):
                # cut-out inputs
                # with natom x nei_type_i
                inputs_i = tf.slice (inputs,
                                     [ 0, start_index       ],
                                     [-1, self.sel_r[type_i]] )
                start_index += self.sel_r[type_i]
                shape_i = inputs_i.get_shape().as_list()
                # with (natom x nei_type_i) x 1
                xyz_scatter = tf.reshape(inputs_i, [-1, 1])
                if (type_input, type_i) not in self.exclude_types:
                    for ii in range(1, len(outputs_size)):
                        w = tf.get_variable('matrix_'+str(ii)+'_'+str(type_i), 
                                            [outputs_size[ii - 1], outputs_size[ii]], 
                                            self.filter_precision,
                                            tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed), 
                                            trainable = trainable)
                        b = tf.get_variable('bias_'+str(ii)+'_'+str(type_i), 
                                            [1, outputs_size[ii]], 
                                            self.filter_precision,
                                            tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                                            trainable = trainable)
                        hidden = tf.reshape(activation_fn(tf.matmul(xyz_scatter, w) + b), [-1, outputs_size[ii]])
                        if self.filter_resnet_dt :
                            idt = tf.get_variable('idt_'+str(ii)+'_'+str(type_i), 
                                                  [1, outputs_size[ii]], 
                                                  self.filter_precision,
                                                  tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed), 
                                                  trainable = trainable)
                        if outputs_size[ii] == outputs_size[ii-1]:
                            if self.filter_resnet_dt :
                                xyz_scatter += hidden * idt
                            else :
                                xyz_scatter += hidden
                        elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                            if self.filter_resnet_dt :
                                xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + hidden * idt
                            else :
                                xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + hidden
                        else:
                            xyz_scatter = hidden
                else:
                    w = tf.zeros((outputs_size[0], outputs_size[-1]), dtype=global_tf_float_precision)
                    xyz_scatter = tf.matmul(xyz_scatter, w)
                # natom x nei_type_i x out_size
                xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1], outputs_size[-1]))
                xyz_scatter_total.append(xyz_scatter)

            # natom x nei x outputs_size
            xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
            # natom x outputs_size
            # 
            res_rescale = 1./5.
            result = tf.reduce_mean(xyz_scatter, axis = 1) * res_rescale

        return result
