import numpy as np
from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.env import op_module
from deepmd.env import default_tf_session_config
from deepmd.DescrptSeA import DescrptSeA

class DescrptSeAEf ():
    def __init__(self, jdata):
        self.descrpt_para = DescrptSeAEfLower(jdata, op_module.descrpt_se_a_ef_para)
        self.descrpt_vert = DescrptSeAEfLower(jdata, op_module.descrpt_se_a_ef_vert)
        
    def get_rcut (self) :
        return self.descrpt_vert.rcut_r

    def get_ntypes (self) :
        return self.descrpt_vert.ntypes

    def get_dim_out (self) :
        return self.descrpt_vert.get_dim_out() + self.descrpt_para.get_dim_out()

    def get_dim_rot_mat_1 (self) :
        return self.descrpt_vert.filter_neuron[-1]

    def get_rot_mat(self) :
        return self.qmat


    def get_nlist (self) :
        return \
            self.descrpt_vert.nlist, \
            self.descrpt_vert.rij, \
            self.descrpt_vert.sel_a, \
            self.descrpt_vert.sel_r

    def compute_input_stats (self,
                             data_coord, 
                             data_box, 
                             data_atype, 
                             natoms_vec,
                             mesh, 
                             input_dict) :
        self.descrpt_vert.compute_input_stats(data_coord, data_box, data_atype, natoms_vec, mesh, input_dict)
        self.descrpt_para.compute_input_stats(data_coord, data_box, data_atype, natoms_vec, mesh, input_dict)

    def build (self, 
               coord_, 
               atype_,
               natoms,
               box_, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        self.dout_vert = self.descrpt_vert.build(coord_, atype_, natoms, box_, mesh, input_dict)
        self.dout_para = self.descrpt_para.build(coord_, atype_, natoms, box_, mesh, input_dict, reuse = True)
        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        nframes = tf.shape(coord)[0]
        self.dout_vert = tf.reshape(self.dout_vert, [nframes * natoms[0], self.descrpt_vert.get_dim_out()])
        self.dout_para = tf.reshape(self.dout_para, [nframes * natoms[0], self.descrpt_para.get_dim_out()])
        self.dout = tf.concat([self.dout_vert, self.dout_para], axis = 1)
        self.dout = tf.reshape(self.dout, [nframes, natoms[0] * self.get_dim_out()])
        self.qmat = self.descrpt_vert.qmat + self.descrpt_para.qmat

        tf.summary.histogram('embedding_net_output', self.dout)

        return self.dout

    def prod_force_virial(self, atom_ener, natoms) :
        f_vert, v_vert, av_vert \
            = self.descrpt_vert.prod_force_virial(atom_ener, natoms)
        f_para, v_para, av_para \
            = self.descrpt_para.prod_force_virial(atom_ener, natoms)
        force = f_vert + f_para
        virial = v_vert + v_para
        atom_vir = av_vert + av_para
        return force, virial, atom_vir


class DescrptSeAEfLower (DescrptSeA):
    def __init__ (self, jdata, op):
        DescrptSeA.__init__(self, jdata)
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
        self.op = op

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

        add_data_requirement('efield', 3, atomic=True, must=True, high_prec=False)

        self.place_holders = {}
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = 'd_sea_ef_'
            for ii in ['coord', 'box']:
                self.place_holders[ii] = tf.placeholder(global_np_float_precision, [None, None], name = name_pfx+'t_'+ii)
            self.place_holders['type'] = tf.placeholder(tf.int32, [None, None], name=name_pfx+'t_type')
            self.place_holders['natoms_vec'] = tf.placeholder(tf.int32, [self.ntypes+2], name=name_pfx+'t_natoms')
            self.place_holders['default_mesh'] = tf.placeholder(tf.int32, [None], name=name_pfx+'t_mesh')
            self.place_holders['efield'] = tf.placeholder(global_np_float_precision, [None, None], name=name_pfx+'t_efield')
            self.stat_descrpt, descrpt_deriv, rij, nlist \
                = self.op(self.place_holders['coord'],
                          self.place_holders['type'],
                          self.place_holders['natoms_vec'],
                          self.place_holders['box'],
                          self.place_holders['default_mesh'],
                          self.place_holders['efield'],
                          tf.constant(avg_zero),
                          tf.constant(std_ones),
                          rcut_a = self.rcut_a,
                          rcut_r = self.rcut_r,
                          rcut_r_smth = self.rcut_r_smth,
                          sel_a = self.sel_a,
                          sel_r = self.sel_r)
        self.sub_sess = tf.Session(graph = sub_graph, config=default_tf_session_config)



    def compute_input_stats (self,
                             data_coord, 
                             data_box, 
                             data_atype, 
                             natoms_vec,
                             mesh, 
                             input_dict) :
        data_efield = input_dict['efield']
        all_davg = []
        all_dstd = []
        if True:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc,bb,tt,nn,mm,ee in zip(data_coord,data_box,data_atype,natoms_vec,mesh,data_efield) :
                sysr,sysr2,sysa,sysa2,sysn \
                    = self._compute_dstats_sys_smth(cc,bb,tt,nn,mm,ee)
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

        self.davg = np.array(all_davg)
        self.dstd = np.array(all_dstd)

    def _normalize_3d(self, a):
        na = tf.norm(a, axis = 1)
        na = tf.tile(tf.reshape(na, [-1,1]), tf.constant([1, 3]))
        return tf.divide(a, na)

    def build (self, 
               coord_, 
               atype_,
               natoms,
               box_, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        efield = input_dict['efield']
        davg = self.davg
        dstd = self.dstd
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
        efield = tf.reshape(efield, [-1, 3])
        efield = self._normalize_3d(efield)
        efield = tf.reshape(efield, [-1, natoms[0] * 3])

        self.descrpt, self.descrpt_deriv, self.rij, self.nlist \
            = self.op (coord,
                       atype,
                       natoms,
                       box,
                       mesh,
                       efield,
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

        # only used when tensorboard was set as true
        tf.summary.histogram('descrpt', self.descrpt)
        tf.summary.histogram('rij', self.rij)
        tf.summary.histogram('nlist', self.nlist)

        self.dout, self.qmat = self._pass_filter(self.descrpt_reshape, atype, natoms, input_dict, suffix = suffix, reuse = reuse, trainable = self.trainable)
        tf.summary.histogram('embedding_net_output', self.dout)

        return self.dout



    def _compute_dstats_sys_smth (self,
                                  data_coord, 
                                  data_box, 
                                  data_atype,                             
                                  natoms_vec,
                                  mesh,
                                  data_efield) :
        dd_all \
            = self.sub_sess.run(self.stat_descrpt, 
                                feed_dict = {
                                    self.place_holders['coord']: data_coord,
                                    self.place_holders['type']: data_atype,
                                    self.place_holders['natoms_vec']: natoms_vec,
                                    self.place_holders['box']: data_box,
                                    self.place_holders['default_mesh']: mesh,
                                    self.place_holders['efield']: data_efield,
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


