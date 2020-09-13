import warnings
import numpy as np

from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement, get_activation_func, get_precision
from deepmd.Network import one_layer
from deepmd.DescrptLocFrame import DescrptLocFrame
from deepmd.DescrptSeA import DescrptSeA

from deepmd.RunOptions import global_tf_float_precision

class EnerFitting ():
    def __init__ (self, jdata, descrpt):
        # model param
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        args = ClassArg()\
               .add('numb_fparam',      int,    default = 0)\
               .add('numb_aparam',      int,    default = 0)\
               .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
               .add('resnet_dt',        bool,   default = True)\
               .add('rcond',            float,  default = 1e-3) \
               .add('seed',             int)               \
               .add('atom_ener',        list,   default = [])\
               .add("activation_function", str,    default = "tanh")\
               .add("precision",           str, default = "default")\
               .add("trainable",        [list, bool], default = True)
        class_data = args.parse(jdata)
        self.numb_fparam = class_data['numb_fparam']
        self.numb_aparam = class_data['numb_aparam']
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.rcond = class_data['rcond']
        self.seed = class_data['seed']
        self.fitting_activation_fn = get_activation_func(class_data["activation_function"])
        self.fitting_precision = get_precision(class_data['precision'])
        self.trainable = class_data['trainable']
        if type(self.trainable) is bool:
            self.trainable = [self.trainable] * (len(self.n_neuron)+1)
        assert(len(self.trainable) == len(self.n_neuron) + 1), 'length of trainable should be that of n_neuron + 1'
        self.atom_ener = []
        for at, ae in enumerate(class_data['atom_ener']):
            if ae is not None:
                self.atom_ener.append(tf.constant(ae, global_tf_float_precision, name = "atom_%d_ener" % at))
            else:
                self.atom_ener.append(None)
        self.useBN = False
        self.bias_atom_e = None
        # data requirement
        if self.numb_fparam > 0 :
            add_data_requirement('fparam', self.numb_fparam, atomic=False, must=True, high_prec=False)
            self.fparam_avg = None
            self.fparam_std = None
            self.fparam_inv_std = None
        if self.numb_aparam > 0:
            add_data_requirement('aparam', self.numb_aparam, atomic=True,  must=True, high_prec=False)
            self.aparam_avg = None
            self.aparam_std = None
            self.aparam_inv_std = None

    def get_numb_fparam(self) :
        return self.numb_fparam

    def get_numb_aparam(self) :
        return self.numb_fparam

    def compute_output_stats(self, all_stat):
        self.bias_atom_e = self._compute_output_stats(all_stat, rcond = self.rcond)

    @classmethod
    def _compute_output_stats(self, all_stat, rcond = 1e-3):
        data = all_stat['energy']
        # data[sys_idx][batch_idx][frame_idx]
        sys_ener = np.array([])
        for ss in range(len(data)):
            sys_data = []
            for ii in range(len(data[ss])):
                for jj in range(len(data[ss][ii])):
                    sys_data.append(data[ss][ii][jj])
            sys_data = np.concatenate(sys_data)
            sys_ener = np.append(sys_ener, np.average(sys_data))
        data = all_stat['natoms_vec']
        sys_tynatom = np.array([])
        nsys = len(data)
        for ss in range(len(data)):
            sys_tynatom = np.append(sys_tynatom, data[ss][0].astype(np.float64))
        sys_tynatom = np.reshape(sys_tynatom, [nsys,-1])
        sys_tynatom = sys_tynatom[:,2:]
        energy_shift,resd,rank,s_value \
            = np.linalg.lstsq(sys_tynatom, sys_ener, rcond = rcond)
        return energy_shift    

    def compute_input_stats(self, all_stat, protection):
        # stat fparam
        if self.numb_fparam > 0:
            cat_data = np.concatenate(all_stat['fparam'], axis = 0)
            cat_data = np.reshape(cat_data, [-1, self.numb_fparam])
            self.fparam_avg = np.average(cat_data, axis = 0)
            self.fparam_std = np.std(cat_data, axis = 0)
            for ii in range(self.fparam_std.size):
                if self.fparam_std[ii] < protection:
                    self.fparam_std[ii] = protection
            self.fparam_inv_std = 1./self.fparam_std
        # stat aparam
        if self.numb_aparam > 0:
            sys_sumv = []
            sys_sumv2 = []
            sys_sumn = []
            for ss_ in all_stat['aparam'] : 
                ss = np.reshape(ss_, [-1, self.numb_aparam])
                sys_sumv.append(np.sum(ss, axis = 0))
                sys_sumv2.append(np.sum(np.multiply(ss, ss), axis = 0))
                sys_sumn.append(ss.shape[0])
            sumv = np.sum(sys_sumv, axis = 0)
            sumv2 = np.sum(sys_sumv2, axis = 0)
            sumn = np.sum(sys_sumn)
            self.aparam_avg = (sumv)/sumn
            self.aparam_std = self._compute_std(sumv2, sumv, sumn)
            for ii in range(self.aparam_std.size):
                if self.aparam_std[ii] < protection:
                    self.aparam_std[ii] = protection
            self.aparam_inv_std = 1./self.aparam_std


    def _compute_std (self, sumv2, sumv, sumn) :
        return np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))
            

    def build (self, 
               inputs,
               input_dict,
               natoms,
               reuse = None,
               suffix = '') :
        bias_atom_e = self.bias_atom_e
        if self.numb_fparam > 0 and ( self.fparam_avg is None or self.fparam_inv_std is None ):
            raise RuntimeError('No data stat result. one should do data statisitic, before build')
        if self.numb_aparam > 0 and ( self.aparam_avg is None or self.aparam_inv_std is None ):
            raise RuntimeError('No data stat result. one should do data statisitic, before build')

        with tf.variable_scope('fitting_attr' + suffix, reuse = reuse) :
            t_dfparam = tf.constant(self.numb_fparam, 
                                    name = 'dfparam', 
                                    dtype = tf.int32)
            t_daparam = tf.constant(self.numb_aparam, 
                                    name = 'daparam', 
                                    dtype = tf.int32)
            if self.numb_fparam > 0: 
                t_fparam_avg = tf.get_variable('t_fparam_avg', 
                                               self.numb_fparam,
                                               dtype = global_tf_float_precision,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.fparam_avg))
                t_fparam_istd = tf.get_variable('t_fparam_istd', 
                                                self.numb_fparam,
                                                dtype = global_tf_float_precision,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.fparam_inv_std))
            if self.numb_aparam > 0: 
                t_aparam_avg = tf.get_variable('t_aparam_avg', 
                                               self.numb_aparam,
                                               dtype = global_tf_float_precision,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.aparam_avg))
                t_aparam_istd = tf.get_variable('t_aparam_istd', 
                                                self.numb_aparam,
                                                dtype = global_tf_float_precision,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.aparam_inv_std))
            
        start_index = 0
        inputs = tf.cast(tf.reshape(inputs, [-1, self.dim_descrpt * natoms[0]]), self.fitting_precision)

        if bias_atom_e is not None :
            assert(len(bias_atom_e) == self.ntypes)

        if self.numb_fparam > 0 :
            fparam = input_dict['fparam']
            fparam = tf.reshape(fparam, [-1, self.numb_fparam])
            fparam = (fparam - t_fparam_avg) * t_fparam_istd            
        if self.numb_aparam > 0 :
            aparam = input_dict['aparam']
            aparam = tf.reshape(aparam, [-1, self.numb_aparam])
            aparam = (aparam - t_aparam_avg) * t_aparam_istd
            aparam = tf.reshape(aparam, [-1, self.numb_aparam * natoms[0]])

        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.dim_descrpt],
                                 [-1, natoms[2+type_i]* self.dim_descrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            layer = inputs_i
            if self.numb_fparam > 0 :
                ext_fparam = tf.tile(fparam, [1, natoms[2+type_i]])
                ext_fparam = tf.reshape(ext_fparam, [-1, self.numb_fparam])
                ext_fparam = tf.cast(ext_fparam,self.fitting_precision)
                layer = tf.concat([layer, ext_fparam], axis = 1)
            if self.numb_aparam > 0 :
                ext_aparam = tf.slice(aparam, 
                                      [ 0, start_index      * self.numb_aparam],
                                      [-1, natoms[2+type_i] * self.numb_aparam])
                ext_aparam = tf.reshape(ext_aparam, [-1, self.numb_aparam])
                ext_aparam = tf.cast(ext_aparam,self.fitting_precision)
                layer = tf.concat([layer, ext_aparam], axis = 1)
            start_index += natoms[2+type_i]
                
            if bias_atom_e is None :
                type_bias_ae = 0.0
            else :
                type_bias_ae = bias_atom_e[type_i]

            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, trainable = self.trainable[ii])
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, trainable = self.trainable[ii])
            final_layer = one_layer(layer, 1, activation_fn = None, bavg = type_bias_ae, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, precision = self.fitting_precision, trainable = self.trainable[-1])

            if type_i < len(self.atom_ener) and self.atom_ener[type_i] is not None:
                inputs_zero = tf.zeros_like(inputs_i, dtype=global_tf_float_precision)
                layer = inputs_zero
                if self.numb_fparam > 0 :
                    layer = tf.concat([layer, ext_fparam], axis = 1)
                if self.numb_aparam > 0 :
                    layer = tf.concat([layer, ext_aparam], axis = 1)
                for ii in range(0,len(self.n_neuron)) :
                    if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                        layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=True, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, trainable = self.trainable[ii])
                    else :
                        layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=True, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, trainable = self.trainable[ii])
                zero_layer = one_layer(layer, 1, activation_fn = None, bavg = type_bias_ae, name='final_layer_type_'+str(type_i)+suffix, reuse=True, seed = self.seed, precision = self.fitting_precision, trainable = self.trainable[-1])
                final_layer += self.atom_ener[type_i] - zero_layer

            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i]])

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)

        return tf.cast(tf.reshape(outs, [-1]), global_tf_float_precision)        


class WFCFitting () :
    def __init__ (self, jdata, descrpt) :
        if not isinstance(descrpt, DescrptLocFrame) :
            raise RuntimeError('WFC only supports DescrptLocFrame')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        args = ClassArg()\
               .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
               .add('resnet_dt',        bool,   default = True)\
               .add('wfc_numb',         int,    must = True)\
               .add('sel_type',         [list,int],   default = [ii for ii in range(self.ntypes)], alias = 'wfc_type')\
               .add('seed',             int)\
               .add("activation_function", str, default = "tanh")\
               .add('precision',           str,    default = "default")
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.wfc_numb = class_data['wfc_numb']
        self.sel_type = class_data['sel_type']
        self.seed = class_data['seed']
        self.fitting_activation_fn = get_activation_func(class_data["activation_function"])
        self.fitting_precision = get_precision(class_data['precision'])
        self.useBN = False


    def get_sel_type(self):
        return self.sel_type

    def get_wfc_numb(self):
        return self.wfc_numb

    def get_out_size(self):
        return self.wfc_numb * 3

    def build (self, 
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') :
        start_index = 0
        inputs = tf.cast(tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]]), self.fitting_precision)
        rot_mat = tf.reshape(rot_mat, [-1, 9 * natoms[0]])

        count = 0
        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.dim_descrpt],
                                 [-1, natoms[2+type_i]* self.dim_descrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            rot_mat_i = tf.slice (rot_mat,
                                  [ 0, start_index*      9],
                                  [-1, natoms[2+type_i]* 9] )
            rot_mat_i = tf.reshape(rot_mat_i, [-1, 3, 3])
            start_index += natoms[2+type_i]
            if not type_i in self.sel_type :
                continue
            layer = inputs_i
            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
            # (nframes x natoms) x (nwfc x 3)
            final_layer = one_layer(layer, self.wfc_numb * 3, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, precision = self.fitting_precision)
            # (nframes x natoms) x nwfc(wc) x 3(coord_local)
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms[2+type_i], self.wfc_numb, 3])
            # (nframes x natoms) x nwfc(wc) x 3(coord)
            final_layer = tf.matmul(final_layer, rot_mat_i)
            # nframes x natoms x nwfc(wc) x 3(coord_local)
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i], self.wfc_numb, 3])

            # concat the results
            if count == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)
            count += 1

        return tf.cast(tf.reshape(outs, [-1]),  global_tf_float_precision)



class PolarFittingLocFrame () :
    def __init__ (self, jdata, descrpt) :
        if not isinstance(descrpt, DescrptLocFrame) :
            raise RuntimeError('PolarFittingLocFrame only supports DescrptLocFrame')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        args = ClassArg()\
               .add('neuron',           list, default = [120,120,120], alias = 'n_neuron')\
               .add('resnet_dt',        bool, default = True)\
               .add('sel_type',         [list,int], default = [ii for ii in range(self.ntypes)], alias = 'pol_type')\
               .add('seed',             int)\
               .add("activation_function", str, default = "tanh")\
               .add('precision',           str,    default = "default")    
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.sel_type = class_data['sel_type']
        self.seed = class_data['seed']
        self.fitting_activation_fn = get_activation_func(class_data["activation_function"])
        self.fitting_precision = get_precision(class_data['precision'])
        self.useBN = False

    def get_sel_type(self):
        return self.sel_type

    def get_out_size(self):
        return 9

    def build (self, 
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') :
        start_index = 0
        inputs = tf.cast(tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]]), self.fitting_precision)
        rot_mat = tf.reshape(rot_mat, [-1, 9 * natoms[0]])

        count = 0
        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.dim_descrpt],
                                 [-1, natoms[2+type_i]* self.dim_descrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            rot_mat_i = tf.slice (rot_mat,
                                  [ 0, start_index*      9],
                                  [-1, natoms[2+type_i]* 9] )
            rot_mat_i = tf.reshape(rot_mat_i, [-1, 3, 3])
            start_index += natoms[2+type_i]
            if not type_i in self.sel_type :
                continue
            layer = inputs_i
            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
            # (nframes x natoms) x 9
            final_layer = one_layer(layer, 9, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, precision = self.fitting_precision)
            # (nframes x natoms) x 3 x 3
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms[2+type_i], 3, 3])
            # (nframes x natoms) x 3 x 3
            final_layer = final_layer + tf.transpose(final_layer, perm = [0,2,1])
            # (nframes x natoms) x 3 x 3(coord)
            final_layer = tf.matmul(final_layer, rot_mat_i)
            # (nframes x natoms) x 3(coord) x 3(coord)
            final_layer = tf.matmul(rot_mat_i, final_layer, transpose_a = True)
            # nframes x natoms x 3 x 3
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i], 3, 3])

            # concat the results
            if count == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)
            count += 1

        return tf.cast(tf.reshape(outs, [-1]),  global_tf_float_precision)


class PolarFittingSeA () :
    def __init__ (self, jdata, descrpt) :
        if not isinstance(descrpt, DescrptSeA) :
            raise RuntimeError('PolarFittingSeA only supports DescrptSeA')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        args = ClassArg()\
               .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
               .add('resnet_dt',        bool,   default = True)\
               .add('fit_diag',         bool,   default = True)\
               .add('diag_shift',       [list,float], default = [0.0 for ii in range(self.ntypes)])\
               .add('scale',            [list,float], default = [1.0 for ii in range(self.ntypes)])\
               .add('sel_type',         [list,int],   default = [ii for ii in range(self.ntypes)], alias = 'pol_type')\
               .add('seed',             int)\
               .add("activation_function", str ,   default = "tanh")\
               .add('precision',           str,    default = "default")
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.sel_type = class_data['sel_type']
        self.fit_diag = class_data['fit_diag']
        self.seed = class_data['seed']
        self.diag_shift = class_data['diag_shift']
        self.scale = class_data['scale']
        self.fitting_activation_fn = get_activation_func(class_data["activation_function"])
        self.fitting_precision = get_precision(class_data['precision'])
        if type(self.sel_type) is not list:
            self.sel_type = [self.sel_type]
        if type(self.diag_shift) is not list:
            self.diag_shift = [self.diag_shift]
        if type(self.scale) is not list:
            self.scale = [self.scale]
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False

    def get_sel_type(self):
        return self.sel_type

    def get_out_size(self):
        return 9

    def compute_input_stats(self, all_stat, protection = 1e-2):
        if not ('polarizability' in all_stat.keys()):
            self.avgeig = np.zeros([9])
            warnings.warn('no polarizability data, cannot do data stat. use zeros as guess')
            return
        data = all_stat['polarizability']
        all_tmp = []
        for ss in range(len(data)):
            tmp = np.concatenate(data[ss], axis = 0)
            tmp = np.reshape(tmp, [-1, 3, 3])
            tmp,_ = np.linalg.eig(tmp)
            tmp = np.absolute(tmp)
            tmp = np.sort(tmp, axis = 1)
            all_tmp.append(tmp)
        all_tmp = np.concatenate(all_tmp, axis = 1)
        self.avgeig = np.average(all_tmp, axis = 0)

    def build (self, 
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') :
        start_index = 0
        inputs = tf.cast(tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]]), self.fitting_precision)
        rot_mat = tf.reshape(rot_mat, [-1, self.dim_rot_mat * natoms[0]])

        count = 0
        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.dim_descrpt],
                                 [-1, natoms[2+type_i]* self.dim_descrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            rot_mat_i = tf.slice (rot_mat,
                                  [ 0, start_index*      self.dim_rot_mat],
                                  [-1, natoms[2+type_i]* self.dim_rot_mat] )
            rot_mat_i = tf.reshape(rot_mat_i, [-1, self.dim_rot_mat_1, 3])
            start_index += natoms[2+type_i]
            if not type_i in self.sel_type :
                continue
            layer = inputs_i
            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
            if self.fit_diag :
                bavg = np.zeros(self.dim_rot_mat_1)
                # bavg[0] = self.avgeig[0]
                # bavg[1] = self.avgeig[1]
                # bavg[2] = self.avgeig[2]
                # (nframes x natoms) x naxis
                final_layer = one_layer(layer, self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, bavg = bavg, precision = self.fitting_precision)
                # (nframes x natoms) x naxis
                final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms[2+type_i], self.dim_rot_mat_1])
                # (nframes x natoms) x naxis x naxis
                final_layer = tf.matrix_diag(final_layer)                
            else :
                bavg = np.zeros(self.dim_rot_mat_1*self.dim_rot_mat_1)
                # bavg[0*self.dim_rot_mat_1+0] = self.avgeig[0]
                # bavg[1*self.dim_rot_mat_1+1] = self.avgeig[1]
                # bavg[2*self.dim_rot_mat_1+2] = self.avgeig[2]
                # (nframes x natoms) x (naxis x naxis)
                final_layer = one_layer(layer, self.dim_rot_mat_1*self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, bavg = bavg, precision = self.fitting_precision)
                # (nframes x natoms) x naxis x naxis
                final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms[2+type_i], self.dim_rot_mat_1, self.dim_rot_mat_1])
                # (nframes x natoms) x naxis x naxis
                final_layer = final_layer + tf.transpose(final_layer, perm = [0,2,1])
            # (nframes x natoms) x naxis x 3(coord)
            final_layer = tf.matmul(final_layer, rot_mat_i)
            # (nframes x natoms) x 3(coord) x 3(coord)
            final_layer = tf.matmul(rot_mat_i, final_layer, transpose_a = True)
            # nframes x natoms x 3 x 3
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i], 3, 3])
            # shift and scale
            sel_type_idx = self.sel_type.index(type_i)
            final_layer = final_layer * self.scale[sel_type_idx]
            final_layer = final_layer + self.diag_shift[sel_type_idx] * tf.eye(3, batch_shape=[tf.shape(inputs)[0], natoms[2+type_i]], dtype = global_tf_float_precision)

            # concat the results
            if count == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)
            count += 1

        return tf.cast(tf.reshape(outs, [-1]), global_tf_float_precision)


class GlobalPolarFittingSeA () :
    def __init__ (self, jdata, descrpt) :
        if not isinstance(descrpt, DescrptSeA) :
            raise RuntimeError('GlobalPolarFittingSeA only supports DescrptSeA')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        self.polar_fitting = PolarFittingSeA(jdata, descrpt)

    def get_sel_type(self):
        return self.polar_fitting.get_sel_type()

    def get_out_size(self):
        return self.polar_fitting.get_out_size()

    def build (self,
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') :
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        outs = self.polar_fitting.build(input_d, rot_mat, natoms, reuse, suffix)
        # nframes x natoms x 9
        outs = tf.reshape(outs, [tf.shape(inputs)[0], -1, 9])
        outs = tf.reduce_sum(outs, axis = 1)
        return tf.reshape(outs, [-1])


class DipoleFittingSeA () :
    def __init__ (self, jdata, descrpt) :
        if not isinstance(descrpt, DescrptSeA) :
            raise RuntimeError('DipoleFittingSeA only supports DescrptSeA')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        args = ClassArg()\
               .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
               .add('resnet_dt',        bool,   default = True)\
               .add('sel_type',         [list,int],   default = [ii for ii in range(self.ntypes)], alias = 'dipole_type')\
               .add('seed',             int)\
               .add("activation_function", str, default = "tanh")\
               .add('precision',           str,    default = "default")
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.sel_type = class_data['sel_type']
        self.seed = class_data['seed']
        self.fitting_activation_fn = get_activation_func(class_data["activation_function"])
        self.fitting_precision = get_precision(class_data['precision'])
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False

    def get_sel_type(self):
        return self.sel_type

    def get_out_size(self):
        return 3

    def build (self, 
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') :
        start_index = 0
        inputs = tf.cast(tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]]), self.fitting_precision)
        rot_mat = tf.reshape(rot_mat, [-1, self.dim_rot_mat * natoms[0]])

        count = 0
        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.dim_descrpt],
                                 [-1, natoms[2+type_i]* self.dim_descrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            rot_mat_i = tf.slice (rot_mat,
                                  [ 0, start_index*      self.dim_rot_mat],
                                  [-1, natoms[2+type_i]* self.dim_rot_mat] )
            rot_mat_i = tf.reshape(rot_mat_i, [-1, self.dim_rot_mat_1, 3])
            start_index += natoms[2+type_i]
            if not type_i in self.sel_type :
                continue
            layer = inputs_i
            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision)
            # (nframes x natoms) x naxis
            final_layer = one_layer(layer, self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, precision = self.fitting_precision)
            # (nframes x natoms) x 1 * naxis
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms[2+type_i], 1, self.dim_rot_mat_1])
            # (nframes x natoms) x 1 x 3(coord)
            final_layer = tf.matmul(final_layer, rot_mat_i)
            # nframes x natoms x 3
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i], 3])

            # concat the results
            if count == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)
            count += 1

        return tf.cast(tf.reshape(outs, [-1]),  global_tf_float_precision)
        # return tf.reshape(outs, [tf.shape(inputs)[0] * natoms[0] * 3 // 3])
