import os,warnings
import numpy as np

from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement
from deepmd.Network import one_layer
from deepmd.DescrptLocFrame import DescrptLocFrame
from deepmd.DescrptSeA import DescrptSeA

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

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
               .add('seed',             int)               
        class_data = args.parse(jdata)
        self.numb_fparam = class_data['numb_fparam']
        self.numb_aparam = class_data['numb_aparam']
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.seed = class_data['seed']
        self.useBN = False
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

    def compute_dstats(self, all_stat, protection):
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
               bias_atom_e = None,
               reuse = None,
               suffix = '') :
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
        inputs = tf.reshape(inputs, [-1, self.dim_descrpt * natoms[0]])
        shape = inputs.get_shape().as_list()

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
                layer = tf.concat([layer, ext_fparam], axis = 1)
            if self.numb_aparam > 0 :
                ext_aparam = tf.slice(aparam, 
                                      [ 0, start_index      * self.numb_aparam],
                                      [-1, natoms[2+type_i] * self.numb_aparam])
                ext_aparam = tf.reshape(ext_aparam, [-1, self.numb_aparam])
                layer = tf.concat([layer, ext_aparam], axis = 1)
            start_index += natoms[2+type_i]
                
            if bias_atom_e is None :
                type_bias_ae = 0.0
            else :
                type_bias_ae = bias_atom_e[type_i]

            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            final_layer = one_layer(layer, 1, activation_fn = None, bavg = type_bias_ae, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0], natoms[2+type_i]])

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)

        return tf.reshape(outs, [-1])        


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
               .add('seed',             int)
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.wfc_numb = class_data['wfc_numb']
        self.sel_type = class_data['sel_type']
        self.seed = class_data['seed']
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
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        rot_mat = tf.reshape(rot_mat, [-1, 9 * natoms[0]])
        shape = inputs.get_shape().as_list()

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
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            # (nframes x natoms) x (nwfc x 3)
            final_layer = one_layer(layer, self.wfc_numb * 3, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
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

        return tf.reshape(outs, [-1])



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
               .add('seed',             int)
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.sel_type = class_data['sel_type']
        self.seed = class_data['seed']
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
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        rot_mat = tf.reshape(rot_mat, [-1, 9 * natoms[0]])
        shape = inputs.get_shape().as_list()

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
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            # (nframes x natoms) x 9
            final_layer = one_layer(layer, 9, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
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

        return tf.reshape(outs, [-1])


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
               .add('sel_type',         [list,int],   default = [ii for ii in range(self.ntypes)], alias = 'pol_type')\
               .add('seed',             int)
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.sel_type = class_data['sel_type']
        self.fit_diag = class_data['fit_diag']
        self.seed = class_data['seed']
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
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
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        rot_mat = tf.reshape(rot_mat, [-1, self.dim_rot_mat * natoms[0]])
        shape = inputs.get_shape().as_list()

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
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            if self.fit_diag :
                # (nframes x natoms) x naxis
                final_layer = one_layer(layer, self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
                # (nframes x natoms) x naxis
                final_layer = tf.reshape(final_layer, [tf.shape(inputs)[0] * natoms[2+type_i], self.dim_rot_mat_1])
                # (nframes x natoms) x naxis x naxis
                final_layer = tf.matrix_diag(final_layer)                
            else :                
                # (nframes x natoms) x (naxis x naxis)
                final_layer = one_layer(layer, self.dim_rot_mat_1*self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
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

            # concat the results
            if count == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)
            count += 1

        return tf.reshape(outs, [-1])


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
               .add('seed',             int)
        class_data = args.parse(jdata)
        self.n_neuron = class_data['neuron']
        self.resnet_dt = class_data['resnet_dt']
        self.sel_type = class_data['sel_type']
        self.seed = class_data['seed']
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False

    def get_sel_type(self):
        return self.sel_type

    def build (self, 
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') :
        start_index = 0
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        rot_mat = tf.reshape(rot_mat, [-1, self.dim_rot_mat * natoms[0]])
        shape = inputs.get_shape().as_list()

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
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            # (nframes x natoms) x naxis
            final_layer = one_layer(layer, self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
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

        return tf.reshape(outs, [-1])
