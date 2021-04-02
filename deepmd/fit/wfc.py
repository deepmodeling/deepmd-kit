import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement, get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter
from deepmd.utils.argcheck import list_to_doc
from deepmd.utils.network import one_layer
from deepmd.descriptor import DescrptLocFrame

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

class WFCFitting () :
    """
    Fitting Wannier function centers (WFCs) with local frame descriptor. Not supported anymore. 
    """
    def __init__ (self, jdata, descrpt):
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
        
        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]),  GLOBAL_TF_FLOAT_PRECISION)
