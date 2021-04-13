import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import add_data_requirement, get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter
from deepmd.utils.argcheck import list_to_doc
from deepmd.utils.network import one_layer
from deepmd.descriptor import DescrptSeA

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

class DipoleFittingSeA () :
    """
    Fit the atomic dipole with descriptor se_a
    """
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120], 
                  resnet_dt : bool = True,
                  sel_type : List[int] = None,
                  seed : int = 1,
                  activation_function : str = 'tanh',
                  precision : str = 'default'                  
    ) :
        """
        Constructor

        Parameters
        ----------
        descrpt : tf.Tensor
                The descrptor
        neuron : List[int]
                Number of neurons in each hidden layer of the fitting net
        resnet_dt : bool
                Time-step `dt` in the resnet construction:
                y = x + dt * \phi (Wx + b)
        sel_type : List[int]
                The atom types selected to have an atomic dipole prediction. If is None, all atoms are selected.
        seed : int
                Random seed for initializing the network parameters.
        activation_function : str
                The activation function in the embedding net. Supported options are {0}
        precision : str
                The precision of the embedding net parameters. Supported options are {1}        
        """
        if not isinstance(descrpt, DescrptSeA) :
            raise RuntimeError('DipoleFittingSeA only supports DescrptSeA')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        # args = ClassArg()\
        #        .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
        #        .add('resnet_dt',        bool,   default = True)\
        #        .add('sel_type',         [list,int],   default = [ii for ii in range(self.ntypes)], alias = 'dipole_type')\
        #        .add('seed',             int)\
        #        .add("activation_function", str, default = "tanh")\
        #        .add('precision',           str,    default = "default")
        # class_data = args.parse(jdata)
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.sel_type = sel_type
        if self.sel_type is None:
            self.sel_type = [ii for ii in range(self.ntypes)]
        self.sel_type = sel_type
        self.seed = seed
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False

    def get_sel_type(self) -> int:
        """
        Get selected type
        """
        return self.sel_type

    def get_out_size(self) -> int:
        """
        Get the output size. Should be 3
        """
        return 3

    def build (self, 
               input_d : tf.Tensor,
               rot_mat : tf.Tensor,
               natoms : tf.Tensor,
               reuse : bool = None,
               suffix : str = '') -> tf.Tensor:
        """
        Build the computational graph for fitting net
        
        Parameters
        ----------
        input_d
                The input descriptor
        rot_mat
                The rotation matrix from the descriptor.
        natoms
                The number of atoms. This tensor has the length of Ntypes + 2
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        reuse
                The weights in the networks should be reused when get the variable.
        suffix
                Name suffix to identify this descriptor

        Return
        ------
        dipole
                The atomic dipole.
        """
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

        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]),  GLOBAL_TF_FLOAT_PRECISION)
        # return tf.reshape(outs, [tf.shape(inputs)[0] * natoms[0] * 3 // 3])
