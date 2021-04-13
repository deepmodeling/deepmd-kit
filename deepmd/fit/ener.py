import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement, get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter
from deepmd.utils.argcheck import list_to_doc
from deepmd.utils.network import one_layer
from deepmd.descriptor import DescrptLocFrame
from deepmd.descriptor import DescrptSeA

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

class EnerFitting ():
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  numb_fparam : int = 0,
                  numb_aparam : int = 0,
                  rcond : float = 1e-3,
                  tot_ener_zero : bool = False,
                  trainable : List[bool] = None,
                  seed : int = 1,
                  atom_ener : List[float] = [],
                  activation_function : str = 'tanh',
                  precision : str = 'default'
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        descrpt
                The descrptor
        neuron
                Number of neurons in each hidden layer of the fitting net
        resnet_dt
                Time-step `dt` in the resnet construction:
                y = x + dt * \phi (Wx + b)
        numb_fparam
                Number of frame parameter
        numb_aparam
                Number of atomic parameter
        rcond
                The condition number for the regression of atomic energy.
        tot_ener_zero
                Force the total energy to zero. Useful for the charge fitting.
        trainable
                If the weights of fitting net are trainable. 
                Suppose that we have N_l hidden layers in the fitting net, 
                this list is of length N_l + 1, specifying if the hidden layers and the output layer are trainable.
        seed
                Random seed for initializing the network parameters.
        atom_ener
                Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.
        activation_function
                The activation function in the embedding net. Supported options are {0}
        precision
                The precision of the embedding net parameters. Supported options are {1}                
        """
        # model param
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        # args = ()\
        #        .add('numb_fparam',      int,    default = 0)\
        #        .add('numb_aparam',      int,    default = 0)\
        #        .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
        #        .add('resnet_dt',        bool,   default = True)\
        #        .add('rcond',            float,  default = 1e-3) \
        #        .add('tot_ener_zero',    bool,   default = False) \
        #        .add('seed',             int)               \
        #        .add('atom_ener',        list,   default = [])\
        #        .add("activation_function", str,    default = "tanh")\
        #        .add("precision",           str, default = "default")\
        #        .add("trainable",        [list, bool], default = True)
        self.numb_fparam = numb_fparam
        self.numb_aparam = numb_aparam
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.rcond = rcond
        self.seed = seed
        self.tot_ener_zero = tot_ener_zero
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        self.trainable = trainable
        if self.trainable is None:
            self.trainable = [True for ii in range(len(self.n_neuron) + 1)]
        if type(self.trainable) is bool:
            self.trainable = [self.trainable] * (len(self.n_neuron)+1)
        assert(len(self.trainable) == len(self.n_neuron) + 1), 'length of trainable should be that of n_neuron + 1'
        self.atom_ener = []
        for at, ae in enumerate(atom_ener):
            if ae is not None:
                self.atom_ener.append(tf.constant(ae, GLOBAL_TF_FLOAT_PRECISION, name = "atom_%d_ener" % at))
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

    def get_numb_fparam(self) -> int:
        """
        Get the number of frame parameters
        """
        return self.numb_fparam

    def get_numb_aparam(self) -> int:
        """
        Get the number of atomic parameters
        """
        return self.numb_fparam

    def compute_output_stats(self, 
                             all_stat: dict
    ) -> None:
        """
        Compute the ouput statistics

        Parameters
        ----------
        all_stat
                must have the following components:
                all_stat['energy'] of shape n_sys x n_batch x n_frame
                can be prepared by model.make_stat_input
        """
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

    def compute_input_stats(self, 
                            all_stat : dict,
                            protection : float = 1e-2) -> None:
        """
        Compute the input statistics

        Parameters:
        all_stat
                if numb_fparam > 0 must have all_stat['fparam']
                if numb_aparam > 0 must have all_stat['aparam']
                can be prepared by model.make_stat_input
        protection
                Divided-by-zero protection
        """
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
               inputs : tf.Tensor,
               natoms : tf.Tensor,
               input_dict : dict = {},
               reuse : bool = None,
               suffix : str = ''
    ) -> tf.Tensor:
        """
        Build the computational graph for fitting net

        Parameters
        ----------
        inputs
                The input descriptor
        input_dict
                Additional dict for inputs. 
                if numb_fparam > 0, should have input_dict['fparam']
                if numb_aparam > 0, should have input_dict['aparam']
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
        ener
                The system energy
        """
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
                                               dtype = GLOBAL_TF_FLOAT_PRECISION,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.fparam_avg))
                t_fparam_istd = tf.get_variable('t_fparam_istd', 
                                                self.numb_fparam,
                                                dtype = GLOBAL_TF_FLOAT_PRECISION,
                                                trainable = False,
                                                initializer = tf.constant_initializer(self.fparam_inv_std))
            if self.numb_aparam > 0: 
                t_aparam_avg = tf.get_variable('t_aparam_avg', 
                                               self.numb_aparam,
                                               dtype = GLOBAL_TF_FLOAT_PRECISION,
                                               trainable = False,
                                               initializer = tf.constant_initializer(self.aparam_avg))
                t_aparam_istd = tf.get_variable('t_aparam_istd', 
                                                self.numb_aparam,
                                                dtype = GLOBAL_TF_FLOAT_PRECISION,
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
                inputs_zero = tf.zeros_like(inputs_i, dtype=GLOBAL_TF_FLOAT_PRECISION)
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

        if self.tot_ener_zero:
            force_tot_ener = 0.0
            outs = tf.reshape(outs, [-1, natoms[0]])
            outs_mean = tf.reshape(tf.reduce_mean(outs, axis = 1), [-1, 1])
            outs_mean = outs_mean - tf.ones_like(outs_mean, dtype = GLOBAL_TF_FLOAT_PRECISION) * (force_tot_ener/global_cvt_2_tf_float(natoms[0]))
            outs = outs - outs_mean
            outs = tf.reshape(outs, [-1])

        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]), GLOBAL_TF_FLOAT_PRECISION)        





