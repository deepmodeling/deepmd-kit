import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import ClassArg, add_data_requirement, get_activation_func, get_precision, activation_fn_dict, precision_dict, docstring_parameter
from deepmd.argcheck import list_to_doc
from deepmd.network import one_layer
from deepmd.descrpt_loc_frame import DescrptLocFrame
from deepmd.descrpt_se_a import DescrptSeA

from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_tf_float_precision

class EnerFitting ():
    @docstring_parameter(list_to_doc(activation_fn_dict.keys()), list_to_doc(precision_dict.keys()))
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
        # args = ClassArg()\
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

        if self.tot_ener_zero:
            force_tot_ener = 0.0
            outs = tf.reshape(outs, [-1, natoms[0]])
            outs_mean = tf.reshape(tf.reduce_mean(outs, axis = 1), [-1, 1])
            outs_mean = outs_mean - tf.ones_like(outs_mean, dtype = global_tf_float_precision) * (force_tot_ener/global_cvt_2_tf_float(natoms[0]))
            outs = outs - outs_mean
            outs = tf.reshape(outs, [-1])

        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]), global_tf_float_precision)        


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
        return tf.cast(tf.reshape(outs, [-1]),  global_tf_float_precision)



class PolarFittingLocFrame () :
    """
    Fitting polarizability with local frame descriptor. not supported anymore. 
    """
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

        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]),  global_tf_float_precision)


class PolarFittingSeA () :
    """
    Fit the atomic polarizability with descriptor se_a
    """
    @docstring_parameter(list_to_doc(activation_fn_dict.keys()), list_to_doc(precision_dict.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  sel_type : List[int] = None,
                  fit_diag : bool = True,
                  scale : List[float] = None,
                  diag_shift : List[float] = None,
                  seed : int = 1,
                  activation_function : str = 'tanh',
                  precision : str = 'default'                  
    ) -> None:
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
                The atom types selected to have an atomic polarizability prediction. If is None, all atoms are selected.
        fit_diag : bool
                Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.
        scale : List[float]
                The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
        diag_shift : List[float]
                The diagonal part of the polarizability matrix of type i will be shifted by diag_shift[i]. The shift operation is carried out after scale.        
        seed : int
                Random seed for initializing the network parameters.
        activation_function : str
                The activation function in the embedding net. Supported options are {0}
        precision : str
                The precision of the embedding net parameters. Supported options are {1}                
        """
        if not isinstance(descrpt, DescrptSeA) :
            raise RuntimeError('PolarFittingSeA only supports DescrptSeA')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        # args = ClassArg()\
        #        .add('neuron',           list,   default = [120,120,120], alias = 'n_neuron')\
        #        .add('resnet_dt',        bool,   default = True)\
        #        .add('fit_diag',         bool,   default = True)\
        #        .add('diag_shift',       [list,float], default = [0.0 for ii in range(self.ntypes)])\
        #        .add('scale',            [list,float], default = [1.0 for ii in range(self.ntypes)])\
        #        .add('sel_type',         [list,int],   default = [ii for ii in range(self.ntypes)], alias = 'pol_type')\
        #        .add('seed',             int)\
        #        .add("activation_function", str ,   default = "tanh")\
        #        .add('precision',           str,    default = "default")
        # class_data = args.parse(jdata)
        self.n_neuron = neuron
        self.resnet_dt = resnet_dt
        self.sel_type = sel_type
        self.fit_diag = fit_diag
        self.seed = seed
        self.diag_shift = diag_shift
        self.scale = scale
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        if self.sel_type is None:
            self.sel_type = [ii for ii in range(self.ntypes)]
        if self.scale is None:
            self.scale = [1.0 for ii in range(self.ntypes)]
        if self.diag_shift is None:
            self.diag_shift = [0.0 for ii in range(self.ntypes)]
        if type(self.sel_type) is not list:
            self.sel_type = [self.sel_type]
        if type(self.diag_shift) is not list:
            self.diag_shift = [self.diag_shift]
        if type(self.scale) is not list:
            self.scale = [self.scale]
        self.dim_rot_mat_1 = descrpt.get_dim_rot_mat_1()
        self.dim_rot_mat = self.dim_rot_mat_1 * 3
        self.useBN = False

    def get_sel_type(self) -> List[int]:
        """
        Get selected atom types
        """
        return self.sel_type

    def get_out_size(self) -> int:
        """
        Get the output size. Should be 9
        """
        return 9

    def compute_input_stats(self, 
                            all_stat, 
                            protection = 1e-2):
        """
        Compute the input statistics

        Parameters:
        all_stat
                Dictionary of inputs. 
                can be prepared by model.make_stat_input
        protection
                Divided-by-zero protection
        """
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
               input_d : tf.Tensor,
               rot_mat : tf.Tensor,
               natoms : tf.Tensor,
               reuse : bool = None,
               suffix : str = '') :
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
        atomic_polar
                The atomic polarizability        
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
        
        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]), global_tf_float_precision)


class GlobalPolarFittingSeA () :
    """
    Fit the system polarizability with descriptor se_a
    """
    @docstring_parameter(list_to_doc(activation_fn_dict.keys()), list_to_doc(precision_dict.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  sel_type : List[int] = None,
                  fit_diag : bool = True,
                  scale : List[float] = None,
                  diag_shift : List[float] = None,
                  seed : int = 1,
                  activation_function : str = 'tanh',
                  precision : str = 'default'                  
    ) -> None:
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
                The atom types selected to have an atomic polarizability prediction
        fit_diag : bool
                Fit the diagonal part of the rotational invariant polarizability matrix, which will be converted to normal polarizability matrix by contracting with the rotation matrix.
        scale : List[float]
                The output of the fitting net (polarizability matrix) for type i atom will be scaled by scale[i]
        diag_shift : List[float]
                The diagonal part of the polarizability matrix of type i will be shifted by diag_shift[i]. The shift operation is carried out after scale.        
        seed : int
                Random seed for initializing the network parameters.
        activation_function : str
                The activation function in the embedding net. Supported options are {0}
        precision : str
                The precision of the embedding net parameters. Supported options are {1}                
        """
        if not isinstance(descrpt, DescrptSeA) :
            raise RuntimeError('GlobalPolarFittingSeA only supports DescrptSeA')
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        self.polar_fitting = PolarFittingSeA(descrpt,
                                             neuron,
                                             resnet_dt,
                                             sel_type,
                                             fit_diag,
                                             scale,
                                             diag_shift,
                                             seed,
                                             activation_function,
                                             precision)

    def get_sel_type(self) -> int:
        """
        Get selected atom types
        """
        return self.polar_fitting.get_sel_type()

    def get_out_size(self) -> int:
        """
        Get the output size. Should be 9
        """
        return self.polar_fitting.get_out_size()

    def build (self,
               input_d,
               rot_mat,
               natoms,
               reuse = None,
               suffix = '') -> tf.Tensor:
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
        polar
                The system polarizability        
        """
        inputs = tf.reshape(input_d, [-1, self.dim_descrpt * natoms[0]])
        outs = self.polar_fitting.build(input_d, rot_mat, natoms, reuse, suffix)
        # nframes x natoms x 9
        outs = tf.reshape(outs, [tf.shape(inputs)[0], -1, 9])
        outs = tf.reduce_sum(outs, axis = 1)
        tf.summary.histogram('fitting_net_output', outs)
        return tf.reshape(outs, [-1])


class DipoleFittingSeA () :
    """
    Fit the atomic dipole with descriptor se_a
    """
    @docstring_parameter(list_to_doc(activation_fn_dict.keys()), list_to_doc(precision_dict.keys()))
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
        return tf.cast(tf.reshape(outs, [-1]),  global_tf_float_precision)
        # return tf.reshape(outs, [tf.shape(inputs)[0] * natoms[0] * 3 // 3])
