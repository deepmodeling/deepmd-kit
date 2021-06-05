import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import tf
from deepmd.common import add_data_requirement, get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter
from deepmd.utils.argcheck import list_to_doc
from deepmd.utils.network import one_layer, one_layer_rand_seed_shift
from deepmd.descriptor import DescrptLocFrame
from deepmd.descriptor import DescrptSeA

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION


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
        return tf.cast(tf.reshape(outs, [-1]),  GLOBAL_TF_FLOAT_PRECISION)


class PolarFittingSeA () :
    """
    Fit the atomic polarizability with descriptor se_a
    """
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  sel_type : List[int] = None,
                  fit_diag : bool = True,
                  scale : List[float] = None,
                  shift_diag : bool = True,     # YWolfeee: will support the user to decide whether to use this function
                  #diag_shift : List[float] = None, YWolfeee: will not support the user to assign a shift
                  seed : int = None,
                  activation_function : str = 'tanh',
                  precision : str = 'default',
                  uniform_seed: bool = False                  
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
        uniform_seed
                Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
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
        self.uniform_seed = uniform_seed
        self.seed_shift = one_layer_rand_seed_shift()
        #self.diag_shift = diag_shift
        self.shift_diag = shift_diag
        self.scale = scale
        self.fitting_activation_fn = get_activation_func(activation_function)
        self.fitting_precision = get_precision(precision)
        if self.sel_type is None:
            self.sel_type = [ii for ii in range(self.ntypes)]
        if self.scale is None:
            self.scale = [1.0 for ii in range(self.ntypes)]
        #if self.diag_shift is None:
        #    self.diag_shift = [0.0 for ii in range(self.ntypes)]
        if type(self.sel_type) is not list:
            self.sel_type = [self.sel_type]
        self.constant_matrix = np.zeros(len(self.sel_type)) # len(sel_type) x 1, store the average diagonal value
        #if type(self.diag_shift) is not list:
        #    self.diag_shift = [self.diag_shift]
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

        # YWolfeee: support polar normalization, initialize to a more appropriate point 
        if self.shift_diag:
            mean_polar = np.zeros([len(self.sel_type), 9])
            sys_matrix, polar_bias = [], []
            for ss in range(len(all_stat['type'])):
                atom_has_polar = [w for w in all_stat['type'][ss][0] if (w in self.sel_type)]   # select atom with polar
                if all_stat['find_atomic_polarizability'][ss] > 0.0:
                    for itype in range(len(self.sel_type)): # Atomic polar mode, should specify the atoms
                        index_lis = [index for index, w in enumerate(atom_has_polar) \
                                        if atom_has_polar[index] == self.sel_type[itype]]   # select index in this type

                        sys_matrix.append(np.zeros((1,len(self.sel_type))))
                        sys_matrix[-1][0,itype] = len(index_lis)

                        polar_bias.append(np.sum(
                            all_stat['atomic_polarizability'][ss].reshape((-1,9))[index_lis],axis=0).reshape((1,9)))
                else:   # No atomic polar in this system, so it should have global polar
                    if not all_stat['find_polarizability'][ss] > 0.0: # This system is jsut a joke?
                        continue
                    # Till here, we have global polar
                    sys_matrix.append(np.zeros((1,len(self.sel_type)))) # add a line in the equations
                    for itype in range(len(self.sel_type)): # Atomic polar mode, should specify the atoms
                        index_lis = [index for index, w in enumerate(atom_has_polar) \
                                        if atom_has_polar[index] == self.sel_type[itype]]   # select index in this type

                        sys_matrix[-1][0,itype] = len(index_lis)
                    
                    # add polar_bias
                    polar_bias.append(all_stat['polarizability'][ss].reshape((1,9)))

            matrix, bias = np.concatenate(sys_matrix,axis=0), np.concatenate(polar_bias,axis=0)
            atom_polar,_,_,_ \
                = np.linalg.lstsq(matrix, bias, rcond = 1e-3)
            for itype in range(len(self.sel_type)):
                self.constant_matrix[itype] = np.mean(np.diagonal(atom_polar[itype].reshape((3,3))))

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
                    layer+= one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, uniform_seed = self.uniform_seed)
                else :
                    layer = one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, uniform_seed = self.uniform_seed)
                if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
            if self.fit_diag :
                bavg = np.zeros(self.dim_rot_mat_1)
                # bavg[0] = self.avgeig[0]
                # bavg[1] = self.avgeig[1]
                # bavg[2] = self.avgeig[2]
                # (nframes x natoms) x naxis
                final_layer = one_layer(layer, self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, bavg = bavg, precision = self.fitting_precision, uniform_seed = self.uniform_seed)
                if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
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
                final_layer = one_layer(layer, self.dim_rot_mat_1*self.dim_rot_mat_1, activation_fn = None, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, bavg = bavg, precision = self.fitting_precision, uniform_seed = self.uniform_seed)
                if (not self.uniform_seed) and (self.seed is not None): self.seed += self.seed_shift
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
            final_layer = final_layer + self.constant_matrix[sel_type_idx] * tf.eye(3, batch_shape=[tf.shape(inputs)[0], natoms[2+type_i]], dtype = GLOBAL_TF_FLOAT_PRECISION)

            # concat the results
            if count == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)
            count += 1
        
        tf.summary.histogram('fitting_net_output', outs)
        return tf.cast(tf.reshape(outs, [-1]), GLOBAL_TF_FLOAT_PRECISION)


class GlobalPolarFittingSeA () :
    """
    Fit the system polarizability with descriptor se_a
    """
    @docstring_parameter(list_to_doc(ACTIVATION_FN_DICT.keys()), list_to_doc(PRECISION_DICT.keys()))
    def __init__ (self, 
                  descrpt : tf.Tensor,
                  neuron : List[int] = [120,120,120],
                  resnet_dt : bool = True,
                  sel_type : List[int] = None,
                  fit_diag : bool = True,
                  scale : List[float] = None,
                  diag_shift : List[float] = None,
                  seed : int = None,
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

