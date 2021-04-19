import warnings
import numpy as np
from typing import Tuple, List

from deepmd.env import paddle
from deepmd.common import ClassArg, add_data_requirement, get_activation_func, get_precision, ACTIVATION_FN_DICT, PRECISION_DICT, docstring_parameter
from deepmd.utils.argcheck import list_to_doc
from deepmd.utils.network import OneLayer
from deepmd.descriptor import DescrptLocFrame
from deepmd.descriptor import DescrptSeA

from deepmd.env import global_cvt_2_tf_float
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION, GLOBAL_PD_FLOAT_PRECISION


class EnerFitting(paddle.nn.Layer):
    def __init__ (self, 
                  descrpt,
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
        super(EnerFitting, self).__init__(name_scope="EnerFitting")
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
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
                self.atom_ener.append(paddle.to_tensor(ae, dtype=GLOBAL_PD_FLOAT_PRECISION))
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
        
        emenets = []
        for type_i in range(self.ntypes):
            layers = []
            for ii in range(0,len(self.n_neuron)):
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1]:
                    layers.append(OneLayer(self.n_neuron[ii-1], self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i), seed = self.seed, use_timestep = self.resnet_dt, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, trainable = self.trainable[ii]))
                else:
                    layers.append(OneLayer(self.dim_descrpt+self.numb_fparam+self.numb_aparam, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i), seed = self.seed, activation_fn = self.fitting_activation_fn, precision = self.fitting_precision, trainable = self.trainable[ii]))
            layers.append(OneLayer(self.n_neuron[-1], 1, name='final_layer_type_'+str(type_i), seed = self.seed, activation_fn = None, precision = self.fitting_precision, trainable = self.trainable[ii]))
            
            emenets.append(paddle.nn.LayerList(layers))
        self.ElementNets = paddle.nn.LayerList(emenets)

        self.t_dfparam = paddle.to_tensor(self.numb_fparam, dtype = "int32")
        self.t_daparam = paddle.to_tensor(self.numb_aparam, dtype = "int32")

        # stat fparam
        if self.numb_fparam > 0:
            self.t_fparam_avg = paddle.to_tensor(np.zeros([1, self.numb_fparam]),
                                                dtype = GLOBAL_PD_FLOAT_PRECISION)
            self.t_fparam_istd = paddle.to_tensor(np.ones([1, self.numb_fparam]),
                                                 dtype = GLOBAL_PD_FLOAT_PRECISION)

        # stat aparam
        if self.numb_aparam > 0:
            self.t_aparam_avg = paddle.to_tensor(np.zeros([1, self.numb_aparam]),
                                                dtype = GLOBAL_PD_FLOAT_PRECISION)
            self.t_aparam_istd = paddle.to_tensor(np.ones([1, self.numb_aparam]),
                                                 dtype = GLOBAL_PD_FLOAT_PRECISION)


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
        if self.bias_atom_e is not None:
            assert (len(self.bias_atom_e) == self.ntypes)
            for type_i in range(self.ntypes):
                type_bias_ae = self.bias_atom_e[type_i]
                paddle.seed(self.seed)
                normal_init_ = paddle.nn.initializer.Normal(mean=type_bias_ae, std=1.0)
                final_layer = self.ElementNets[type_i][-1]
                normal_init_(final_layer.bias)


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

            self.t_fparam_avg = paddle.to_tensor(self.fparam_avg,
                                                dtype = GLOBAL_PD_FLOAT_PRECISION)
            self.t_fparam_istd = paddle.to_tensor(self.fparam_inv_std,
                                                 dtype = GLOBAL_PD_FLOAT_PRECISION)

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

            self.t_aparam_avg = paddle.to_tensor(self.aparam_avg,
                                                dtype = GLOBAL_PD_FLOAT_PRECISION)
            self.t_aparam_istd = paddle.to_tensor(self.aparam_inv_std,
                                                 dtype = GLOBAL_PD_FLOAT_PRECISION)


    def _compute_std (self, sumv2, sumv, sumn) :
        return np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))


    def forward(self, inputs, natoms, input_dict, reuse=None, suffix=''):
        if self.numb_fparam > 0 and (self.fparam_avg is None or self.fparam_inv_std is None):
            raise RuntimeError('No data stat result. one should do data statisitic, before build')
        if self.numb_aparam > 0 and (self.aparam_avg is None or self.aparam_inv_std is None):
            raise RuntimeError('No data stat result. one should do data statisitic, before build')

        start_index = 0
        inputs = paddle.cast(paddle.reshape(inputs, [-1, self.dim_descrpt * natoms[0]]), self.fitting_precision)

        if self.numb_fparam > 0:
            fparam = input_dict['fparam']
            fparam = paddle.reshape(fparam, [-1, self.numb_fparam])
            fparam = (fparam - self.fparam_avg) * self.fparam_inv_std
        if self.numb_aparam > 0:
            aparam = input_dict['aparam']
            aparam = paddle.reshape(aparam, [-1, self.numb_aparam])
            aparam = (aparam - self.aparam_avg) * self.aparam_inv_std
            aparam = paddle.reshape(aparam, [-1, self.numb_aparam * natoms[0]])

        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = paddle.slice(inputs, [1],
                                    [start_index * self.dim_descrpt],
                                    [(start_index + natoms[2 + type_i]) * self.dim_descrpt])
            inputs_i = paddle.reshape(inputs_i, [-1, self.dim_descrpt])
            layer = inputs_i
            if self.numb_fparam > 0:
                ext_fparam = paddle.tile(fparam, [1, natoms[2 + type_i]])
                ext_fparam = paddle.reshape(ext_fparam, [-1, self.numb_fparam])
                ext_fparam = paddle.cast(ext_fparam, self.fitting_precision)
                layer = paddle.concat([layer, ext_fparam], axis=1)
            if self.numb_aparam > 0:
                ext_aparam = paddle.slice(aparam, [1]
                                          [start_index * self.numb_aparam],
                                          [(start_index + natoms[2 + type_i]) * self.numb_aparam])
                ext_aparam = paddle.reshape(ext_aparam, [-1, self.numb_aparam])
                ext_aparam = paddle.cast(ext_aparam, self.fitting_precision)
                layer = paddle.concat([layer, ext_aparam], axis=1)
            start_index += natoms[2 + type_i]


            for ii in range(0, len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer += self.ElementNets[type_i][ii](layer)
                else :
                    layer = self.ElementNets[type_i][ii](layer)
            final_layer = self.ElementNets[type_i][len(self.n_neuron)](layer)

            # if type_i < len(self.atom_ener) and self.atom_ener[type_i] is not None: (Not implement)

            final_layer = paddle.reshape(final_layer, [inputs.shape[0], natoms[2 + type_i]])

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = paddle.concat([outs, final_layer], axis=1)

        return paddle.cast(paddle.reshape(outs, [-1]), GLOBAL_PD_FLOAT_PRECISION)
