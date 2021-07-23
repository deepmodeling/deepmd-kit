#!/usr/bin/env python3
import logging
import os
import time
import shutil
import numpy as np
from deepmd.env import tf, paddle
from deepmd.env import default_tf_session_config
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, GLOBAL_PD_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.fit import EnerFitting, WFCFitting, PolarFittingLocFrame, PolarFittingSeA, GlobalPolarFittingSeA, DipoleFittingSeA
from deepmd.descriptor import DescrptLocFrame
from deepmd.descriptor import DescrptSeA
from deepmd.descriptor import DescrptSeAT
from deepmd.descriptor import DescrptSeAEbd
from deepmd.descriptor import DescrptSeAEf
from deepmd.descriptor import DescrptSeR
from deepmd.descriptor import DescrptSeAR
from deepmd.descriptor import DescrptHybrid
from deepmd.model import EnerModel, WFCModel, DipoleModel, PolarModel, GlobalPolarModel
from deepmd.loss import EnerStdLoss, EnerDipoleLoss, TensorLoss
from deepmd.utils.learning_rate import LearningRateExp
from deepmd.utils.neighbor_stat import NeighborStat

from tensorflow.python.client import timeline
from deepmd.env import op_module

from collections import defaultdict
import sys


# load grad of force module
import deepmd.op

from deepmd.common import j_must_have, ClassArg

log = logging.getLogger(__name__)


def _is_subdir(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    if path == directory:
        return False
    relative = os.path.relpath(path, directory) + os.sep
    return not relative.startswith(os.pardir + os.sep)

def _generate_descrpt_from_param_dict(descrpt_param):
    try:
        descrpt_type = descrpt_param['type']
    except KeyError:
        raise KeyError('the type of descriptor should be set by `type`')
    descrpt_param.pop('type', None)
    to_pop = []
    for kk in descrpt_param:
        if kk[0] == '_':
            to_pop.append(kk)
    for kk in to_pop:
        descrpt_param.pop(kk, None)
    if descrpt_type == 'loc_frame':
        descrpt = DescrptLocFrame(**descrpt_param)
    elif descrpt_type == 'se_a' :            
        descrpt = DescrptSeA(**descrpt_param)
    elif descrpt_type == 'se_a_3be' or descrpt_type == 'se_at' :
        descrpt = DescrptSeAT(**descrpt_param)
    elif descrpt_type == 'se_a_tpe' or descrpt_type == 'se_a_ebd' :
        descrpt = DescrptSeAEbd(**descrpt_param)
    elif descrpt_type == 'se_a_ef' :
        descrpt = DescrptSeAEf(**descrpt_param)
    elif descrpt_type == 'se_r' :
        descrpt = DescrptSeR(**descrpt_param)
    elif descrpt_type == 'se_ar' :
        descrpt = DescrptSeAR(descrpt_param)
    else :
        raise RuntimeError('unknow model type ' + descrpt_type)
    return descrpt

class DPTrainer (object):
    def __init__(self, 
                 jdata, 
                 run_opt):
        self.run_opt = run_opt
        self._init_param(jdata)

    def _init_param(self, jdata):
        # model config        
        model_param = j_must_have(jdata, 'model')
        descrpt_param = j_must_have(model_param, 'descriptor')
        fitting_param = j_must_have(model_param, 'fitting_net')
        self.model_param    = model_param
        self.descrpt_param  = descrpt_param

        # descriptor
        try:
            descrpt_type = descrpt_param['type']
        except KeyError:
            raise KeyError('the type of descriptor should be set by `type`')

        if descrpt_type != 'hybrid':
            self.descrpt = _generate_descrpt_from_param_dict(descrpt_param)
        else :
            descrpt_list = []
            for ii in descrpt_param.get('list', []):
                descrpt_list.append(_generate_descrpt_from_param_dict(ii))
            self.descrpt = DescrptHybrid(descrpt_list)
        
        # fitting net
        try: 
            fitting_type = fitting_param['type']
        except:
            fitting_type = 'ener'
        
        self.fitting_type = fitting_type
        self.descrpt_type = descrpt_type

        fitting_param.pop('type', None)
        fitting_param['descrpt'] = self.descrpt
        if fitting_type == 'ener':
            self.fitting = EnerFitting(**fitting_param)
        # elif fitting_type == 'wfc':            
        #     self.fitting = WFCFitting(fitting_param, self.descrpt)
        elif fitting_type == 'dipole':
            if descrpt_type == 'se_a':
                self.fitting = DipoleFittingSeA(**fitting_param)
            else :
                raise RuntimeError('fitting dipole only supports descrptors: se_a')
        elif fitting_type == 'polar':
            # if descrpt_type == 'loc_frame':
            #     self.fitting = PolarFittingLocFrame(fitting_param, self.descrpt)
            if descrpt_type == 'se_a':
                self.fitting = PolarFittingSeA(**fitting_param)
            else :
                raise RuntimeError('fitting polar only supports descrptors: loc_frame and se_a')
        elif fitting_type == 'global_polar':
            if descrpt_type == 'se_a':
                self.fitting = GlobalPolarFittingSeA(**fitting_param)
            else :
                raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_a')
        else :
            raise RuntimeError('unknow fitting type ' + fitting_type)

        # init model
        # infer model type by fitting_type
        if fitting_type == 'ener':
            self.model = EnerModel(
                self.descrpt, 
                self.fitting, 
                model_param.get('type_map'),
                model_param.get('data_stat_nbatch', 10),
                model_param.get('data_stat_protect', 1e-2),
                model_param.get('use_srtab'),
                model_param.get('smin_alpha'),
                model_param.get('sw_rmin'),
                model_param.get('sw_rmax')
            )
        # elif fitting_type == 'wfc':
        #     self.model = WFCModel(model_param, self.descrpt, self.fitting)
        elif fitting_type == 'dipole':
            self.model = DipoleModel(
                self.descrpt, 
                self.fitting, 
                model_param.get('type_map'),
                model_param.get('data_stat_nbatch', 10),
                model_param.get('data_stat_protect', 1e-2)
            )
        elif fitting_type == 'polar':
            self.model = PolarModel(
                self.descrpt, 
                self.fitting,
                model_param.get('type_map'),
                model_param.get('data_stat_nbatch', 10),
                model_param.get('data_stat_protect', 1e-2)
            )
        elif fitting_type == 'global_polar':
            self.model = GlobalPolarModel(
                self.descrpt, 
                self.fitting,
                model_param.get('type_map'),
                model_param.get('data_stat_nbatch', 10),
                model_param.get('data_stat_protect', 1e-2)
            )
        else :
            raise RuntimeError('get unknown fitting type when building model')

        # learning rate
        lr_param = j_must_have(jdata, 'learning_rate')
        try: 
            lr_type = lr_param['type']
        except:
            lr_type = 'exp'
        if lr_type == 'exp':
            self.lr = LearningRateExp(lr_param['start_lr'],
                                      lr_param['stop_lr'],
                                      lr_param['decay_steps'])
        else :
            raise RuntimeError('unknown learning_rate type ' + lr_type)        

        # loss
        # infer loss type by fitting_type
        try :
            loss_param = jdata['loss']
            loss_type = loss_param.get('type', 'ener')
        except:
            loss_param = None
            loss_type = 'ener'

        if fitting_type == 'ener':
            loss_param.pop('type', None)
            loss_param['starter_learning_rate'] = self.lr.start_lr()
            if loss_type == 'ener':
                self.loss = EnerStdLoss(**loss_param)
            elif loss_type == 'ener_dipole':
                self.loss = EnerDipoleLoss(**loss_param)
            else:
                raise RuntimeError('unknow loss type')
        elif fitting_type == 'wfc':
            self.loss = TensorLoss(loss_param, 
                                   model = self.model, 
                                   tensor_name = 'wfc',
                                   tensor_size = self.model.get_out_size(),
                                   label_name = 'wfc')
        elif fitting_type == 'dipole':
            self.loss = TensorLoss(loss_param, 
                                   model = self.model, 
                                   tensor_name = 'dipole',
                                   tensor_size = 3,
                                   label_name = 'dipole')
        elif fitting_type == 'polar':
            self.loss = TensorLoss(loss_param, 
                                   model = self.model, 
                                   tensor_name = 'polar',
                                   tensor_size = 9,
                                   label_name = 'polarizability')
        elif fitting_type == 'global_polar':
            self.loss = TensorLoss(loss_param, 
                                   model = self.model, 
                                   tensor_name = 'global_polar',
                                   tensor_size = 9,
                                   atomic = False,
                                   label_name = 'polarizability')
        else :
            raise RuntimeError('get unknown fitting type when building loss function')

        # training
        training_param = j_must_have(jdata, 'training')

        # ! first .add() altered by Marián Rynik
        tr_args = ClassArg()\
                  .add('numb_test',     [int, list, str],    default = 1)\
                  .add('disp_file',     str,    default = 'lcurve.out')\
                  .add('disp_freq',     int,    default = 1)\
                  .add('save_freq',     int,    default = 1000)\
                  .add('save_ckpt',     str,    default = 'model.ckpt')\
                  .add('display_in_training', bool, default = True)\
                  .add('timing_in_training',  bool, default = True)\
                  .add('profiling',     bool,   default = False)\
                  .add('profiling_file',str,    default = 'timeline.json')\
                  .add('tensorboard',     bool,   default = False)\
                  .add('tensorboard_log_dir',str,    default = 'log')\
                  .add('sys_probs',   list    )\
                  .add('auto_prob_style', str, default = "prob_sys_size")
        tr_data = tr_args.parse(training_param)
        # not needed
        # self.numb_test = tr_data['numb_test']
        self.disp_file = tr_data['disp_file']
        self.disp_freq = tr_data['disp_freq']
        self.save_freq = tr_data['save_freq']
        self.save_ckpt = tr_data['save_ckpt']
        self.display_in_training = tr_data['display_in_training']
        self.timing_in_training  = tr_data['timing_in_training']
        self.profiling = tr_data['profiling']
        self.profiling_file = tr_data['profiling_file']
        self.tensorboard = tr_data['tensorboard']
        self.tensorboard_log_dir = tr_data['tensorboard_log_dir']
        self.sys_probs = tr_data['sys_probs']        
        self.auto_prob_style = tr_data['auto_prob_style']        
        self.useBN = False
        if fitting_type == 'ener' and  self.fitting.get_numb_fparam() > 0 :
            self.numb_fparam = self.fitting.get_numb_fparam()
        else :
            self.numb_fparam = 0


    def build (self, 
               data, 
               stop_batch = 0) :
        self.ntypes = self.model.get_ntypes()
        # Usually, the type number of the model should be equal to that of the data
        # However, nt_model > nt_data should be allowed, since users may only want to 
        # train using a dataset that only have some of elements 
        assert (self.ntypes >= data.get_ntypes()), "ntypes should match that found in data"
        self.stop_batch = stop_batch

        self.batch_size = data.get_batch_size()

        if self.numb_fparam > 0 :
            log.info("training with %d frame parameter(s)" % self.numb_fparam)
        else:
            log.info("training without frame parameter")

        self.type_map = data.get_type_map()

        self.model.data_stat(data)
        self.lr_scheduler = self.lr.build(self.stop_batch)


    def train (self, 
               data,
               stop_batch) :
        
        self.stop_batch = stop_batch

        self.print_head()
        fp = None
        if self.run_opt.is_chief :
            fp = open(self.disp_file, "a")

        is_first_step = True
        self.cur_batch = 0
        
        adam = paddle.optimizer.Adam(learning_rate = self.lr_scheduler, parameters=self.model.parameters())
        
        log.info("start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e" % 
                 (self.lr_scheduler.get_lr(),
                  self.lr.value(self.cur_batch), 
                  self.lr.decay_steps_,
                  self.lr.decay_rate_,
                  self.lr.value(stop_batch)) 
        )

        prf_options = None
        prf_run_metadata = None
        if self.profiling :
            pass

        tb_train_writer = None
        tb_test_writer = None
        
        train_time = 0

        data_dict = data.get_data_dict()
        while self.cur_batch < stop_batch :
            batch_data = data.get_batch (sys_probs = self.sys_probs,
                                         auto_prob_style = self.auto_prob_style
            )
            model_inputs = {}
            for kk in batch_data.keys():
                if kk == 'find_type' or kk == 'type' :
                    continue
                prec = GLOBAL_PD_FLOAT_PRECISION
                if 'find_' in kk :
                    model_inputs[kk] = paddle.to_tensor(batch_data[kk], dtype="float32")
                else:
                    model_inputs[kk] = paddle.to_tensor(np.reshape(batch_data[kk], [-1]), dtype=prec)
            for ii in ['type'] :
                model_inputs[ii] = paddle.to_tensor(np.reshape(batch_data[ii], [-1]), dtype="int32")
            for ii in ['natoms_vec', 'default_mesh'] :
                model_inputs[ii] = paddle.to_tensor(batch_data[ii], dtype="int32")
            model_inputs['is_training'] = paddle.to_tensor(True)
            
            if self.display_in_training and is_first_step :
                self.test_on_the_fly(fp, data, model_inputs, tb_test_writer)
                is_first_step = False
            if self.timing_in_training : tic = time.time()

            model_pred = self.model(model_inputs['coord'], model_inputs['type'], model_inputs['natoms_vec'], model_inputs['box'], model_inputs['default_mesh'], model_inputs, suffix = "", reuse = False)
            l2_l, l2_more = self.loss.calculate_loss(self.lr_scheduler.get_lr(), model_inputs['natoms_vec'], model_pred, model_inputs, suffix = "test")

            adam.clear_grad()
            l2_l.backward()
            adam.step()

            if self.timing_in_training : toc = time.time()
            if self.timing_in_training : train_time += toc - tic
            self.cur_batch += 1

            if (self.cur_batch % self.lr.decay_steps_) == 0:
                self.lr_scheduler.step()

            if self.display_in_training and (self.cur_batch % self.disp_freq == 0) :
                tic = time.time()
                self.test_on_the_fly(fp, data, model_inputs, tb_test_writer)
                toc = time.time()
                test_time = toc - tic
                if self.timing_in_training :
                    log.info("batch %7d training time %.2f s, testing time %.2f s"
                                  % (self.cur_batch, train_time, test_time))
                    train_time = 0
            
        if self.run_opt.is_chief: 
            fp.close ()
        if self.profiling and self.run_opt.is_chief :
            fetched_timeline = timeline.Timeline(prf_run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self.profiling_file, 'w') as f:
                f.write(chrome_trace)
        
        self.save_model(model_inputs, self.save_ckpt + "/model")

    def save_model(self, model_inputs_, folder_name_):
        # Since "paddle.jit.to_static" modifiess the model in-place
        # We have to make a temporary model copy to avoid damage to the original model.
        save_path = os.getcwd() + "/" + folder_name_
        if self.fitting_type == "ener" and self.descrpt_type == "se_a":
          input_names = ['coord', 'type', 'natoms_vec', 'box', 'default_mesh']
          input_specs = [paddle.static.InputSpec(model_inputs_[name].shape, model_inputs_[name].dtype, name=name) for name in input_names]
        else:
          raise NotImplementedError

        model = paddle.jit.to_static(self.model, input_spec=input_specs)
        paddle.jit.save(model, save_path)

        log.info("saved checkpoint to %s" % (save_path))

    def get_global_step (self) :
        return self.cur_batch

    def print_head (self) :
        if self.run_opt.is_chief:
            fp = open(self.disp_file, "a")
            print_str = "# %5s" % 'batch'
            print_str += self.loss.print_header()
            print_str += '   %8s\n' % 'lr'
            fp.write(print_str)
            fp.close ()

    def test_on_the_fly (self,
                         fp,
                         data,
                         model_train_inputs,
                         tb_writer) :
        # Do not need to pass numb_test here as data object already knows it.
        # Both DeepmdDataSystem and ClassArg parse the same json file
        model_test_inputs = {}
        test_data = data.get_test(n_test=data.get_sys_ntest())

        for kk in test_data.keys():
            if kk == 'find_type' or kk == 'type' :
                continue
            prec = GLOBAL_PD_FLOAT_PRECISION
            if 'find_' in kk:
                model_test_inputs[kk] = paddle.to_tensor(test_data[kk], dtype="float32")
            else:
                # again the data object knows appropriate test data shape,
                # there is no need to slice again!
                # feed_dict_test[self.place_holders[kk]] = np.reshape(test_data[kk][:self.numb_test[data.pick_idx]], [-1])
                model_test_inputs[kk] = paddle.to_tensor(np.reshape(test_data[kk], [-1]), dtype=prec)
        for ii in ['type'] :
            model_test_inputs[ii] = paddle.to_tensor(np.reshape(test_data[ii], [-1]), dtype="int32")   
        for ii in ['natoms_vec', 'default_mesh'] :
            model_test_inputs[ii] = paddle.to_tensor(test_data[ii], dtype="int32")

        model_test_inputs['is_training'] = paddle.to_tensor(False)

        current_batch = self.cur_batch
        current_lr = self.lr_scheduler.get_lr()
        if self.run_opt.is_chief:
            print_str = "%7d" % current_batch

            model_pred = self.model(model_train_inputs['coord'], model_train_inputs['type'], model_train_inputs['natoms_vec'], model_train_inputs['box'], model_train_inputs['default_mesh'], model_train_inputs, suffix = "", reuse = False)
            l2_l, l2_more = self.loss.calculate_loss(self.lr_scheduler.get_lr(), model_train_inputs['natoms_vec'], model_pred, model_train_inputs, suffix = "test")

            error_train = l2_l.numpy()
            error_e_train = l2_more['l2_ener_loss'].numpy()
            error_f_train = l2_more['l2_force_loss'].numpy()
            error_v_train = l2_more['l2_virial_loss'].numpy()
            error_ae_train = l2_more['l2_atom_ener_loss'].numpy()
            error_pf_train = l2_more['l2_pref_force_loss'].numpy()

            model_pred = self.model(model_test_inputs['coord'], model_test_inputs['type'], model_test_inputs['natoms_vec'], model_test_inputs['box'], model_test_inputs['default_mesh'], model_test_inputs, suffix = "", reuse = False)
            l2_l, l2_more = self.loss.calculate_loss(self.lr_scheduler.get_lr(), model_test_inputs['natoms_vec'], model_pred, model_test_inputs, suffix = "test")

            error_test = l2_l.numpy()
            error_e_test = l2_more['l2_ener_loss'].numpy()
            error_f_test = l2_more['l2_force_loss'].numpy()
            error_v_test = l2_more['l2_virial_loss'].numpy()
            error_ae_test = l2_more['l2_atom_ener_loss'].numpy()
            error_pf_test = l2_more['l2_pref_force_loss'].numpy()

            prop_fmt = "   %11.2e %11.2e"
            natoms = test_data['natoms_vec']
            print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))
            if self.loss.has_e :
                print_str += prop_fmt % (np.sqrt(error_e_test) / natoms[0], np.sqrt(error_e_train) / natoms[0])
            if self.loss.has_ae :
                print_str += prop_fmt % (np.sqrt(error_ae_test), np.sqrt(error_ae_train))
            if self.loss.has_f :
                print_str += prop_fmt % (np.sqrt(error_f_test), np.sqrt(error_f_train))
            if self.loss.has_v :
                print_str += prop_fmt % (np.sqrt(error_v_test) / natoms[0], np.sqrt(error_v_train) / natoms[0])
            if self.loss.has_pf:
                print_str += prop_fmt % (np.sqrt(error_pf_test), np.sqrt(error_pf_train))

            print("batch %7d, lr %f, l2_l %f, l2_ener_loss %f, l2_force_loss %f, l2_virial_loss %f, l2_atom_ener_loss %f, l2_pref_force_loss %f" % (current_batch, current_lr, error_train, error_e_train, error_f_train, error_v_train, error_ae_train, error_pf_train))
            print_str += "   %8.1e\n" % current_lr
            fp.write(print_str)
            fp.flush ()
