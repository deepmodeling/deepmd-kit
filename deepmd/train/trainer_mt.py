#!/usr/bin/env python3
import logging
import os
import time
import shutil
import google.protobuf.message
import numpy as np
from deepmd.env import tf
from deepmd.env import default_tf_session_config
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION
from deepmd.fit import EnerFitting, WFCFitting, PolarFittingLocFrame, PolarFittingSeA, GlobalPolarFittingSeA, DipoleFittingSeA
from deepmd.descriptor import DescrptLocFrame
from deepmd.descriptor import DescrptSeA
from deepmd.descriptor import DescrptSeT
from deepmd.descriptor import DescrptSeAEbd
from deepmd.descriptor import DescrptSeAEf
from deepmd.descriptor import DescrptSeR
from deepmd.descriptor import DescrptSeAR
from deepmd.descriptor import DescrptHybrid
from deepmd.model import EnerModel, WFCModel, DipoleModel, PolarModel, GlobalPolarModel
from deepmd.loss import EnerStdLoss, EnerDipoleLoss, TensorLoss
from deepmd.utils.errors import GraphTooLargeError
from deepmd.utils.learning_rate import LearningRateExp
from deepmd.utils.neighbor_stat import NeighborStat
from deepmd.utils.sess import run_sess
from deepmd.utils.type_embed import TypeEmbedNet

from tensorflow.python.client import timeline
from deepmd.env import op_module

# load grad of force module
import deepmd.op

from deepmd.common import j_must_have, ClassArg
from .trainer import _is_subdir, _generate_descrpt_from_param_dict

log = logging.getLogger(__name__)


class DPTrainer_mt (object):
    def __init__(self, 
                 jdata, 
                 run_opt):
        self.run_opt = run_opt
        self._init_param(jdata)

    def _init_param(self, jdata):
        # model config        
        model_param = j_must_have(jdata, 'model')
        training_param = j_must_have(jdata, 'training')
        descrpt_param = j_must_have(model_param, 'descriptor')
        fitting_param = j_must_have(model_param, 'fitting_net')
        typeebd_param = model_param.get('type_embedding', None)
        model_build_param = j_must_have(training_param, 'tasks')
        self.model_param    = model_param
        self.descrpt_param  = descrpt_param
        self.fitting_param = fitting_param
        self.model_build_param = model_build_param
        self.seed = training_param['seed']
        
        # descriptor
        self.descriptor_dict = {}
        for sub_descrpt in self.descrpt_param:
            try:
                descrpt_type = sub_descrpt['type']
            except KeyError:
                raise KeyError('the type of descriptor should be set by `type`')

            if descrpt_type != 'hybrid':
                tmp_descrpt = _generate_descrpt_from_param_dict(sub_descrpt)
            else :
                descrpt_list = []
                for ii in sub_descrpt.get('list', []):
                    descrpt_list.append(_generate_descrpt_from_param_dict(ii))
                tmp_descrpt = DescrptHybrid(descrpt_list)
            self.descriptor_dict[str(sub_descrpt['name'])] = tmp_descrpt

        # type embedding (share the same one)
        if typeebd_param is not None:
            self.typeebd = TypeEmbedNet(
                neuron=typeebd_param['neuron'],
                resnet_dt=typeebd_param['resnet_dt'],
                activation_function=typeebd_param['activation_function'],
                precision=typeebd_param['precision'],
                trainable=typeebd_param['trainable'],
                seed=typeebd_param['seed']
            )
        else:
            self.typeebd = None

        lr_param = j_must_have(jdata, 'learning_rate')
        lr_param_dict = {}
        for sub_lr in lr_param:
            lr_param_dict[sub_lr['name']] = sub_lr
        loss_param = jdata['loss']
        loss_param_dict = {}
        for sub_loss in loss_param:
            loss_param_dict[sub_loss['name']] = sub_loss
        
        self.fitting_descrpt = {}
        for sub_model in self.model_build_param:
            self.fitting_descrpt[sub_model['fitting_net']] = sub_model['descriptor']
        # fitting net
        self.fitting_dict = {}
        self.loss_dict = {}
        self.lr_dict = {}
        self.model_dict = {}
        self.fitting_type_dict = {}
        for sub_net in fitting_param:
            name_fitting = sub_net['name']
            try: 
                sub_net_type = sub_net['type']
            except:
                sub_net_type = 'ener'
            self.fitting_type_dict[name_fitting] = sub_net_type
            sub_net.pop('type', None)
            name_descrpt = self.fitting_descrpt[name_fitting]
            sub_net['descrpt'] = self.descriptor_dict[str(name_descrpt)]
            if sub_net_type == 'ener':
                self.fitting_dict[name_fitting] = EnerFitting(
                    descrpt = sub_net['descrpt'],
                    #type_map = model_param.get('type_map'), # this is the total type map
                    neuron = sub_net["neuron"],
                    resnet_dt = sub_net["resnet_dt"],
                    seed = sub_net["seed"],
                    name = name_fitting
                )
                
            elif sub_net_type == 'dipole':
                if descrpt_type == 'se_e2_a':
                    self.fitting_dict[name_fitting] = DipoleFittingSeA(**sub_net)
                else :
                    raise RuntimeError('fitting dipole only supports descrptors: se_e2_a')
            elif sub_net_type == 'polar':
                if descrpt_type == 'se_e2_a':
                    self.fitting_dict[name_fitting] = PolarFittingSeA(**sub_net)
                else :
                    raise RuntimeError('fitting polar only supports descrptors: loc_frame and se_e2_a')
            elif sub_net_type == 'global_polar':
                if descrpt_type == 'se_e2_a':
                    self.fitting_dict[name_fitting] = GlobalPolarFittingSeA(**sub_net)
                else :
                    raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_e2_a')
            else :
                raise RuntimeError('unknow fitting type ' + sub_net_type)

            # init model
            # infer model type by fitting_type
        self.model_component = {}
        self.method_name_list = []
        for sub_model in self.model_build_param:
            fitting_name = sub_model['fitting_net']
            descrpt_name = sub_model['descriptor']
            lr_name = sub_model['learning_rate']
            loss_name = sub_model['loss']
            model_name = sub_model['name']
            self.method_name_list.append(model_name)
            sub_model_component = {}
            sub_model_component['fitting'] = fitting_name
            sub_model_component['descrpt'] = descrpt_name
            self.model_component[model_name] = sub_model_component
            self.model_dict[model_name] = EnerModel(
                self.descriptor_dict[descrpt_name], 
                self.fitting_dict[fitting_name], 
                self.typeebd,
                #sub_net.get('type_map'), #this is the local type map
                model_param.get('type_map'), # this is the total type map
                model_param.get('data_stat_nbatch', 10),
                model_param.get('data_stat_protect', 1e-2),
                model_param.get('use_srtab'),
                model_param.get('smin_alpha'),
                model_param.get('sw_rmin'),
                model_param.get('sw_rmax'),
                model_name
            )

            # learning rate
            sub_lr = lr_param_dict[lr_name]
            try: 
                sub_lr_type = sub_lr['type']
            except:
                sub_lr_type = 'exp'
            if sub_lr_type == 'exp':
                self.lr_dict[lr_name] = LearningRateExp(sub_lr['start_lr'],
                                      sub_lr['stop_lr'],
                                      sub_lr['decay_steps'],
                                      name = lr_name)
            else :
                raise RuntimeError('unknown learning_rate type ' + sub_lr_type)        

            # loss
            # infer loss type by fitting_type
            sub_loss = loss_param_dict[loss_name]
            try :
                loss_type = sub_loss.get('type', 'ener')
            except:
                loss_type = 'ener'

            sub_net_type = self.fitting_type_dict[fitting_name]
            if sub_net_type == 'ener':
                sub_loss.pop('type', None)
                sub_loss['starter_learning_rate'] = self.lr_dict[lr_name].start_lr()
                if loss_type == 'ener':
                    self.loss_dict[loss_name] = EnerStdLoss(**sub_loss)
                elif loss_type == 'ener_dipole':
                    self.loss_dict[loss_name] = EnerDipoleLoss(**sub_loss)
                else:
                    raise RuntimeError('unknow loss type')
            elif sub_net_type == 'wfc':
                self.loss_dict[loss_name] = TensorLoss(sub_loss, 
                                   model = self.model_dict[model_name], 
                                   tensor_name = 'wfc',
                                   tensor_size = self.model_dict[model_name].get_out_size(),
                                   label_name = 'wfc')
            elif sub_net_type == 'dipole':
                self.loss_dict[loss_name] = TensorLoss(sub_loss, 
                                   model = self.model_dict[model_name], 
                                   tensor_name = 'dipole',
                                   tensor_size = 3,
                                   label_name = 'dipole')
            elif sub_net_type == 'polar':
                self.loss_dict[loss_name] = TensorLoss(sub_loss, 
                                   model = self.model_dict[model_name], 
                                   tensor_name = 'polar',
                                   tensor_size = 9,
                                   label_name = 'polarizability')
            elif sub_net_type == 'global_polar':
                self.loss_dict[loss_name] = TensorLoss(sub_loss, 
                                   model = self.model_dict[model_name], 
                                   tensor_name = 'global_polar',
                                   tensor_size = 9,
                                   atomic = False,
                                   label_name = 'polarizability')
            else :
                raise RuntimeError('get unknown fitting type when building loss function')

        
        self.l2_l_dict = {}
        self.l2_more_dict = {}
        for sub_loss in self.loss_dict.keys():
            self.l2_l_dict[sub_loss] = None
            self.l2_more_dict[sub_loss] = None

        # training
        
        tr_data = jdata['training']
        self.disp_file = tr_data.get('disp_file', 'lcurve.out')
        self.disp_freq = tr_data.get('disp_freq', 1000)
        self.save_freq = tr_data.get('save_freq', 1000)
        self.save_ckpt = tr_data.get('save_ckpt', 'model.ckpt')
        self.display_in_training = tr_data.get('disp_training', True)
        self.timing_in_training  = tr_data.get('time_training', True)
        self.profiling = tr_data.get('profiling', False)
        self.profiling_file = tr_data.get('profiling_file', 'timeline.json')
        self.tensorboard = tr_data.get('tensorboard', False)
        self.tensorboard_log_dir = tr_data.get('tensorboard_log_dir', 'log')
        # self.sys_probs = tr_data['sys_probs']
        # self.auto_prob_style = tr_data['auto_prob']
        self.useBN = False
        for fitting_name in self.fitting_dict.keys():
            fitting_type = self.fitting_type_dict[fitting_name]
            tmp_fitting = self.fitting_dict[fitting_name]
            if fitting_type == 'ener' and  tmp_fitting.get_numb_fparam() > 0 :
                self.numb_fparam = tmp_fitting.get_numb_fparam()
            else :
                self.numb_fparam = 0
            break

        if tr_data.get("validation_data", None) is not None:
            self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
        else:
            self.valid_numb_batch = 1


    def build (self, 
               data, 
               stop_batch = 0) :
        # datadocker
        #self.ntypes = len(self.model_param.get('type_map'))
        self.ntypes = 0
        for descrpt_name in self.descriptor_dict:
            sub_descrpt = self.descriptor_dict[descrpt_name]
            self.ntypes+=sub_descrpt.get_ntypes()

        
        
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

        self.type_map = data.get_type_map() # this is the total type_map from the datadocker
        
        for i in range(len(self.model_dict.keys())):
            model_name = list(self.model_dict.keys())[i]
            sub_model = self.model_dict[model_name]
            sub_data = data.get_data_system_idx(i)
            sub_model.data_stat(sub_data)

        if 'compress' in self.model_param and self.model_param['compress']['compress']:
            assert 'rcut' in self.descrpt_param,      "Error: descriptor must have attr rcut!"
            self.neighbor_stat \
                = NeighborStat(self.ntypes, self.descrpt_param['rcut'])
            self.min_nbor_dist, self.max_nbor_size \
                = self.neighbor_stat.get_stat(data)
            self.descrpt.enable_compression(self.min_nbor_dist, self.model_param['compress']['model_file'], self.model_param['compress']['table_config'][0], self.model_param['compress']['table_config'][1], self.model_param['compress']['table_config'][2], self.model_param['compress']['table_config'][3])

        


        self._build_lr()
        self._build_network(data)
        self._build_training(data)


    def _build_lr(self):
        self._extra_train_ops   = []
        self.learning_rate_dict = {}
        self.global_step = tf.train.get_or_create_global_step()
        for lr_name in self.lr_dict.keys():
            self.learning_rate_dict[lr_name] = self.lr_dict[lr_name].build(self.global_step, self.stop_batch)
        log.info("built lr")

    def _build_network(self, data):        
        self.place_holders = {}
        data_dict, data_name = data.get_data_dict() 
        for kk in data_dict.keys():
            if kk == 'type':
                continue
            prec = GLOBAL_TF_FLOAT_PRECISION
            if data_dict[kk]['high_prec'] :
                prec = GLOBAL_ENER_FLOAT_PRECISION
            self.place_holders[kk] = tf.placeholder(prec, [None], name = 't_' + kk)
            self.place_holders['find_'+kk] = tf.placeholder(tf.float32, name = 't_find_' + kk)

        self.place_holders['type']      = tf.placeholder(tf.int32,   [None], name='t_type')
        self.place_holders['natoms_vec']        = tf.placeholder(tf.int32,   [data.get_ntypes()+2], name='t_natoms')
        self.place_holders['default_mesh']      = tf.placeholder(tf.int32,   [None], name='t_mesh')
        self.place_holders['is_training']       = tf.placeholder(tf.bool)
        
        for model_name in self.model_dict.keys():
            sub_model = self.model_dict[model_name]
            suffix_dict = self.model_component[model_name]
            suffix_dict['type_embed'] = 'type_embedding'
            tmp_model_pred\
                = sub_model.build (self.place_holders['coord'], 
                                self.place_holders['type'], 
                                self.place_holders['natoms_vec'], 
                                self.place_holders['box'], 
                                self.place_holders['default_mesh'],
                                self.place_holders,
                                suffix = suffix_dict, 
                                reuse = tf.AUTO_REUSE)

        
            sub_loss = self.loss_dict[model_name] # model name should be the same as fitting and loss 
            #if sub_loss.get_name() == data_name:
            tmp_l2_l, tmp_l2_more\
                    = sub_loss.build (self.learning_rate_dict[model_name],
                               self.place_holders['natoms_vec'], 
                               tmp_model_pred,
                               self.place_holders,
                               suffix = model_name)
            self.l2_l_dict[model_name] = tmp_l2_l
            self.l2_more_dict[model_name] = tmp_l2_more

        log.info("built network")

    def _build_training(self,data):
        data_dict,data_name = data.get_data_dict() 
        trainable_variables = tf.trainable_variables()
        self.optimizer_dict = {}
        for loss_name in self.loss_dict.keys():
            sub_loss = self.loss_dict[loss_name]
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_dict[loss_name])   
            self.optimizer_dict[loss_name] = optimizer  
        grads = tf.gradients(self.l2_l_dict[data_name], trainable_variables)
        apply_op = self.optimizer_dict[data_name].apply_gradients (zip (grads, trainable_variables),
                                              global_step=self.global_step,
                                              name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        log.info("built training")

    def _init_sess_serial(self) :
        self.sess = tf.Session(config=default_tf_session_config)
        self.saver = tf.train.Saver()
        saver = self.saver
        if self.run_opt.init_mode == 'init_from_scratch' :
            log.info("initialize model from scratch")
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            fp = open(self.disp_file, "w")
            fp.close ()
        elif self.run_opt.init_mode == 'init_from_model' :
            log.info("initialize from model %s" % self.run_opt.init_model)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            saver.restore (self.sess, self.run_opt.init_model)            
            self.sess.run(self.global_step.assign(0))
            fp = open(self.disp_file, "w")
            fp.close ()
        elif self.run_opt.init_mode == 'restart' :
            log.info("restart from model %s" % self.run_opt.restart)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            saver.restore (self.sess, self.run_opt.restart)
        else :
            raise RuntimeError ("unkown init mode")

    def train (self, train_data, valid_data=None) :

        # if valid_data is None:  # no validation set specified.
        #     valid_data = train_data  # using training set as validation set.

        stop_batch = self.stop_batch
        if self.run_opt.is_distrib :
            raise RuntimeError('distributed training for multi-task is not supported at the moment')
        else :
            self._init_sess_serial()

        # self.print_head()
        fp = None
        if self.run_opt.is_chief :
            fp = open(self.disp_file, "a")

        cur_batch = self.sess.run(self.global_step)
        is_first_step = True
        self.cur_batch = cur_batch
        for lr_name in self.lr_dict.keys():
            tmp_lr = self.lr_dict[lr_name]
            log.info("system %s, start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e" % 
                 (lr_name,
                  self.sess.run(self.learning_rate_dict[lr_name]),
                  tmp_lr.value(cur_batch), 
                  tmp_lr.decay_steps_,
                  tmp_lr.decay_rate_,
                  tmp_lr.value(stop_batch)) 
            )

        prf_options = None
        prf_run_metadata = None
        if self.profiling :
            prf_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            prf_run_metadata = tf.RunMetadata()

        # set tensorboard execution environment
        if self.tensorboard :
            summary_merged_op = tf.summary.merge_all()
            shutil.rmtree(self.tensorboard_log_dir)
            tb_train_writer = tf.summary.FileWriter(self.tensorboard_log_dir + '/train', self.sess.graph)
            tb_valid_writer = tf.summary.FileWriter(self.tensorboard_log_dir + '/test')
        else:
            tb_train_writer = None
            tb_valid_writer = None
        
        train_time = 0
        n_methods = train_data.get_nmethod()
        
        while cur_batch < stop_batch :

            # first round validation:
            pick_method = np.random.randint(0,n_methods)
            train_batch = train_data.get_batch(pick_method)
            if self.display_in_training and is_first_step:
                valid_batches = [valid_data.get_batch(pick_method) for ii in range(self.valid_numb_batch)] if valid_data is not None else None
                self.valid_on_the_fly(fp, [train_batch], valid_batches,print_header=True,method = pick_method)
                is_first_step = False

            if self.timing_in_training: tic = time.time()
            train_feed_dict = self.get_feed_dict(train_batch, is_training=True)
            # use tensorboard to visualize the training of deepmd-kit
            # it will takes some extra execution time to generate the tensorboard data
            if self.tensorboard :
                summary, _ = self.sess.run([summary_merged_op, self.train_op], feed_dict=train_feed_dict,
                                           options=prf_options, run_metadata=prf_run_metadata)
                tb_train_writer.add_summary(summary, cur_batch)
            else :
                self.sess.run([self.train_op], feed_dict=train_feed_dict,
                              options=prf_options, run_metadata=prf_run_metadata)
            if self.timing_in_training: toc = time.time()
            if self.timing_in_training: train_time += toc - tic
            cur_batch = self.sess.run(self.global_step)
            self.cur_batch = cur_batch

            # on-the-fly validation
            if self.display_in_training and (cur_batch % self.disp_freq == 0):
                if self.timing_in_training:
                    tic = time.time()
                valid_batches = [valid_data.get_batch(pick_method) for ii in range(self.valid_numb_batch)] if valid_data is not None else None
                self.valid_on_the_fly(fp, [train_batch], valid_batches,method=pick_method)
                if self.timing_in_training:
                    toc = time.time()
                    test_time = toc - tic
                    log.info("batch %7d method %s training time %.2f s, testing time %.2f s"
                                  % (cur_batch,pick_method, train_time, test_time))
                    train_time = 0
                    
                if self.save_freq > 0 and cur_batch % self.save_freq == 0 and self.run_opt.is_chief :
                    if self.saver is not None :
                        self.saver.save (self.sess, os.getcwd() + "/" + self.save_ckpt)
                        log.info("saved checkpoint %s" % self.save_ckpt)
        if self.run_opt.is_chief: 
            fp.close ()
        if self.profiling and self.run_opt.is_chief :
            fetched_timeline = timeline.Timeline(prf_run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self.profiling_file, 'w') as f:
                f.write(chrome_trace)

    def get_feed_dict(self, batch, is_training):
        feed_dict = {}
        for kk in batch.keys():
            if kk == 'find_type' or kk == 'type':
                continue
            if 'find_' in kk:
                feed_dict[self.place_holders[kk]] = batch[kk]
            else:
                feed_dict[self.place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ['type']:
            feed_dict[self.place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ['natoms_vec', 'default_mesh']:
            feed_dict[self.place_holders[ii]] = batch[ii]
        feed_dict[self.place_holders['is_training']] = is_training
        
        return feed_dict

    def get_global_step(self):
        return self.sess.run(self.global_step)

    # def print_head (self) :  # depreciated
    #     if self.run_opt.is_chief:
    #         fp = open(self.disp_file, "a")
    #         print_str = "# %5s" % 'batch'
    #         print_str += self.loss.print_header()
    #         print_str += '   %8s\n' % 'lr'
    #         fp.write(print_str)
    #         fp.close ()

    def valid_on_the_fly(self,
                         fp,
                         train_batches,
                         valid_batches,
                         print_header=False,
                         method = None):
        train_results = self.get_evaluation_results(train_batches,method)
        valid_results = self.get_evaluation_results(valid_batches,method)

        cur_batch = self.cur_batch
        current_lr = self.sess.run(self.learning_rate_dict[self.method_name_list[method]])
        if print_header:
            self.print_header(fp, train_results, valid_results)
        self.print_on_training(fp, train_results, valid_results, cur_batch, current_lr,method)

    @staticmethod
    def print_header(fp, train_results, valid_results):
        print_str = ''
        print_str += "# %5s" % 'step'
        print_str += "  %6s" % 'method'
        if valid_results is not None:
            prop_fmt =  '   %11s %11s'
            for k in train_results.keys():
                print_str += prop_fmt % (k + '_val', k + '_trn')
        else:
            prop_fmt = '   %11s'
            for k in train_results.keys():
                print_str += prop_fmt % (k + '_trn')
        print_str += '   %8s\n' % 'lr'
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_on_training(fp, train_results, valid_results, cur_batch, cur_lr,method):
        print_str = ''
        print_str += "%7d" % cur_batch
        print_str += "%8d" % method
        if valid_results is not None:
            prop_fmt = "   %11.2e %11.2e"
            for k in valid_results.keys():
                # assert k in train_results.keys()
                print_str += prop_fmt % (valid_results[k], train_results[k])
        else:
            prop_fmt = "   %11.2e"
            for k in train_results.keys():
                print_str += prop_fmt % (train_results[k])
        print_str += "   %8.1e\n" % cur_lr
        fp.write(print_str)
        fp.flush()

    def get_evaluation_results(self, batch_list,method):
        if batch_list is None: return None
        numb_batch = len(batch_list)

        sum_results = {}    # sum of losses on all atoms
        sum_natoms = 0
        for i in range(numb_batch):
            batch = batch_list[i]
            natoms = batch["natoms_vec"]
            feed_dict = self.get_feed_dict(batch, is_training=False)
            sub_loss = self.loss_dict[self.method_name_list[method]]
            results = sub_loss.eval(self.sess, feed_dict, natoms)

            for k, v in results.items():
                if k == "natoms":
                    sum_natoms += v
                else:
                    sum_results[k] = sum_results.get(k, 0.) + v * results["natoms"]
        avg_results = {k: v / sum_natoms for k, v in sum_results.items() if not k == "natoms"}
        return avg_results
