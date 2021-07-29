#!/usr/bin/env python3
import logging
import os
import time
import shutil
import google.protobuf.message
import numpy as np
from deepmd.env import tf
from deepmd.env import get_tf_session_config
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
    elif descrpt_type == 'se_e2_a' or descrpt_type == 'se_a' :
        descrpt = DescrptSeA(**descrpt_param)
    elif descrpt_type == 'se_e2_r' or descrpt_type == 'se_r' :
        descrpt = DescrptSeR(**descrpt_param)
    elif descrpt_type == 'se_e3' or descrpt_type == 'se_at' or descrpt_type == 'se_a_3be' :
        descrpt = DescrptSeT(**descrpt_param)
    elif descrpt_type == 'se_a_tpe' or descrpt_type == 'se_a_ebd' :
        descrpt = DescrptSeAEbd(**descrpt_param)
    elif descrpt_type == 'se_a_ef' :
        descrpt = DescrptSeAEf(**descrpt_param)
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
        typeebd_param = model_param.get('type_embedding', None)
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
        fitting_param.pop('type', None)
        fitting_param['descrpt'] = self.descrpt
        if fitting_type == 'ener':
            self.fitting = EnerFitting(**fitting_param)
        # elif fitting_type == 'wfc':            
        #     self.fitting = WFCFitting(fitting_param, self.descrpt)
        elif fitting_type == 'dipole':
            if descrpt_type == 'se_e2_a':
                self.fitting = DipoleFittingSeA(**fitting_param)
            else :
                raise RuntimeError('fitting dipole only supports descrptors: se_e2_a')
        elif fitting_type == 'polar':
            # if descrpt_type == 'loc_frame':
            #     self.fitting = PolarFittingLocFrame(fitting_param, self.descrpt)
            if descrpt_type == 'se_e2_a':
                self.fitting = PolarFittingSeA(**fitting_param)
            else :
                raise RuntimeError('fitting polar only supports descrptors: loc_frame and se_e2_a')
        elif fitting_type == 'global_polar':
            if descrpt_type == 'se_e2_a':
                self.fitting = GlobalPolarFittingSeA(**fitting_param)
            else :
                raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_e2_a')
        else :
            raise RuntimeError('unknow fitting type ' + fitting_type)

        # type embedding
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

        # init model
        # infer model type by fitting_type
        if fitting_type == 'ener':
            self.model = EnerModel(
                self.descrpt, 
                self.fitting, 
                self.typeebd,
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
        tr_data = jdata['training']
        self.disp_file = tr_data.get('disp_file', 'lcurve.out')
        self.disp_freq = tr_data.get('disp_freq', 1000)
        self.save_freq = tr_data.get('save_freq', 1000)
        self.save_ckpt = tr_data.get('save_ckpt', 'model.ckpt')
        self.display_in_training = tr_data.get('disp_training', True)
        self.timing_in_training  = tr_data.get('time_training', True)
        self.profiling = self.run_opt.is_chief and tr_data.get('profiling', False)
        self.profiling_file = tr_data.get('profiling_file', 'timeline.json')
        self.tensorboard = self.run_opt.is_chief and tr_data.get('tensorboard', False)
        self.tensorboard_log_dir = tr_data.get('tensorboard_log_dir', 'log')
        # self.sys_probs = tr_data['sys_probs']
        # self.auto_prob_style = tr_data['auto_prob']
        self.useBN = False
        if fitting_type == 'ener' and  self.fitting.get_numb_fparam() > 0 :
            self.numb_fparam = self.fitting.get_numb_fparam()
        else :
            self.numb_fparam = 0

        if tr_data.get("validation_data", None) is not None:
            self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
        else:
            self.valid_numb_batch = 1


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

        if 'compress' in self.model_param and self.model_param['compress']['compress']:
            assert 'rcut' in self.descrpt_param,      "Error: descriptor must have attr rcut!"
            self.neighbor_stat \
                = NeighborStat(self.ntypes, self.descrpt_param['rcut'])
            self.min_nbor_dist, self.max_nbor_size \
                = self.neighbor_stat.get_stat(data)
            self.descrpt.enable_compression(self.min_nbor_dist, self.model_param['compress']['model_file'], self.model_param['compress']['table_config'][0], self.model_param['compress']['table_config'][1], self.model_param['compress']['table_config'][2], self.model_param['compress']['table_config'][3])

        self._build_lr()
        self._build_network(data)
        self._build_training()


    def _build_lr(self):
        self._extra_train_ops   = []
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = self.lr.build(self.global_step, self.stop_batch)
        log.info("built lr")

    def _build_network(self, data):        
        self.place_holders = {}
        data_dict = data.get_data_dict()
        for kk in data_dict.keys():
            if kk == 'type':
                continue
            prec = GLOBAL_TF_FLOAT_PRECISION
            if data_dict[kk]['high_prec'] :
                prec = GLOBAL_ENER_FLOAT_PRECISION
            self.place_holders[kk] = tf.placeholder(prec, [None], name = 't_' + kk)
            self.place_holders['find_'+kk] = tf.placeholder(tf.float32, name = 't_find_' + kk)

        self.place_holders['type']      = tf.placeholder(tf.int32,   [None], name='t_type')
        self.place_holders['natoms_vec']        = tf.placeholder(tf.int32,   [self.ntypes+2], name='t_natoms')
        self.place_holders['default_mesh']      = tf.placeholder(tf.int32,   [None], name='t_mesh')
        self.place_holders['is_training']       = tf.placeholder(tf.bool)
        self.model_pred\
            = self.model.build (self.place_holders['coord'], 
                                self.place_holders['type'], 
                                self.place_holders['natoms_vec'], 
                                self.place_holders['box'], 
                                self.place_holders['default_mesh'],
                                self.place_holders,
                                suffix = "", 
                                reuse = False)

        self.l2_l, self.l2_more\
            = self.loss.build (self.learning_rate,
                               self.place_holders['natoms_vec'], 
                               self.model_pred,
                               self.place_holders,
                               suffix = "test")

        log.info("built network")

    def _build_training(self):
        trainable_variables = tf.trainable_variables()
        if self.run_opt.is_distrib:
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate*self.run_opt.world_size)
            optimizer = self.run_opt._HVD.DistributedOptimizer(optimizer)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        grads = tf.gradients(self.l2_l, trainable_variables)
        apply_op = optimizer.apply_gradients (zip (grads, trainable_variables),
                                              global_step=self.global_step,
                                              name='train_step')
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        log.info("built training")

    def _init_session(self):
        config = get_tf_session_config()
        device, idx = self.run_opt.my_device.split(":", 1)
        if device == "gpu":
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = idx
        self.sess = tf.Session(config=config)

        # Initializes or restore global variables
        init_op = tf.global_variables_initializer()
        if self.run_opt.is_chief:
            self.saver = tf.train.Saver()
            if self.run_opt.init_mode == 'init_from_scratch' :
                log.info("initialize model from scratch")
                run_sess(self.sess, init_op)
                fp = open(self.disp_file, "w")
                fp.close ()
            elif self.run_opt.init_mode == 'init_from_model' :
                log.info("initialize from model %s" % self.run_opt.init_model)
                run_sess(self.sess, init_op)
                self.saver.restore (self.sess, self.run_opt.init_model)            
                run_sess(self.sess, self.global_step.assign(0))
                fp = open(self.disp_file, "w")
                fp.close ()
            elif self.run_opt.init_mode == 'restart' :
                log.info("restart from model %s" % self.run_opt.restart)
                run_sess(self.sess, init_op)
                self.saver.restore (self.sess, self.run_opt.restart)
            else :
                raise RuntimeError ("unkown init mode")
        else:
            run_sess(self.sess, init_op)
            self.saver = None

        # Ensure variable consistency among tasks when training starts
        if self.run_opt.is_distrib:
            bcast_op = self.run_opt._HVD.broadcast_global_variables(0)
            if self.run_opt.is_chief:
                log.info('broadcast global variables to other tasks')
            else:
                log.info('receive global variables from task#0')
            run_sess(self.sess, bcast_op)

    def train (self, train_data, valid_data=None) :

        # if valid_data is None:  # no validation set specified.
        #     valid_data = train_data  # using training set as validation set.

        stop_batch = self.stop_batch
        self._init_session()

        # Before data shard is enabled, only cheif do evaluation and record it
        # self.print_head()
        fp = None
        if self.run_opt.is_chief :
            fp = open(self.disp_file, "a")

        cur_batch = run_sess(self.sess, self.global_step)
        is_first_step = True
        self.cur_batch = cur_batch
        log.info("start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e" % 
                 (run_sess(self.sess, self.learning_rate),
                  self.lr.value(cur_batch), 
                  self.lr.decay_steps_,
                  self.lr.decay_rate_,
                  self.lr.value(stop_batch)) 
        )

        prf_options = None
        prf_run_metadata = None
        if self.profiling:
            prf_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            prf_run_metadata = tf.RunMetadata()

        # set tensorboard execution environment
        if self.tensorboard:
            summary_merged_op = tf.summary.merge_all()
            # Remove TB old logging directory from previous run
            try:
                shutil.rmtree(self.tensorboard_log_dir)
            except FileNotFoundError:
                pass  # directory does not exist, this is OK
            except Exception as e:
                # general error when removing directory, warn user
                log.exception(
                    f"Could not remove old tensorboard logging directory: "
                    f"{self.tensorboard_log_dir}. Error: {e}"
                )
            else:
                log.debug("Removing old tensorboard log directory.")
            tb_train_writer = tf.summary.FileWriter(self.tensorboard_log_dir + '/train', self.sess.graph)
            tb_valid_writer = tf.summary.FileWriter(self.tensorboard_log_dir + '/test')
        else:
            tb_train_writer = None
            tb_valid_writer = None
        
        train_time = 0
        while cur_batch < stop_batch :

            # first round validation:
            train_batch = train_data.get_batch()
            if self.display_in_training and is_first_step:
                if self.run_opt.is_chief:
                    valid_batches = [valid_data.get_batch() for ii in range(self.valid_numb_batch)] if valid_data is not None else None
                    self.valid_on_the_fly(fp, [train_batch], valid_batches, print_header=True)
                is_first_step = False

            if self.timing_in_training: tic = time.time()
            train_feed_dict = self.get_feed_dict(train_batch, is_training=True)
            # use tensorboard to visualize the training of deepmd-kit
            # it will takes some extra execution time to generate the tensorboard data
            if self.tensorboard :
                summary, _ = run_sess(self.sess, [summary_merged_op, self.train_op], feed_dict=train_feed_dict,
                                           options=prf_options, run_metadata=prf_run_metadata)
                tb_train_writer.add_summary(summary, cur_batch)
            else :
                run_sess(self.sess, [self.train_op], feed_dict=train_feed_dict,
                              options=prf_options, run_metadata=prf_run_metadata)
            if self.timing_in_training: toc = time.time()
            if self.timing_in_training: train_time += toc - tic
            cur_batch = run_sess(self.sess, self.global_step)
            self.cur_batch = cur_batch

            # on-the-fly validation
            if self.display_in_training and (cur_batch % self.disp_freq == 0):
                if self.timing_in_training:
                    tic = time.time()
                if self.run_opt.is_chief:
                    valid_batches = [valid_data.get_batch() for ii in range(self.valid_numb_batch)] if valid_data is not None else None
                    self.valid_on_the_fly(fp, [train_batch], valid_batches)
                if self.timing_in_training:
                    toc = time.time()
                    test_time = toc - tic
                    log.info("batch %7d training time %.2f s, testing time %.2f s"
                                  % (cur_batch, train_time, test_time))
                    train_time = 0
                if self.save_freq > 0 and cur_batch % self.save_freq == 0 and self.saver is not None:
                    try:
                        self.saver.save (self.sess, os.getcwd() + "/" + self.save_ckpt)
                    except google.protobuf.message.DecodeError as e:
                        raise GraphTooLargeError(
                            "The graph size exceeds 2 GB, the hard limitation of protobuf."
                            " Then a DecodeError was raised by protobuf. You should "
                            "reduce the size of your model."
                        ) from e
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
        return run_sess(self.sess, self.global_step)

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
                         print_header=False):
        train_results = self.get_evaluation_results(train_batches)
        valid_results = self.get_evaluation_results(valid_batches)

        cur_batch = self.cur_batch
        current_lr = run_sess(self.sess, self.learning_rate)
        if print_header:
            self.print_header(fp, train_results, valid_results)
        self.print_on_training(fp, train_results, valid_results, cur_batch, current_lr)

    @staticmethod
    def print_header(fp, train_results, valid_results):
        print_str = ''
        print_str += "# %5s" % 'step'
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
    def print_on_training(fp, train_results, valid_results, cur_batch, cur_lr):
        print_str = ''
        print_str += "%7d" % cur_batch
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

    def get_evaluation_results(self, batch_list):
        if batch_list is None: return None
        numb_batch = len(batch_list)

        sum_results = {}    # sum of losses on all atoms
        sum_natoms = 0
        for i in range(numb_batch):
            batch = batch_list[i]
            natoms = batch["natoms_vec"]
            feed_dict = self.get_feed_dict(batch, is_training=False)
            results = self.loss.eval(self.sess, feed_dict, natoms)

            for k, v in results.items():
                if k == "natoms":
                    sum_natoms += v
                else:
                    sum_results[k] = sum_results.get(k, 0.) + v * results["natoms"]
        avg_results = {k: v / sum_natoms for k, v in sum_results.items() if not k == "natoms"}
        return avg_results
