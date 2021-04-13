#!/usr/bin/env python3
import logging
import os
import time
import shutil
import numpy as np
from deepmd.env import tf
from deepmd.env import default_tf_session_config
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
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
                  .add('disp_freq',     int,    default = 100)\
                  .add('save_freq',     int,    default = 1000)\
                  .add('save_ckpt',     str,    default = 'model.ckpt')\
                  .add('display_in_training', bool, default = True)\
                  .add('timing_in_training',  bool, default = True)\
                  .add('profiling',     bool,   default = False)\
                  .add('profiling_file',str,    default = 'timeline.json')\
                  .add('tensorboard',     bool,   default = False)\
                  .add('tensorboard_log_dir',str,    default = 'log')\
                  .add('sys_probs',   list    )\
                  .add('auto_prob', str, default = "prob_sys_size")
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
        self.auto_prob_style = tr_data['auto_prob']        
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

        if 'compress' in self.model_param and self.model_param['compress']['compress']:
            assert 'rcut' in self.descrpt_param,      "Error: descriptor must have attr rcut!"
            self.neighbor_stat \
                = NeighborStat(self.ntypes, self.descrpt_param['rcut'])
            self.min_nbor_dist, self.max_nbor_size \
                = self.neighbor_stat.get_stat(data)
            self.descrpt.enable_compression(self.min_nbor_dist, self.model_param['compress']['model_file'], self.model_param['compress']['table_config'][0], self.model_param['compress']['table_config'][1], self.model_param['compress']['table_config'][2], self.model_param['compress']['table_config'][3])

        worker_device = "/job:%s/task:%d/%s" % (self.run_opt.my_job_name,
                                                self.run_opt.my_task_index,
                                                self.run_opt.my_device)

        with tf.device(tf.train.replica_device_setter(worker_device = worker_device,
                                                      cluster = self.run_opt.cluster_spec)):
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
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        if self.run_opt.is_distrib :
            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer,
                replicas_to_aggregate = self.run_opt.cluster_spec.num_tasks("worker"),
                total_num_replicas = self.run_opt.cluster_spec.num_tasks("worker"),
                name = "sync_replicas")
            self.sync_replicas_hook = optimizer.make_session_run_hook(self.run_opt.is_chief)            
        grads = tf.gradients(self.l2_l, trainable_variables)
        apply_op = optimizer.apply_gradients (zip (grads, trainable_variables),
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

    def _init_sess_distrib(self):
        ckpt_dir = os.path.join(os.getcwd(), self.save_ckpt)
        assert(_is_subdir(ckpt_dir, os.getcwd())), "the checkpoint dir must be a subdir of the current dir"
        if self.run_opt.init_mode == 'init_from_scratch' :
            log.info("initialize model from scratch")
            if self.run_opt.is_chief :
                if os.path.exists(ckpt_dir):
                    shutil.rmtree(ckpt_dir)
                if not os.path.exists(ckpt_dir) :
                    os.makedirs(ckpt_dir)
                fp = open(self.disp_file, "w")
                fp.close ()
        elif self.run_opt.init_mode == 'init_from_model' :
            raise RuntimeError("distributed training does not support %s" % self.run_opt.init_mode)
        elif self.run_opt.init_mode == 'restart' :
            log.info("restart from model %s" % ckpt_dir)
            if self.run_opt.is_chief :
                assert(os.path.isdir(ckpt_dir)), "the checkpoint dir %s should exists" % ckpt_dir
        else :
            raise RuntimeError ("unkown init mode")

        saver = tf.train.Saver(max_to_keep = 1)
        self.saver = None
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # config = tf.ConfigProto(allow_soft_placement=True,
        #                         gpu_options = gpu_options,
        #                         intra_op_parallelism_threads=self.run_opt.num_intra_threads,
        #                         inter_op_parallelism_threads=self.run_opt.num_inter_threads)
        config = tf.ConfigProto(intra_op_parallelism_threads=self.run_opt.num_intra_threads,
                                inter_op_parallelism_threads=self.run_opt.num_inter_threads)
        # The stop_hook handles stopping after running given steps
        # stop_hook = tf.train.StopAtStepHook(last_step = stop_batch)
        # hooks = [self.sync_replicas_hook, stop_hook]
        hooks = [self.sync_replicas_hook]
        scaffold = tf.train.Scaffold(saver=saver)
        # Use monitor session for distributed computation
        self.sess = tf.train.MonitoredTrainingSession(master = self.run_opt.server.target,
                                                      is_chief = self.run_opt.is_chief,
                                                      config = config,
                                                      hooks = hooks,
                                                      scaffold = scaffold,
                                                      checkpoint_dir = ckpt_dir)
        # ,
        # save_checkpoint_steps = self.save_freq)

    def train (self, 
               data) :
        stop_batch = self.stop_batch
        if self.run_opt.is_distrib :
            self._init_sess_distrib()
        else :
            self._init_sess_serial()

        self.print_head()
        fp = None
        if self.run_opt.is_chief :
            fp = open(self.disp_file, "a")

        cur_batch = self.sess.run(self.global_step)
        is_first_step = True
        self.cur_batch = cur_batch
        log.info("start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e" % 
                 (self.sess.run(self.learning_rate),
                  self.lr.value(cur_batch), 
                  self.lr.decay_steps_,
                  self.lr.decay_rate_,
                  self.lr.value(stop_batch)) 
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
            tb_test_writer = tf.summary.FileWriter(self.tensorboard_log_dir + '/test')
        else:
            tb_train_writer = None
            tb_test_writer = None
        
        train_time = 0
        while cur_batch < stop_batch :
            batch_data = data.get_batch (sys_probs = self.sys_probs,
                                         auto_prob_style = self.auto_prob_style
            )
            feed_dict_batch = {}
            for kk in batch_data.keys():
                if kk == 'find_type' or kk == 'type' :
                    continue
                if 'find_' in kk :
                    feed_dict_batch[self.place_holders[kk]] = batch_data[kk]
                else:
                    feed_dict_batch[self.place_holders[kk]] = np.reshape(batch_data[kk], [-1])
            for ii in ['type'] :
                feed_dict_batch[self.place_holders[ii]] = np.reshape(batch_data[ii], [-1])
            for ii in ['natoms_vec', 'default_mesh'] :
                feed_dict_batch[self.place_holders[ii]] = batch_data[ii]
            feed_dict_batch[self.place_holders['is_training']] = True

            if self.display_in_training and is_first_step :
                self.test_on_the_fly(fp, data, feed_dict_batch, tb_test_writer)
                is_first_step = False
            if self.timing_in_training : tic = time.time()
            # use tensorboard to visualize the training of deepmd-kit
            # it will takes some extra execution time to generate the tensorboard data
            if self.tensorboard :
                summary, _ = self.sess.run([summary_merged_op, self.train_op], feed_dict = feed_dict_batch, options=prf_options, run_metadata=prf_run_metadata)
                tb_train_writer.add_summary(summary, cur_batch)
            else :
                self.sess.run([self.train_op], feed_dict = feed_dict_batch, options=prf_options, run_metadata=prf_run_metadata)
            if self.timing_in_training : toc = time.time()
            if self.timing_in_training : train_time += toc - tic
            cur_batch = self.sess.run(self.global_step)
            self.cur_batch = cur_batch

            if self.display_in_training and (cur_batch % self.disp_freq == 0) :
                tic = time.time()
                self.test_on_the_fly(fp, data, feed_dict_batch, tb_test_writer)
                toc = time.time()
                test_time = toc - tic
                if self.timing_in_training :
                    log.info("batch %7d training time %.2f s, testing time %.2f s"
                                  % (cur_batch, train_time, test_time))
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

    def get_global_step (self) :
        return self.sess.run(self.global_step)

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
                         feed_dict_batch,
                         tb_writer) :
        # Do not need to pass numb_test here as data object already knows it.
        # Both DeepmdDataSystem and ClassArg parse the same json file
        test_data = data.get_test(n_test=data.get_sys_ntest())
        feed_dict_test = {}
        for kk in test_data.keys():
            if kk == 'find_type' or kk == 'type' :
                continue
            if 'find_' in kk:
                feed_dict_test[self.place_holders[kk]] = test_data[kk]
            else:
                # again the data object knows appropriate test data shape,
                # there is no need to slice again!
                # feed_dict_test[self.place_holders[kk]] = np.reshape(test_data[kk][:self.numb_test[data.pick_idx]], [-1])
                feed_dict_test[self.place_holders[kk]] = np.reshape(test_data[kk], [-1])
        for ii in ['type'] :
            feed_dict_test[self.place_holders[ii]] = np.reshape(test_data[ii], [-1])            
        for ii in ['natoms_vec', 'default_mesh'] :
            feed_dict_test[self.place_holders[ii]] = test_data[ii]
        feed_dict_test[self.place_holders['is_training']] = False

        cur_batch = self.cur_batch
        current_lr = self.sess.run(self.learning_rate)
        if self.run_opt.is_chief:
            print_str = "%7d" % cur_batch
            print_str += self.loss.print_on_training(
                tb_writer,
                cur_batch,
                self.sess,
                test_data['natoms_vec'],
                feed_dict_test,
                feed_dict_batch
            )

            print_str += "   %8.1e\n" % current_lr
            fp.write(print_str)
            fp.flush ()
