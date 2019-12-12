#!/usr/bin/env python3
import os
import platform
import sys
import time
import shutil
import warnings
import numpy as np
from deepmd.env import tf
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float
from deepmd.Fitting import EnerFitting, WFCFitting, PolarFittingLocFrame, PolarFittingSeA, GlobalPolarFittingSeA, DipoleFittingSeA
from deepmd.DescrptLocFrame import DescrptLocFrame
from deepmd.DescrptSeA import DescrptSeA
from deepmd.DescrptSeR import DescrptSeR
from deepmd.DescrptSeAR import DescrptSeAR
from deepmd.Model import Model, WFCModel, DipoleModel, PolarModel, GlobalPolarModel
from deepmd.Loss import EnerStdLoss, TensorLoss
from deepmd.LearningRate import LearningRateExp

from tensorflow.python.framework import ops
from tensorflow.python.client import timeline

# load force module
if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.{}".format(ext) )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.{}".format(ext))

# load grad of force module
sys.path.append (module_path )
import deepmd._prod_force_grad
import deepmd._prod_virial_grad
import deepmd._prod_force_se_a_grad
import deepmd._prod_virial_se_a_grad
import deepmd._prod_force_se_r_grad
import deepmd._prod_virial_se_r_grad
import deepmd._soft_min_force_grad
import deepmd._soft_min_virial_grad
from deepmd.RunOptions import RunOptions
from deepmd.TabInter import TabInter

from deepmd.common import j_must_have, ClassArg, add_data_requirement, data_requirement

def _is_subdir(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    if path == directory:
        return False
    relative = os.path.relpath(path, directory) + os.sep
    return not relative.startswith(os.pardir + os.sep)


class NNPTrainer (object):
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

        # descriptor
        descrpt_type = j_must_have(descrpt_param, 'type')
        if descrpt_type == 'loc_frame':
            self.descrpt = DescrptLocFrame(descrpt_param)
        elif descrpt_type == 'se_a' :
            self.descrpt = DescrptSeA(descrpt_param)
        elif descrpt_type == 'se_r' :
            self.descrpt = DescrptSeR(descrpt_param)
        elif descrpt_type == 'se_ar' :
            self.descrpt = DescrptSeAR(descrpt_param)
        else :
            raise RuntimeError('unknow model type ' + descrpt_type)

        # fitting net
        try: 
            fitting_type = fitting_param['type']
        except:
            fitting_type = 'ener'
        if fitting_type == 'ener':
            self.fitting = EnerFitting(fitting_param, self.descrpt)
        elif fitting_type == 'wfc':            
            self.fitting = WFCFitting(fitting_param, self.descrpt)
        elif fitting_type == 'dipole':
            if descrpt_type == 'se_a':
                self.fitting = DipoleFittingSeA(fitting_param, self.descrpt)
            else :
                raise RuntimeError('fitting dipole only supports descrptors: se_a')
        elif fitting_type == 'polar':
            if descrpt_type == 'loc_frame':
                self.fitting = PolarFittingLocFrame(fitting_param, self.descrpt)
            elif descrpt_type == 'se_a':
                self.fitting = PolarFittingSeA(fitting_param, self.descrpt)
            else :
                raise RuntimeError('fitting polar only supports descrptors: loc_frame and se_a')
        elif fitting_type == 'global_polar':
            if descrpt_type == 'se_a':
                self.fitting = GlobalPolarFittingSeA(fitting_param, self.descrpt)
            else :
                raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_a')
        else :
            raise RuntimeError('unknow fitting type ' + fitting_type)

        # init model
        # infer model type by fitting_type
        if fitting_type == Model.model_type:
            self.model = Model(model_param, self.descrpt, self.fitting)
        elif fitting_type == 'wfc':
            self.model = WFCModel(model_param, self.descrpt, self.fitting)
        elif fitting_type == 'dipole':
            self.model = DipoleModel(model_param, self.descrpt, self.fitting)
        elif fitting_type == 'polar':
            self.model = PolarModel(model_param, self.descrpt, self.fitting)
        elif fitting_type == 'global_polar':
            self.model = GlobalPolarModel(model_param, self.descrpt, self.fitting)
        else :
            raise RuntimeError('get unknown fitting type when building model')

        # learning rate
        lr_param = j_must_have(jdata, 'learning_rate')
        try: 
            lr_type = lr_param['type']
        except:
            lr_type = 'exp'
        if lr_type == 'exp':
            self.lr = LearningRateExp(lr_param)
        else :
            raise RuntimeError('unknown learning_rate type ' + lr_type)        

        # loss
        # infer loss type by fitting_type
        try :
            loss_param = jdata['loss']
        except:
            loss_param = None
        if fitting_type == 'ener':
            self.loss = EnerStdLoss(loss_param, starter_learning_rate = self.lr.start_lr())
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

        tr_args = ClassArg()\
                  .add('numb_test',     int,    default = 1)\
                  .add('disp_file',     str,    default = 'lcurve.out')\
                  .add('disp_freq',     int,    default = 100)\
                  .add('save_freq',     int,    default = 1000)\
                  .add('save_ckpt',     str,    default = 'model.ckpt')\
                  .add('display_in_training', bool, default = True)\
                  .add('timing_in_training',  bool, default = True)\
                  .add('profiling',     bool,   default = False)\
                  .add('profiling_file',str,    default = 'timeline.json')\
                  .add('sys_weights',   list    )
        tr_data = tr_args.parse(training_param)
        self.numb_test = tr_data['numb_test']
        self.disp_file = tr_data['disp_file']
        self.disp_freq = tr_data['disp_freq']
        self.save_freq = tr_data['save_freq']
        self.save_ckpt = tr_data['save_ckpt']
        self.display_in_training = tr_data['display_in_training']
        self.timing_in_training  = tr_data['timing_in_training']
        self.profiling = tr_data['profiling']
        self.profiling_file = tr_data['profiling_file']
        self.sys_weights = tr_data['sys_weights']        
        self.useBN = False
        if fitting_type == 'ener' and  self.fitting.get_numb_fparam() > 0 :
            self.numb_fparam = self.fitting.get_numb_fparam()
        else :
            self.numb_fparam = 0


    def _message (self, msg) :
        self.run_opt.message(msg)

    def build (self, 
               data) :
        self.ntypes = self.model.get_ntypes()
        assert (self.ntypes == data.get_ntypes()), "ntypes should match that found in data"

        self.batch_size = data.get_batch_size()

        if self.numb_fparam > 0 :
            self._message("training with %d frame parameter(s)" % self.numb_fparam)
        else:
            self._message("training without frame parameter")

        self.type_map = data.get_type_map()

        self.model.data_stat(data)

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
        self.learning_rate = self.lr.build(self.global_step)
        self._message("built lr")

    def _build_network(self, data):        
        self.place_holders = {}
        data_dict = data.get_data_dict()
        for kk in data_dict.keys():
            if kk == 'type':
                continue
            prec = global_tf_float_precision
            if data_dict[kk]['high_prec'] :
                prec = global_ener_float_precision
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

        self._message("built network")

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
        self._message("built training")

    def _init_sess_serial(self) :
        self.sess = tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads=self.run_opt.num_intra_threads, 
                                  inter_op_parallelism_threads=self.run_opt.num_inter_threads
            ))
        self.saver = tf.train.Saver()
        saver = self.saver
        if self.run_opt.init_mode == 'init_from_scratch' :
            self._message("initialize model from scratch")
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            fp = open(self.disp_file, "w")
            fp.close ()
        elif self.run_opt.init_mode == 'init_from_model' :
            self._message("initialize from model %s" % self.run_opt.init_model)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            saver.restore (self.sess, self.run_opt.init_model)            
            self.sess.run(self.global_step.assign(0))
            fp = open(self.disp_file, "w")
            fp.close ()
        elif self.run_opt.init_mode == 'restart' :
            self._message("restart from model %s" % self.run_opt.restart)
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            saver.restore (self.sess, self.run_opt.restart)
        else :
            raise RuntimeError ("unkown init mode")

    def _init_sess_distrib(self):
        ckpt_dir = os.path.join(os.getcwd(), self.save_ckpt)
        assert(_is_subdir(ckpt_dir, os.getcwd())), "the checkpoint dir must be a subdir of the current dir"
        if self.run_opt.init_mode == 'init_from_scratch' :
            self._message("initialize model from scratch")
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
            self._message("restart from model %s" % ckpt_dir)
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
               data, 
               stop_batch) :
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
        self.run_opt.message("start training at lr %.2e (== %.2e), final lr will be %.2e" % 
                             (self.sess.run(self.learning_rate),
                              self.lr.value(cur_batch), 
                              self.lr.value(stop_batch)) 
        )

        prf_options = None
        prf_run_metadata = None
        if self.profiling :
            prf_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            prf_run_metadata = tf.RunMetadata()

        train_time = 0
        while cur_batch < stop_batch :
            batch_data = data.get_batch (sys_weights = self.sys_weights)
            cur_batch_size = batch_data["coord"].shape[0]
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
                self.test_on_the_fly(fp, data, feed_dict_batch)
                is_first_step = False
            if self.timing_in_training : tic = time.time()
            self.sess.run([self.train_op], feed_dict = feed_dict_batch, options=prf_options, run_metadata=prf_run_metadata)
            if self.timing_in_training : toc = time.time()
            if self.timing_in_training : train_time += toc - tic
            cur_batch = self.sess.run(self.global_step)
            self.cur_batch = cur_batch

            if self.display_in_training and (cur_batch % self.disp_freq == 0) :
                tic = time.time()
                self.test_on_the_fly(fp, data, feed_dict_batch)
                toc = time.time()
                test_time = toc - tic
                if self.timing_in_training :
                    self._message("batch %7d training time %.2f s, testing time %.2f s"
                                  % (cur_batch, train_time, test_time))
                    train_time = 0
                if self.save_freq > 0 and cur_batch % self.save_freq == 0 and self.run_opt.is_chief :
                    if self.saver is not None :
                        self.saver.save (self.sess, os.getcwd() + "/" + self.save_ckpt)
                        self._message("saved checkpoint %s" % self.save_ckpt)
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
                         feed_dict_batch) :
        test_data = data.get_test ()
        feed_dict_test = {}
        for kk in test_data.keys():
            if kk == 'find_type' or kk == 'type' :
                continue
            if 'find_' in kk:
                feed_dict_test[self.place_holders[kk]] = test_data[kk]
            else:
                feed_dict_test[self.place_holders[kk]] = np.reshape(test_data[kk][:self.numb_test], [-1])
        for ii in ['type'] :
            feed_dict_test[self.place_holders[ii]] = np.reshape(test_data[ii][:self.numb_test], [-1])            
        for ii in ['natoms_vec', 'default_mesh'] :
            feed_dict_test[self.place_holders[ii]] = test_data[ii]
        feed_dict_test[self.place_holders['is_training']] = False

        cur_batch = self.cur_batch
        current_lr = self.sess.run(self.learning_rate)
        if self.run_opt.is_chief:
            print_str = "%7d" % cur_batch
            print_str += self.loss.print_on_training(self.sess,
                                                     test_data['natoms_vec'],
                                                     feed_dict_test,
                                                     feed_dict_batch)
            print_str += "   %8.1e\n" % current_lr
            fp.write(print_str)
            fp.flush ()


