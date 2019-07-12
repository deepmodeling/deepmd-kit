#!/usr/bin/env python3
import os
import sys
import time
import shutil
import warnings
import numpy as np
import tensorflow as tf
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float
from Fitting import EnerFitting
from DescrptLocFrame import DescrptLocFrame
from DescrptSeA import DescrptSeA
from DescrptSeR import DescrptSeR
from Model import Model
from Loss import LossStd
from LearningRate import LearningRateExp

from tensorflow.python.framework import ops
from tensorflow.python.client import timeline

# load force module
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.so" )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.so")

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

from deepmd.common import j_must_have, j_must_have_d, j_have

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
        else :
            raise RuntimeError('unknow model type ' + model_type)

        # fitting net
        try: 
            fitting_type = fitting_param['type']
        except:
            fitting_type = 'ener'
        if fitting_type == 'ener':
            self.fitting = EnerFitting(fitting_param, self.descrpt)
        else :
            raise RuntimeError('unknow fitting type ' + fitting_type)

        # init model
        self.model = Model(model_param, self.descrpt, self.fitting)

        # learning rate
        lr_param = j_must_have(jdata, 'learning_rate')
        try: 
            lr_type = lr_param['type']
        except:
            lr_type = 'exp'
        if lr_type == 'exp':
            self.lr = LearningRateExp(lr_param)
        else :
            raise RuntimeError('unknow learning_rate type ' + lr_type)        

        # loss
        loss_param = j_must_have(jdata, 'loss')
        try: 
            loss_type = loss_param['type']
        except:
            loss_type = 'std'
        if loss_type == 'std':
            self.loss = LossStd(loss_param, self.lr.start_lr())
        else :
            raise RuntimeError('unknow loss type ' + loss_type)

        # training
        training_param = j_must_have(jdata, 'training')
        
        self.numb_test = j_must_have (training_param, 'numb_test')
        self.useBN = False

        self.disp_file = "lcurve.out"
        if j_have (training_param, "disp_file") : self.disp_file = training_param["disp_file"]
        self.disp_freq = j_must_have (training_param, 'disp_freq')
        self.save_freq = j_must_have (training_param, 'save_freq')
        self.save_ckpt = j_must_have (training_param, 'save_ckpt')

        self.display_in_training = j_must_have (training_param, 'disp_training')
        self.timing_in_training = j_must_have (training_param, 'time_training')
        self.profiling = False
        if j_have (training_param, 'profiling') :
            self.profiling = training_param['profiling']
            if self.profiling :
                self.profiling_file = j_must_have (training_param, 'profiling_file')

        self.sys_weights = None
        if j_have(training_param, 'sys_weights') :
            self.sys_weights = training_param['sys_weights']


    def _message (self, msg) :
        self.run_opt.message(msg)

    def build (self, 
               data) :
        self.ntypes = self.model.get_ntypes()
        assert (self.ntypes == data.get_ntypes()), "ntypes should match that found in data"

        self.batch_size = data.get_batch_size()

        self.numb_fparam = data.numb_fparam()
        if self.numb_fparam > 0 :
            self._message("training with %d frame parameter(s)" % self.numb_fparam)
        elif self.numb_fparam < 0 :
            self._message("training without frame parameter")
        else :
            raise RuntimeError("number of frame parameter == 0")

        self.type_map = data.get_type_map()

        davg, dstd, bias_e = self.model.data_stat(data)

        worker_device = "/job:%s/task:%d/%s" % (self.run_opt.my_job_name,
                                                self.run_opt.my_task_index,
                                                self.run_opt.my_device)

        with tf.device(tf.train.replica_device_setter(worker_device = worker_device,
                                                      cluster = self.run_opt.cluster_spec)):
            self._build_lr()
            self._build_network(davg, dstd, bias_e)
            self._build_training()


    def _build_lr(self):
        self._extra_train_ops   = []
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = self.lr.build(self.global_step)
        self._message("built lr")

    def _build_network(self, davg, dstd, bias_atom_e):

        self.t_prop_c           = tf.placeholder(tf.float32, [4],    name='t_prop_c')
        self.t_energy           = tf.placeholder(global_ener_float_precision, [None], name='t_energy')
        self.t_force            = tf.placeholder(global_tf_float_precision, [None], name='t_force')
        self.t_virial           = tf.placeholder(global_tf_float_precision, [None], name='t_virial')
        self.t_atom_ener        = tf.placeholder(global_tf_float_precision, [None], name='t_atom_ener')
        self.t_coord            = tf.placeholder(global_tf_float_precision, [None], name='i_coord')
        self.t_type             = tf.placeholder(tf.int32,   [None], name='i_type')
        self.t_natoms           = tf.placeholder(tf.int32,   [self.ntypes+2], name='i_natoms')
        self.t_box              = tf.placeholder(global_tf_float_precision, [None, 9], name='i_box')
        self.t_mesh             = tf.placeholder(tf.int32,   [None], name='i_mesh')
        self.is_training        = tf.placeholder(tf.bool)
        if self.numb_fparam > 0 :
            self.t_fparam       = tf.placeholder(global_tf_float_precision, [None], name='i_fparam')
        else :
            self.t_fparam       = None

        self.energy, self.force, self.virial, self.atom_ener, self.atom_virial\
            = self.model.build (self.t_coord, 
                                self.t_type, 
                                self.t_natoms, 
                                self.t_box, 
                                self.t_mesh,
                                self.t_fparam,
                                davg = davg,
                                dstd = dstd,
                                bias_atom_e = bias_atom_e, 
                                suffix = "", 
                                reuse = False)

        self.l2_l, self.l2_el, self.l2_fl, self.l2_vl, self.l2_ael \
            = self.loss.build (self.learning_rate,
                               self.t_natoms, \
                               self.t_prop_c, \
                               self.t_energy, self.energy, \
                               self.t_force, self.force, \
                               self.t_virial, self.virial, \
                               self.t_atom_ener, self.atom_ener, \
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
            batch_prop_c, \
                batch_energy, batch_force, batch_virial, batch_atom_ener, \
                batch_coord, batch_box, batch_type, batch_fparam, \
                natoms_vec, \
                default_mesh \
                = data.get_batch (sys_weights = self.sys_weights)
            cur_batch_size = batch_energy.shape[0]
            feed_dict_batch = {self.t_prop_c:        batch_prop_c,
                               self.t_energy:        batch_energy, 
                               self.t_force:         np.reshape(batch_force, [-1]),
                               self.t_virial:        np.reshape(batch_virial, [-1]),
                               self.t_atom_ener:     np.reshape(batch_atom_ener, [-1]),
                               self.t_coord:         np.reshape(batch_coord, [-1]),
                               self.t_box:           batch_box,
                               self.t_type:          np.reshape(batch_type, [-1]),
                               self.t_natoms:        natoms_vec,
                               self.t_mesh:          default_mesh,
                               self.is_training:     True}
            if self.numb_fparam > 0 :
                feed_dict_batch[self.t_fparam] = np.reshape(batch_fparam, [-1])
            if self.display_in_training and cur_batch == 0 :
                self.test_on_the_fly(fp, data, feed_dict_batch)
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
            prop_fmt = '   %9s %9s'
            print_str += prop_fmt % ('l2_tst', 'l2_trn')
            if self.loss.has_e :
                print_str += prop_fmt % ('l2_e_tst', 'l2_e_trn')
            if self.loss.has_ae :
                print_str += prop_fmt % ('l2_ae_tst', 'l2_ae_trn')
            if self.loss.has_f :
                print_str += prop_fmt % ('l2_f_tst', 'l2_f_trn')
            if self.loss.has_v :
                print_str += prop_fmt % ('l2_v_tst', 'l2_v_trn')
            print_str += '   %8s\n' % 'lr'
            fp.write(print_str)
            fp.close ()

    def test_on_the_fly (self,
                         fp,
                         data,
                         feed_dict_batch) :
        test_prop_c, \
            test_energy, test_force, test_virial, test_atom_ener, \
            test_coord, test_box, test_type, test_fparam, \
            natoms_vec, \
            default_mesh \
            = data.get_test ()
        feed_dict_test = {self.t_prop_c:        test_prop_c,
                          self.t_energy:        test_energy              [:self.numb_test],
                          self.t_force:         np.reshape(test_force    [:self.numb_test, :], [-1]),
                          self.t_virial:        np.reshape(test_virial   [:self.numb_test, :], [-1]),
                          self.t_atom_ener:     np.reshape(test_atom_ener[:self.numb_test, :], [-1]),
                          self.t_coord:         np.reshape(test_coord    [:self.numb_test, :], [-1]),
                          self.t_box:           test_box                 [:self.numb_test, :],
                          self.t_type:          np.reshape(test_type     [:self.numb_test, :], [-1]),
                          self.t_natoms:        natoms_vec,
                          self.t_mesh:          default_mesh,
                          self.is_training:     False}
        if self.numb_fparam > 0 :
            feed_dict_test[self.t_fparam] = np.reshape(test_fparam  [:self.numb_test, :], [-1])
        error_test, error_e_test, error_f_test, error_v_test, error_ae_test \
            = self.sess.run([self.l2_l, \
                             self.l2_el, \
                             self.l2_fl, \
                             self.l2_vl, \
                             self.l2_ael], 
                            feed_dict=feed_dict_test)
        error_train, error_e_train, error_f_train, error_v_train, error_ae_train \
            = self.sess.run([self.l2_l, \
                             self.l2_el, \
                             self.l2_fl, \
                             self.l2_vl, \
                             self.l2_ael], 
                            feed_dict=feed_dict_batch)
        cur_batch = self.cur_batch
        current_lr = self.sess.run(self.learning_rate)
        if self.run_opt.is_chief:
            print_str = "%7d" % cur_batch
            prop_fmt = "   %9.2e %9.2e"
            print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))
            if self.loss.has_e :
                print_str += prop_fmt % (np.sqrt(error_e_test) / natoms_vec[0], np.sqrt(error_e_train) / natoms_vec[0])
            if self.loss.has_ae :
                print_str += prop_fmt % (np.sqrt(error_ae_test), np.sqrt(error_ae_train))
            if self.loss.has_f :
                print_str += prop_fmt % (np.sqrt(error_f_test), np.sqrt(error_f_train))
            if self.loss.has_v :
                print_str += prop_fmt % (np.sqrt(error_v_test) / natoms_vec[0], np.sqrt(error_v_train) / natoms_vec[0])
            print_str += "   %8.1e\n" % current_lr
            fp.write(print_str)
            fp.flush ()


