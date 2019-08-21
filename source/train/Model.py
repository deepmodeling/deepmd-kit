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
import deepmd._prod_force_norot_grad
import deepmd._prod_virial_norot_grad
from deepmd.RunOptions import RunOptions

def j_must_have (jdata, key) :
    if not key in jdata.keys() :
        raise RuntimeError ("json database must provide key " + key )
    else :
        return jdata[key]

def j_must_have_d (jdata, key, deprecated_key) :
    if not key in jdata.keys() :
        # raise RuntimeError ("json database must provide key " + key )
        for ii in deprecated_key :
            if ii in jdata.keys() :
                warnings.warn("the key \"%s\" is deprecated, please use \"%s\" instead" % (ii,key))
                return jdata[ii]
        raise RuntimeError ("json database must provide key " + key )        
    else :
        return jdata[key]

def j_have (jdata, key) :
    return key in jdata.keys() 

def _is_subdir(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    if path == directory:
        return False
    relative = os.path.relpath(path, directory) + os.sep
    return not relative.startswith(os.pardir + os.sep)

class LearingRate (object) :
    def __init__ (self, 
                  jdata, 
                  tot_numb_batches) :
        self.decay_steps_ = j_must_have(jdata, 'decay_steps')
        self.decay_rate_ = j_must_have(jdata, 'decay_rate')
        self.start_lr_ = j_must_have(jdata, 'start_lr')        
        self.tot_numb_batches = tot_numb_batches

    def value (self, 
              batch) :
        return self.start_lr_ * np.power (self.decay_rate_, (batch // self.decay_steps()))

    def decay_steps (self) :
#        return self.decay_steps_ * self.tot_numb_batches
        return self.decay_steps_
    
    def decay_rate (self) : 
        return self.decay_rate_

    def start_lr (self) :
        return self.start_lr_

class NNPModel (object):
    def __init__(self, 
                 jdata, 
                 run_opt):
        self.run_opt = run_opt
        self._init_param(jdata)
        self.null_mesh = tf.constant ([-1])

    def _init_param(self, jdata):
        # descrpt config
        self.use_smooth = False
        if j_have (jdata, "use_smooth") :
            self.use_smooth = jdata["use_smooth"]
        self.sel_a = j_must_have (jdata, 'sel_a')
        self.sel_r = [ 0 for ii in range(len(self.sel_a)) ]
        if not self.use_smooth :
            self.sel_r = j_must_have (jdata, 'sel_r')
        else :
            if j_have (jdata, 'sel_r') :
                warnings.warn ('ignoring key sel_r in the json database and set sel_r to %s' % str(self.sel_r))
        self.rcut_a = -1
        self.rcut_r = j_must_have (jdata, 'rcut')
        if j_have(jdata, 'rcut_smth') :
            self.rcut_r_smth = jdata['rcut_smth']
        else :
            self.rcut_r_smth = self.rcut_r
        # axis
        self.axis_rule = []
        if j_have (jdata, 'axis_rule') :
            self.axis_rule = jdata['axis_rule']
        # filter of smooth version
        if self.use_smooth :
            if j_have(jdata, 'coord_norm') :
                self.coord_norm = jdata['coord_norm']
            else :
                self.coord_norm = True
            self.filter_neuron = j_must_have (jdata, 'filter_neuron')
            self.n_axis_neuron = j_must_have_d (jdata, 'axis_neuron', ['n_axis_neuron'])
            self.filter_resnet_dt = False
            if j_have(jdata, 'filter_resnet_dt') :
                self.filter_resnet_dt = jdata['filter_resnet_dt']        
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        # network size
        self.n_neuron = j_must_have_d (jdata, 'fitting_neuron', ['n_neuron'])
        self.resnet_dt = True
        if j_have(jdata, 'resnet_dt') :
            warnings.warn("the key \"%s\" is deprecated, please use \"%s\" instead" % ('resnet_dt','fitting_resnet_dt'))
            self.resnet_dt = jdata['resnet_dt']
        if j_have(jdata, 'fitting_resnet_dt') :
            self.resnet_dt = jdata['fitting_resnet_dt']
        if self.use_smooth :            
            if j_have(jdata, 'type_fitting_net') :
                self.type_fitting_net = jdata['type_fitting_net']
            else :
                self.type_fitting_net = False            

        self.numb_test = j_must_have (jdata, 'numb_test')
        self.useBN = False

        self.start_pref_e = j_must_have (jdata, 'start_pref_e')
        self.limit_pref_e = j_must_have (jdata, 'limit_pref_e')
        self.start_pref_f = j_must_have (jdata, 'start_pref_f')
        self.limit_pref_f = j_must_have (jdata, 'limit_pref_f')
        self.start_pref_v = j_must_have (jdata, 'start_pref_v')
        self.limit_pref_v = j_must_have (jdata, 'limit_pref_v')
        self.start_pref_ae = 0
        if j_have(jdata, 'start_pref_ae') :
            self.start_pref_ae = jdata['start_pref_ae']
        self.limit_pref_ae = 0
        if j_have(jdata, 'limit_pref_ae') :
            self.limit_pref_ae = jdata['limit_pref_ae']
        self.has_e = (self.start_pref_e != 0 or self.limit_pref_e != 0)
        self.has_f = (self.start_pref_f != 0 or self.limit_pref_f != 0)
        self.has_v = (self.start_pref_v != 0 or self.limit_pref_v != 0)
        self.has_ae = (self.start_pref_ae != 0 or self.limit_pref_ae != 0)

        self.disp_file = "lcurve.out"
        if j_have (jdata, "disp_file") : self.disp_file = jdata["disp_file"]
        self.disp_freq = j_must_have (jdata, 'disp_freq')
        self.save_freq = j_must_have (jdata, 'save_freq')
        self.save_ckpt = j_must_have (jdata, 'save_ckpt')

        self.seed = None
        if j_have (jdata, 'seed') :
            self.seed = jdata['seed']

        self.display_in_training = j_must_have (jdata, 'disp_training')
        self.timing_in_training = j_must_have (jdata, 'time_training')
        self.profiling = False
        if j_have (jdata, 'profiling') :
            self.profiling = jdata['profiling']
            if self.profiling :
                self.profiling_file = j_must_have (jdata, 'profiling_file')

        self.sys_weights = None
        if j_have(jdata, 'sys_weights') :
            self.sys_weights = jdata['sys_weights']


    def _message (self, msg) :
        self.run_opt.message(msg)

    def build (self, 
               data, 
               lr) :
        self.lr = lr
        self.ntypes = len(self.sel_a)
        assert (self.ntypes == len(self.sel_r)), "size sel r array should match ntypes"
        assert (self.ntypes == data.get_ntypes()), "ntypes should match that found in data"

        self.batch_size = data.get_batch_size()

        davg, dstd, bias_e = self._data_stat(data)

        worker_device = "/job:%s/task:%d/%s" % (self.run_opt.my_job_name,
                                                self.run_opt.my_task_index,
                                                self.run_opt.my_device)
        with tf.device(tf.train.replica_device_setter(worker_device = worker_device,
                                                      cluster = self.run_opt.cluster_spec)):
            self._build_lr(lr)
            self._build_network(davg, dstd, bias_e)
            self._build_training()

    def _data_stat(self, data):
        all_stat_coord = []
        all_stat_box = []
        all_stat_type = []
        all_natoms_vec = []
        all_default_mesh = []
        for ii in range(data.get_nsystems()) :
            stat_prop_c, \
                stat_energy, stat_force, stat_virial, start_atom_ener, \
                stat_coord, stat_box, stat_type, natoms_vec, default_mesh \
                = data.get_batch (sys_idx = ii)
            natoms_vec = natoms_vec.astype(np.int32)            
            all_stat_coord.append(stat_coord)
            all_stat_box.append(stat_box)
            all_stat_type.append(stat_type)
            all_natoms_vec.append(natoms_vec)
            all_default_mesh.append(default_mesh)

        if self.use_smooth and not self.coord_norm :
            davg, dstd = self.no_norm_dstats ()
            self._message("skipped coord/descrpt stats")
        else :
            davg, dstd = self.compute_dstats (all_stat_coord, all_stat_box, all_stat_type, all_natoms_vec, all_default_mesh)
            self._message("computed coord/descrpt stats")
        if self.run_opt.is_chief:
            np.savetxt ("stat.avg.out", davg.T)
            np.savetxt ("stat.std.out", dstd.T)

        bias_atom_e = data.compute_energy_shift()
        self._message("computed energy bias")

        return davg, dstd, bias_atom_e

    def _build_lr(self, lr):
        self._extra_train_ops   = []
        self.global_step = tf.train.get_or_create_global_step()
        self.starter_learning_rate = lr.start_lr()
        self.learning_rate = tf.train.exponential_decay(lr.start_lr(), 
                                                        self.global_step,
                                                        lr.decay_steps(),
                                                        lr.decay_rate(), 
                                                        staircase=True)
        self._message("built lr")

    def _build_network(self, davg, dstd, bias_atom_e):
        self.t_avg = tf.get_variable('t_avg', 
                                     davg.shape, 
                                     dtype = global_tf_float_precision,
                                     trainable = False,
                                     initializer = tf.constant_initializer(davg, dtype = global_tf_float_precision))
        self.t_std = tf.get_variable('t_std', 
                                     dstd.shape, 
                                     dtype = global_tf_float_precision,
                                     trainable = False,
                                     initializer = tf.constant_initializer(dstd, dtype = global_tf_float_precision))

        t_rcut = tf.constant(np.max([self.rcut_r, self.rcut_a]), name = 't_rcut', dtype = global_tf_float_precision)
        t_ntypes = tf.constant(self.ntypes, name = 't_ntypes', dtype = tf.int32)

        self.t_prop_c           = tf.placeholder(tf.float32, [4],    name='t_prop_c')
        self.t_energy           = tf.placeholder(global_ener_float_precision, [None], name='t_energy')
        self.t_force            = tf.placeholder(global_tf_float_precision, [None], name='t_force')
        self.t_virial           = tf.placeholder(global_tf_float_precision, [None], name='t_virial')
        self.t_atom_ener        = tf.placeholder(global_tf_float_precision, [None], name='t_atom_ener')
        self.t_coord            = tf.placeholder(global_tf_float_precision, [None], name='t_coord')
        self.t_type             = tf.placeholder(tf.int32,   [None], name='t_type')
        self.t_natoms           = tf.placeholder(tf.int32,   [self.ntypes+2], name='t_natoms')
        self.t_box              = tf.placeholder(global_tf_float_precision, [None, 9], name='t_box')
        self.t_mesh             = tf.placeholder(tf.int32,   [None], name='t_mesh')
        self.is_training        = tf.placeholder(tf.bool)

        self.batch_size_value = list(set(self.batch_size))
        self.batch_size_value.sort()
        self.numb_batch_size_value = len(self.batch_size_value)

        self.energy_frz, self.force_frz, self.virial_frz, self.atom_ener_frz \
            = self.build_interaction (1,
                                      self.t_coord, 
                                      self.t_type, 
                                      self.t_natoms, 
                                      self.t_box, 
                                      self.t_mesh, 
                                      bias_atom_e = bias_atom_e, 
                                      suffix = "test", 
                                      reuse = False)
        self.energy_tst, self.force_tst, self.virial_tst, self.atom_ener_tst \
            = self.build_interaction (self.numb_test,   
                                      self.t_coord, 
                                      self.t_type, 
                                      self.t_natoms, 
                                      self.t_box, 
                                      self.t_mesh, 
                                      bias_atom_e = bias_atom_e, 
                                      suffix = "train_test", 
                                      reuse = True)
        self.energy_bch = []
        self.force_bch = []
        self.virial_bch = []
        self.atom_ener_bch = []
        for ii in range(self.numb_batch_size_value) :
            tmp_energy_bch, tmp_force_bch, tmp_virial_bch, tmp_atom_ener_bch \
                = self.build_interaction (self.batch_size_value[ii],  
                                          self.t_coord, 
                                          self.t_type, 
                                          self.t_natoms, 
                                          self.t_box, 
                                          self.t_mesh, 
                                          bias_atom_e = bias_atom_e, 
                                          suffix = "train_batch_" + str(self.batch_size_value[ii]), 
                                          reuse = True)
            self.energy_bch.append(tmp_energy_bch)
            self.force_bch.append(tmp_force_bch)
            self.virial_bch.append(tmp_virial_bch)
            self.atom_ener_bch.append(tmp_atom_ener_bch)

        self.l2_l_tst, self.l2_el_tst, self.l2_fl_tst, self.l2_vl_tst, self.l2_ael_tst \
            = self.loss (self.t_natoms, \
                         self.t_prop_c, \
                         self.t_energy, self.energy_tst, \
                         self.t_force, self.force_tst, \
                         self.t_virial, self.virial_tst, \
                         self.t_atom_ener, self.atom_ener_tst, \
                         suffix = "train_test")
        self.l2_l_bch = []
        self.l2_el_bch = []
        self.l2_fl_bch = []
        self.l2_vl_bch = []
        self.l2_ael_bch = []
        for ii in range(self.numb_batch_size_value) :                    
            tmp_l2_l_bch, tmp_l2_el_bch, tmp_l2_fl_bch, tmp_l2_vl_bch, tmp_l2_ael_bch \
                = self.loss (self.t_natoms, \
                             self.t_prop_c, \
                             self.t_energy, self.energy_bch[ii], \
                             self.t_force, self.force_bch[ii], \
                             self.t_virial, self.virial_bch[ii], \
                             self.t_atom_ener, self.atom_ener_bch[ii], \
                             suffix = "train_batch_" + str(self.batch_size_value[ii]))
            self.l2_l_bch.append(tmp_l2_l_bch)
            self.l2_el_bch.append(tmp_l2_el_bch)
            self.l2_fl_bch.append(tmp_l2_fl_bch)
            self.l2_vl_bch.append(tmp_l2_vl_bch)
            self.l2_ael_bch.append(tmp_l2_ael_bch)

        self._message("built network")

    def _build_training(self):
        self.train_op = []
        trainable_variables = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        if self.run_opt.is_distrib :
            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer,
                replicas_to_aggregate = self.run_opt.cluster_spec.num_tasks("worker"),
                total_num_replicas = self.run_opt.cluster_spec.num_tasks("worker"),
                name = "sync_replicas")
            self.sync_replicas_hook = optimizer.make_session_run_hook(self.run_opt.is_chief)            
        for ii in range(self.numb_batch_size_value) :
            grads = tf.gradients(self.l2_l_bch[ii], trainable_variables)
            apply_op = optimizer.apply_gradients (zip (grads, trainable_variables),
                                                  global_step=self.global_step,
                                                  name='train_step')
            train_ops = [apply_op] + self._extra_train_ops
            self.train_op.append(tf.group(*train_ops))
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
                batch_coord, batch_box, batch_type, \
                natoms_vec, \
                default_mesh \
                = data.get_batch (sys_weights = self.sys_weights)
            cur_batch_size = batch_energy.shape[0]
            cur_bs_idx = self.batch_size_value.index(cur_batch_size)
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
            if self.display_in_training and cur_batch == 0 :
                self.test_on_the_fly(fp, data, feed_dict_batch, cur_bs_idx)
            if self.timing_in_training : tic = time.time()
            self.sess.run([self.train_op[cur_bs_idx]], feed_dict = feed_dict_batch, options=prf_options, run_metadata=prf_run_metadata)
            if self.timing_in_training : toc = time.time()
            if self.timing_in_training : train_time += toc - tic
            cur_batch = self.sess.run(self.global_step)
            self.cur_batch = cur_batch

            if self.display_in_training and (cur_batch % self.disp_freq == 0) :
                tic = time.time()
                self.test_on_the_fly(fp, data, feed_dict_batch, cur_bs_idx)
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
            if self.has_e :
                print_str += prop_fmt % ('l2_e_tst', 'l2_e_trn')
            if self.has_ae :
                print_str += prop_fmt % ('l2_ae_tst', 'l2_ae_trn')
            if self.has_f :
                print_str += prop_fmt % ('l2_f_tst', 'l2_f_trn')
            if self.has_v :
                print_str += prop_fmt % ('l2_v_tst', 'l2_v_trn')
            print_str += '   %8s\n' % 'lr'
            fp.write(print_str)
            fp.close ()

    def test_on_the_fly (self,
                         fp,
                         data,
                         feed_dict_batch, 
                         ii) :
        test_prop_c, \
            test_energy, test_force, test_virial, test_atom_ener, \
            test_coord, test_box, test_type, \
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
        error_test, error_e_test, error_f_test, error_v_test, error_ae_test \
            = self.sess.run([self.l2_l_tst, \
                             self.l2_el_tst, \
                             self.l2_fl_tst, \
                             self.l2_vl_tst, \
                             self.l2_ael_tst], 
                            feed_dict=feed_dict_test)
        error_train, error_e_train, error_f_train, error_v_train, error_ae_train \
            = self.sess.run([self.l2_l_bch[ii], \
                             self.l2_el_bch[ii], \
                             self.l2_fl_bch[ii], \
                             self.l2_vl_bch[ii], \
                             self.l2_ael_bch[ii]], 
                            feed_dict=feed_dict_batch)
        cur_batch = self.cur_batch
        current_lr = self.sess.run(self.learning_rate)
        if self.run_opt.is_chief:
            print_str = "%7d" % cur_batch
            prop_fmt = "   %9.2e %9.2e"
            print_str += prop_fmt % (np.sqrt(error_test), np.sqrt(error_train))
            if self.has_e :
                print_str += prop_fmt % (np.sqrt(error_e_test) / natoms_vec[0], np.sqrt(error_e_train) / natoms_vec[0])
            if self.has_ae :
                print_str += prop_fmt % (np.sqrt(error_ae_test), np.sqrt(error_ae_train))
            if self.has_f :
                print_str += prop_fmt % (np.sqrt(error_f_test), np.sqrt(error_f_train))
            if self.has_v :
                print_str += prop_fmt % (np.sqrt(error_v_test) / natoms_vec[0], np.sqrt(error_v_train) / natoms_vec[0])
            print_str += "   %8.1e\n" % current_lr
            fp.write(print_str)
            fp.flush ()

    def compute_dstats_sys_smth (self,
                                 data_coord, 
                                 data_box, 
                                 data_atype, 
                                 natoms_vec,
                                 mesh,
                                 reuse = None) :    
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            descrpt, descrpt_deriv, rij, nlist \
                = op_module.descrpt_norot (tf.constant(data_coord),
                                           tf.constant(data_atype),
                                           tf.constant(natoms_vec, dtype = tf.int32),
                                           tf.constant(data_box),
                                           tf.constant(mesh),
                                           tf.constant(avg_zero),
                                           tf.constant(std_ones),
                                           rcut_a = self.rcut_a,
                                           rcut_r = self.rcut_r,
                                           rcut_r_smth = self.rcut_r_smth,
                                           sel_a = self.sel_a,
                                           sel_r = self.sel_r)
        # self.sess.run(tf.global_variables_initializer())
        sub_sess = tf.Session(graph = sub_graph,
                              config=tf.ConfigProto(intra_op_parallelism_threads=self.run_opt.num_intra_threads, 
                                                    inter_op_parallelism_threads=self.run_opt.num_inter_threads

                              ))
        dd_all = sub_sess.run(descrpt)
        sub_sess.close()
        natoms = natoms_vec
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysr = []
        sysa = []
        sysn = []
        sysr2 = []
        sysa2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index        
            # compute
            dd = np.reshape (dd, [-1, 4])
            ddr = dd[:,:1]
            dda = dd[:,1:]
            sumr = np.sum(ddr)
            suma = np.sum(dda) / 3.
            sumn = dd.shape[0]
            sumr2 = np.sum(np.multiply(ddr, ddr))
            suma2 = np.sum(np.multiply(dda, dda)) / 3.
            sysr.append(sumr)
            sysa.append(suma)
            sysn.append(sumn)
            sysr2.append(sumr2)
            sysa2.append(suma2)
        return sysr, sysr2, sysa, sysa2, sysn

    def compute_dstats_sys_nonsmth (self,
                                    data_coord, 
                                    data_box, 
                                    data_atype, 
                                    natoms_vec,
                                    mesh,
                                    reuse = None) :    
        avg_zero = np.zeros([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        std_ones = np.ones ([self.ntypes,self.ndescrpt]).astype(global_np_float_precision)
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            descrpt, descrpt_deriv, rij, nlist, axis \
                = op_module.descrpt (tf.constant(data_coord),
                                     tf.constant(data_atype),
                                     tf.constant(natoms_vec, dtype = tf.int32),
                                     tf.constant(data_box),
                                     tf.constant(mesh),
                                     tf.constant(avg_zero),
                                     tf.constant(std_ones),
                                     rcut_a = self.rcut_a,
                                     rcut_r = self.rcut_r,
                                     sel_a = self.sel_a,
                                     sel_r = self.sel_r,
                                     axis_rule = self.axis_rule)
        # self.sess.run(tf.global_variables_initializer())
        sub_sess = tf.Session(graph = sub_graph, 
                              config=tf.ConfigProto(intra_op_parallelism_threads=self.run_opt.num_intra_threads, 
                                                    inter_op_parallelism_threads=self.run_opt.num_inter_threads
                              ))
        dd_all = sub_sess.run(descrpt)
        sub_sess.close()
        natoms = natoms_vec
        dd_all = np.reshape(dd_all, [-1, self.ndescrpt * natoms[0]])
        start_index = 0
        sysv = []
        sysn = []
        sysv2 = []
        for type_i in range(self.ntypes):
            end_index = start_index + self.ndescrpt * natoms[2+type_i]
            dd = dd_all[:, start_index:end_index]
            dd = np.reshape(dd, [-1, self.ndescrpt])
            start_index = end_index        
            # compute
            sumv = np.sum(dd, axis = 0)
            sumn = dd.shape[0]
            sumv2 = np.sum(np.multiply(dd,dd), axis = 0)            
            sysv.append(sumv)
            sysn.append(sumn)
            sysv2.append(sumv2)
        return sysv, sysv2, sysn


    def compute_std (self,sumv2, sumv, sumn) :
        return np.sqrt(sumv2/sumn - np.multiply(sumv/sumn, sumv/sumn))

    def compute_dstats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                        mesh,
                        reuse = None) :    
        env_bk = None
        if 'TF_CPP_MIN_LOG_LEVEL' in os.environ:
            env_bk = os.environ['TF_CPP_MIN_LOG_LEVEL']
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        all_davg = []
        all_dstd = []
        if self.use_smooth:
            sumr = []
            suma = []
            sumn = []
            sumr2 = []
            suma2 = []
            for cc,bb,tt,nn,mm in zip(data_coord,data_box,data_atype,natoms_vec,mesh) :
                sysr,sysr2,sysa,sysa2,sysn \
                    = self.compute_dstats_sys_smth(cc,bb,tt,nn,mm,reuse)
                sumr.append(sysr)
                suma.append(sysa)
                sumn.append(sysn)
                sumr2.append(sysr2)
                suma2.append(sysa2)
            sumr = np.sum(sumr, axis = 0)
            suma = np.sum(suma, axis = 0)
            sumn = np.sum(sumn, axis = 0)
            sumr2 = np.sum(sumr2, axis = 0)
            suma2 = np.sum(suma2, axis = 0)
            for type_i in range(self.ntypes) :
                davgunit = [sumr[type_i]/sumn[type_i], 0, 0, 0]
                dstdunit = [self.compute_std(sumr2[type_i], sumr[type_i], sumn[type_i]), 
                            self.compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self.compute_std(suma2[type_i], suma[type_i], sumn[type_i]), 
                            self.compute_std(suma2[type_i], suma[type_i], sumn[type_i])
                            ]
                davg = np.tile(davgunit, self.ndescrpt // 4)
                dstd = np.tile(dstdunit, self.ndescrpt // 4)
                all_davg.append(davg)
                all_dstd.append(dstd)
        else :
            sumv = []
            sumn = []
            sumv2 = []
            for cc,bb,tt,nn,mm in zip(data_coord,data_box,data_atype,natoms_vec,mesh) :
                sysv,sysv2,sysn \
                    = self.compute_dstats_sys_nonsmth(cc,bb,tt,nn,mm,reuse)
                sumv.append(sysv)
                sumn.append(sysn)
                sumv2.append(sysv2)
            sumv = np.sum(sumv, axis = 0)
            sumn = np.sum(sumn, axis = 0)
            sumv2 = np.sum(sumv2, axis = 0)
            for type_i in range(self.ntypes) :
                davg = sumv[type_i] /  sumn[type_i]
                dstd = self.compute_std(sumv2[type_i], sumv[type_i], sumn[type_i])
                for ii in range (len(dstd)) :
                    if (np.abs(dstd[ii]) < 1e-2) :
                        dstd[ii] = 1e-2            
                all_davg.append(davg)
                all_dstd.append(dstd)

        davg = np.array(all_davg)
        dstd = np.array(all_dstd)
        if env_bk is not None :
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = env_bk
        else :
            os.environ.pop('TF_CPP_MIN_LOG_LEVEL', None)
        return davg, dstd

    def no_norm_dstats (self, avgv = 0, stdv = 1) :
        davg = np.zeros([self.ntypes, self.ndescrpt]) + avgv
        dstd = np.ones ([self.ntypes, self.ndescrpt]) * stdv
        return davg, dstd

    def loss (self, 
              natoms,
              prop_c,
              energy, 
              energy_hat,
              force,
              force_hat, 
              virial,
              virial_hat, 
              atom_ener,
              atom_ener_hat, 
              suffix):
        l2_ener_loss = tf.reduce_mean( tf.square(energy - energy_hat), name='l2_'+suffix)

        force_reshape = tf.reshape (force, [-1])
        force_hat_reshape = tf.reshape (force_hat, [-1])
        l2_force_loss = tf.reduce_mean (tf.square(force_hat_reshape - force_reshape), name = "l2_force_" + suffix)

        virial_reshape = tf.reshape (virial, [-1])
        virial_hat_reshape = tf.reshape (virial_hat, [-1])
        l2_virial_loss = tf.reduce_mean (tf.square(virial_hat_reshape - virial_reshape), name = "l2_virial_" + suffix)

        atom_ener_reshape = tf.reshape (atom_ener, [-1])
        atom_ener_hat_reshape = tf.reshape (atom_ener_hat, [-1])
        l2_atom_ener_loss = tf.reduce_mean (tf.square(atom_ener_hat_reshape - atom_ener_reshape), name = "l2_atom_ener_" + suffix)

        atom_norm  = 1./ global_cvt_2_tf_float(natoms[0]) 
        atom_norm_ener  = 1./ global_cvt_2_ener_float(natoms[0]) 
        pref_e = global_cvt_2_ener_float(prop_c[0] * (self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * self.learning_rate / self.starter_learning_rate) )
        pref_f = global_cvt_2_tf_float(prop_c[1] * (self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * self.learning_rate / self.starter_learning_rate) )
        pref_v = global_cvt_2_tf_float(prop_c[2] * (self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * self.learning_rate / self.starter_learning_rate) )
        pref_ae= global_cvt_2_tf_float(prop_c[3] * (self.limit_pref_ae+ (self.start_pref_ae-self.limit_pref_ae) * self.learning_rate / self.starter_learning_rate) )

        l2_loss = 0
        if self.has_e :
            l2_loss += atom_norm_ener * (pref_e * l2_ener_loss)
        if self.has_f :
            l2_loss += global_cvt_2_ener_float(pref_f * l2_force_loss)
        if self.has_v :
            l2_loss += global_cvt_2_ener_float(atom_norm * (pref_v * l2_virial_loss))
        if self.has_ae :
            l2_loss += global_cvt_2_ener_float(pref_ae * l2_atom_ener_loss)

        return l2_loss, l2_ener_loss, l2_force_loss, l2_virial_loss, l2_atom_ener_loss

    def build_interaction (self, 
                           nframes,
                           coord_, 
                           atype_,
                           natoms,
                           box, 
                           mesh,
                           suffix, 
                           bias_atom_e = None,
                           reuse = None):        
        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        if self.use_smooth :
            descrpt, descrpt_deriv, rij, nlist \
                = op_module.descrpt_norot (coord,
                                           atype,
                                           natoms,
                                           box,                                    
                                           mesh,
                                           self.t_avg,
                                           self.t_std,
                                           rcut_a = self.rcut_a,
                                           rcut_r = self.rcut_r,
                                           rcut_r_smth = self.rcut_r_smth,
                                           sel_a = self.sel_a,
                                           sel_r = self.sel_r)
        else :
            descrpt, descrpt_deriv, rij, nlist, axis \
                = op_module.descrpt (coord,
                                     atype,
                                     natoms,
                                     box,                                    
                                     mesh,
                                     self.t_avg,
                                     self.t_std,
                                     rcut_a = self.rcut_a,
                                     rcut_r = self.rcut_r,
                                     sel_a = self.sel_a,
                                     sel_r = self.sel_r,
                                     axis_rule = self.axis_rule)

        descrpt_reshape = tf.reshape(descrpt, [-1, self.ndescrpt])
        
        atom_ener = self.build_atom_net (nframes, descrpt_reshape, natoms, bias_atom_e = bias_atom_e, reuse = reuse)

        energy_raw = tf.reshape(atom_ener, [-1, natoms[0]], name = 'atom_energy_'+suffix)
        energy = tf.reduce_sum(global_cvt_2_ener_float(energy_raw), axis=1, name='energy_'+suffix)

        net_deriv_tmp = tf.gradients (atom_ener, descrpt_reshape)
        net_deriv = net_deriv_tmp[0]
        net_deriv_reshape = tf.reshape (net_deriv, [-1, natoms[0] * self.ndescrpt])

        if self.use_smooth :
            force = op_module.prod_force_norot (net_deriv_reshape,
                                                descrpt_deriv,
                                                nlist,
                                                natoms,
                                                n_a_sel = self.nnei_a,
                                                n_r_sel = self.nnei_r)
        else :
            force = op_module.prod_force (net_deriv_reshape,
                                          descrpt_deriv,
                                          nlist,
                                          axis,
                                          natoms,
                                          n_a_sel = self.nnei_a,
                                          n_r_sel = self.nnei_r)
        force = tf.reshape (force, [-1, 3 * natoms[1]], name = "force_"+suffix)

        if self.use_smooth :
            virial, atom_virial \
                = op_module.prod_virial_norot (net_deriv_reshape,
                                               descrpt_deriv,
                                               rij,
                                               nlist,
                                               natoms,
                                               n_a_sel = self.nnei_a,
                                               n_r_sel = self.nnei_r)
        else :
            virial, atom_virial \
                = op_module.prod_virial (net_deriv_reshape,
                                         descrpt_deriv,
                                         rij,
                                         nlist,
                                         axis,
                                         natoms,
                                         n_a_sel = self.nnei_a,
                                         n_r_sel = self.nnei_r)
        virial = tf.reshape (virial, [-1, 9], name = "virial_"+suffix)
        atom_virial = tf.reshape (atom_virial, [-1, 9 * natoms[1]], name = "atom_virial_"+suffix)

        return energy, force, virial, energy_raw
    
    def build_atom_net (self, 
                        nframes,
                        inputs, 
                        natoms,
                        bias_atom_e = None,
                        reuse = None) :
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.ndescrpt * natoms[0]])
        shape = inputs.get_shape().as_list()
        if bias_atom_e is not None :
            assert(len(bias_atom_e) == self.ntypes)

        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.ndescrpt],
                                 [-1, natoms[2+type_i]* self.ndescrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.ndescrpt])
            start_index += natoms[2+type_i]
            if bias_atom_e is None :
                type_bias_ae = 0.0
            else :
                type_bias_ae = bias_atom_e[type_i]

            # compute atom energy
            if self.use_smooth :
                if self.type_fitting_net :
                    layer = self._DS_layer_type_ext(inputs_i, name='DS_layer_type_'+str(type_i), natoms=natoms, reuse=reuse, seed = self.seed)
                else :
                    layer = self._DS_layer(inputs_i, name='DS_layer_type_'+str(type_i), natoms=natoms, reuse=reuse, seed = self.seed)
                for ii in range(0,len(self.n_neuron)) :
                    if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                        layer+= self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i), reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                    else :
                        layer = self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i), reuse=reuse, seed = self.seed)
            else :
                layer = self._one_layer(inputs_i, self.n_neuron[0], name='layer_0_type_'+str(type_i), reuse=reuse, seed = self.seed)
                for ii in range(1,len(self.n_neuron)) :
                    layer = self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i), reuse=reuse, seed = self.seed)
            final_layer = self._one_layer(layer, 1, activation_fn = None, bavg = type_bias_ae, name='final_layer_type_'+str(type_i), reuse=reuse, seed = self.seed)
            final_layer = tf.reshape(final_layer, [nframes, natoms[2+type_i]])
            # final_layer = tf.cond (tf.equal(natoms[2+type_i], 0), lambda: tf.zeros((0, 0), dtype=global_tf_float_precision), lambda : tf.reshape(final_layer, [-1, natoms[2+type_i]]))

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)


        return tf.reshape(outs, [-1])

    def _one_layer(self, 
                   inputs, 
                   outputs_size, 
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None, 
                   use_timestep = False):
        with tf.variable_scope(name, reuse=reuse):
            shape = inputs.get_shape().as_list()
            w = tf.get_variable('matrix', 
                                [shape[1], outputs_size], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed))
            b = tf.get_variable('bias', 
                                [outputs_size], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
            hidden = tf.matmul(inputs, w) + b
            if activation_fn != None and use_timestep :
                idt = tf.get_variable('idt',
                                      [outputs_size],
                                      global_tf_float_precision,
                                      tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed))

        if activation_fn != None:
            if self.useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if use_timestep :
                    return activation_fn(hidden) * idt
                else :
                    return activation_fn(hidden)                    
        else:
            if self.useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden
    
    def _DS_layer(self, 
                   inputs, 
                   natoms,
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None):
        # natom x (nei x 4)
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          xyz_scatter_total = []
          for type_i in range(self.ntypes):
            # cut-out inputs
            # with natom x (nei_type_i x 4)  
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      4],
                                 [-1, self.sel_a[type_i]* 4] )
            start_index += self.sel_a[type_i]
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei_type_i) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4])
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
            for ii in range(1, len(outputs_size)):
              w = tf.get_variable('matrix_'+str(ii)+'_'+str(type_i), 
                                [outputs_size[ii - 1], outputs_size[ii]], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed))
              b = tf.get_variable('bias_'+str(ii)+'_'+str(type_i), 
                                [1, outputs_size[ii]], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
              if self.filter_resnet_dt :
                  idt = tf.get_variable('idt_'+str(ii)+'_'+str(type_i), 
                                        [1, outputs_size[ii]], 
                                        global_tf_float_precision,
                                        tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed))
              if outputs_size[ii] == outputs_size[ii-1]:
                  if self.filter_resnet_dt :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
              elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                  if self.filter_resnet_dt :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
              else:
                  xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
            # natom x nei_type_i x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))
            xyz_scatter_total.append(xyz_scatter)

          # natom x nei x outputs_size
          xyz_scatter = tf.concat(xyz_scatter_total, axis=1)
          # natom x nei x 4
          inputs_reshape = tf.reshape(inputs, [-1, shape[1]//4, 4])
          # natom x 4 x outputs_size
          xyz_scatter_1 = tf.matmul(inputs_reshape, xyz_scatter, transpose_a = True)
          xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape[1])
          # natom x 4 x outputs_size_2
          xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
          # natom x outputs_size x outputs_size_2
          result = tf.matmul(xyz_scatter_1, xyz_scatter_2, transpose_a = True)
          # natom x (outputs_size x outputs_size_2)
          result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])

        return result

    def _DS_layer_type_ext(self, 
                           inputs, 
                           natoms,
                           activation_fn=tf.nn.tanh, 
                           stddev=1.0,
                           bavg=0.0,
                           name='linear', 
                           reuse=None,
                           seed=None):
        # natom x (nei x 4)
        shape = inputs.get_shape().as_list()
        outputs_size = [1] + self.filter_neuron
        outputs_size_2 = self.n_axis_neuron
        with tf.variable_scope(name, reuse=reuse):
          start_index = 0
          result_all = []
          xyz_scatter_1_all = []
          xyz_scatter_2_all = []
          for type_i in range(self.ntypes):
            # cut-out inputs
            # with natom x (nei_type_i x 4)  
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      4],
                                 [-1, self.sel_a[type_i]* 4] )
            start_index += self.sel_a[type_i]
            shape_i = inputs_i.get_shape().as_list()
            # with (natom x nei_type_i) x 4  
            inputs_reshape = tf.reshape(inputs_i, [-1, 4])
            xyz_scatter = tf.reshape(tf.slice(inputs_reshape, [0,0],[-1,1]),[-1,1])
            for ii in range(1, len(outputs_size)):
              w = tf.get_variable('matrix_'+str(ii)+'_'+str(type_i), 
                                [outputs_size[ii - 1], outputs_size[ii]], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed))
              b = tf.get_variable('bias_'+str(ii)+'_'+str(type_i), 
                                [1, outputs_size[ii]], 
                                global_tf_float_precision,
                                tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
              if self.filter_resnet_dt :
                  idt = tf.get_variable('idt_'+str(ii)+'_'+str(type_i), 
                                        [1, outputs_size[ii]], 
                                        global_tf_float_precision,
                                        tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed))
              if outputs_size[ii] == outputs_size[ii-1]:
                  if self.filter_resnet_dt :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter += activation_fn(tf.matmul(xyz_scatter, w) + b)
              elif outputs_size[ii] == outputs_size[ii-1] * 2: 
                  if self.filter_resnet_dt :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b) * idt
                  else :
                      xyz_scatter = tf.concat([xyz_scatter,xyz_scatter], 1) + activation_fn(tf.matmul(xyz_scatter, w) + b)
              else:
                  xyz_scatter = activation_fn(tf.matmul(xyz_scatter, w) + b)
            # natom x nei_type_i x out_size
            xyz_scatter = tf.reshape(xyz_scatter, (-1, shape_i[1]//4, outputs_size[-1]))
            # natom x nei_type_i x 4  
            inputs_i_reshape = tf.reshape(inputs_i, [-1, shape_i[1]//4, 4])
            # natom x 4 x outputs_size
            xyz_scatter_1 = tf.matmul(inputs_i_reshape, xyz_scatter, transpose_a = True)
            xyz_scatter_1 = xyz_scatter_1 * (4.0 / shape_i[1])
            # natom x 4 x outputs_size_2
            xyz_scatter_2 = tf.slice(xyz_scatter_1, [0,0,0],[-1,-1,outputs_size_2])
            xyz_scatter_1_all.append(xyz_scatter_1)
            xyz_scatter_2_all.append(xyz_scatter_2)

          # for type_i in range(self.ntypes):
          #   for type_j in range(type_i, self.ntypes):
          #     # natom x outputs_size x outputs_size_2
          #     result = tf.matmul(xyz_scatter_1_all[type_i], xyz_scatter_2_all[type_j], transpose_a = True)
          #     # natom x (outputs_size x outputs_size_2)
          #     result = tf.reshape(result, [-1, outputs_size_2 * outputs_size[-1]])
          #     result_all.append(tf.identity(result))
          xyz_scatter_2_coll = tf.concat(xyz_scatter_2_all, axis = 2)
          for type_i in range(self.ntypes) :
              # natom x outputs_size x (outputs_size_2 x ntypes)
              result = tf.matmul(xyz_scatter_1_all[type_i], xyz_scatter_2_coll, transpose_a = True)
              # natom x (outputs_size x outputs_size_2 x ntypes)
              result = tf.reshape(result, [-1, outputs_size_2 * self.ntypes * outputs_size[-1]])
              result_all.append(tf.identity(result))              

          # natom x (ntypes x outputs_size x outputs_size_2 x ntypes)
          result_all = tf.concat(result_all, axis = 1)

        return result_all
