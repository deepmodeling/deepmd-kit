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
from deepmd.descriptor import DescrptSeT
from deepmd.descriptor import DescrptSeAEbd
from deepmd.descriptor import DescrptSeAEf
from deepmd.descriptor import DescrptSeR
from deepmd.descriptor import DescrptSeAR
from deepmd.descriptor import DescrptHybrid
from deepmd.model import EnerModel, WFCModel, DipoleModel, PolarModel, GlobalPolarModel
from deepmd.loss import EnerStdLoss, EnerDipoleLoss, TensorLoss
from deepmd.utils.learning_rate import LearningRateExp
from deepmd.utils.neighbor_stat import NeighborStat
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
        self.fitting_param = fitting_param
        
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
        loss_param = jdata['loss']
        
        # fitting net (correlated to loss, because it have a type)
        fitting_list = []
        self.loss_list = []
        self.lr_list = []
        self.model_list = []
        sub_nets = fitting_param.keys()
        for net in sub_nets:
            sub_net = fitting_param[net]
            try: 
                sub_net_type = sub_net['type']
            except:
                sub_net_type = 'ener'
                
            sub_net.pop('type', None)
            sub_net['descrpt'] = self.descrpt
            if sub_net_type == 'ener':
                fitting_list.append(EnerFitting(
                    descrpt = self.descrpt,
                    type_map = model_param.get('type_map'),
                    neuron = sub_net["neuron"],
                    resnet_dt = sub_net["resnet_dt"],
                    seed = sub_net["seed"],
                    name = str(net)
                ))
                
            elif sub_net_type == 'dipole':
                if descrpt_type == 'se_e2_a':
                    fitting_list.append( DipoleFittingSeA(**sub_net))
                else :
                    raise RuntimeError('fitting dipole only supports descrptors: se_e2_a')
            elif sub_net_type == 'polar':
                if descrpt_type == 'se_e2_a':
                    fitting_list.append( PolarFittingSeA(**sub_net))
                else :
                    raise RuntimeError('fitting polar only supports descrptors: loc_frame and se_e2_a')
            elif sub_net_type == 'global_polar':
                if descrpt_type == 'se_e2_a':
                    fitting_list.append( GlobalPolarFittingSeA(**sub_net))
                else :
                    raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_e2_a')
            else :
                raise RuntimeError('unknow fitting type ' + sub_net_type)

        # init model
        # infer model type by fitting_type
            tmp_model = EnerModel(
                self.descrpt, 
                fitting_list[-1], 
                self.typeebd,
                model_param.get('type_map'),
                model_param.get('data_stat_nbatch', 10),
                model_param.get('data_stat_protect', 1e-2),
                model_param.get('use_srtab'),
                model_param.get('smin_alpha'),
                model_param.get('sw_rmin'),
                model_param.get('sw_rmax')
            )
            self.model_list.append(tmp_model)

            # learning rate
            sub_lr = lr_param[net]
            try: 
                sub_lr_type = sub_lr['type']
            except:
                sub_lr_type = 'exp'
            if sub_lr_type == 'exp':
                tmp_lr = LearningRateExp(sub_lr['start_lr'],
                                      sub_lr['stop_lr'],
                                      sub_lr['decay_steps'],
                                      name = net)
                self.lr_list.append(tmp_lr)
            else :
                raise RuntimeError('unknown learning_rate type ' + sub_lr_type)        

            # loss
            # infer loss type by fitting_type
            sub_loss = loss_param[net]
            try :
                loss_type = sub_loss.get('type', 'ener')
            except:
                loss_type = 'ener'

            if sub_net_type == 'ener':
                sub_loss.pop('type', None)
                sub_loss['starter_learning_rate'] = tmp_lr.start_lr()
                if loss_type == 'ener':
                    self.loss_list.append(EnerStdLoss(**sub_loss,name = net))
                elif loss_type == 'ener_dipole':
                    self.loss_list.append(EnerDipoleLoss(**sub_loss)) 
                else:
                    raise RuntimeError('unknow loss type')
            elif sub_net_type == 'wfc':
                self.loss_list.append(TensorLoss(sub_loss, 
                                   model = tmp_model, 
                                   tensor_name = 'wfc',
                                   tensor_size = tmp_model.get_out_size(),
                                   label_name = 'wfc'))
            elif sub_net_type == 'dipole':
                self.loss_list.append(TensorLoss(sub_loss, 
                                   model = tmp_model, 
                                   tensor_name = 'dipole',
                                   tensor_size = 3,
                                   label_name = 'dipole'))
            elif sub_net_type == 'polar':
                self.loss_list.append(TensorLoss(sub_loss, 
                                   model = tmp_model, 
                                   tensor_name = 'polar',
                                   tensor_size = 9,
                                   label_name = 'polarizability'))
            elif sub_net_type == 'global_polar':
                self.loss_list.append(TensorLoss(sub_loss, 
                                   model = tmp_model, 
                                   tensor_name = 'global_polar',
                                   tensor_size = 9,
                                   atomic = False,
                                   label_name = 'polarizability'))
            else :
                raise RuntimeError('get unknown fitting type when building loss function')

        
        self.l2_l_list = {}
        self.l2_more_list = {}
        for sub_loss in self.loss_list:
            self.l2_l_list[sub_loss.get_name()] = None
            self.l2_more_list[sub_loss.get_name()] = None

        # training
        fitting_type = 'ener'
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
        if fitting_type == 'ener' and  fitting_list[0].get_numb_fparam() > 0 :
            self.numb_fparam = fitting_list[0].get_numb_fparam()
        else :
            self.numb_fparam = 0

        if tr_data.get("validation_data", None) is not None:
            self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
        else:
            self.valid_numb_batch = 1


    def build (self, 
               data, 
               stop_batch = 0) :
        # datadocker
        self.ntypes = 0
        for sub_model in self.model_list:
            self.ntypes += sub_model.get_ntypes() # total type number
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
        for i in range(len(self.model_list)):
            sub_model = self.model_list[i]
            sub_data = data.get_data_system_idx(i)
            sub_model.data_stat(sub_data)

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
            self._build_training(data)


    def _build_lr(self):
        self._extra_train_ops   = []
        self.learning_rate_list = []
        self.global_step = tf.train.get_or_create_global_step()
        for tmp_lr in self.lr_list:
            self.learning_rate_list.append(tmp_lr.build(self.global_step, self.stop_batch))
        log.info("built lr")

    def _build_network(self, data):        
        self.place_holders = {}
        data_dict,data_name = data.get_data_dict() 
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
        
        for i in range(len(self.model_list)):
            sub_model = self.model_list[i]
            tmp_model_pred\
                = sub_model.build (self.place_holders['coord'], 
                                self.place_holders['type'], 
                                self.place_holders['natoms_vec'], 
                                self.place_holders['box'], 
                                self.place_holders['default_mesh'],
                                self.place_holders,
                                suffix = sub_model.get_name(), 
                                reuse = tf.AUTO_REUSE)

        
            sub_loss = self.loss_list[i]
            #if sub_loss.get_name() == data_name:
            tmp_l2_l, tmp_l2_more\
                    = sub_loss.build (self.learning_rate_list[i],
                               self.place_holders['natoms_vec'], 
                               tmp_model_pred,
                               self.place_holders,
                               suffix = sub_loss.get_name())
            self.l2_l_list[data_name] = tmp_l2_l
            self.l2_more_list[data_name] = tmp_l2_more

        log.info("built network")

    def _build_training(self,data):
        data_dict,data_name = data.get_data_dict() 
        trainable_variables = tf.trainable_variables()
        self.optimizer_list = {}
        for i in range(len(self.loss_list)):
            sub_loss = self.loss_list[i]
            optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate_list[i])


            if self.run_opt.is_distrib :
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer,
                    replicas_to_aggregate = self.run_opt.cluster_spec.num_tasks("worker"),
                    total_num_replicas = self.run_opt.cluster_spec.num_tasks("worker"),
                    name = "sync_replicas")
                self.sync_replicas_hook = optimizer.make_session_run_hook(self.run_opt.is_chief)   
            self.optimizer_list[sub_loss.get_name()] = optimizer  
        grads = tf.gradients(self.l2_l_list[data_name], trainable_variables)
        apply_op = self.optimizer_list[data_name].apply_gradients (zip (grads, trainable_variables),
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

    def train (self, train_data, valid_data=None) :

        # if valid_data is None:  # no validation set specified.
        #     valid_data = train_data  # using training set as validation set.

        stop_batch = self.stop_batch
        if self.run_opt.is_distrib :
            self._init_sess_distrib()
        else :
            self._init_sess_serial()

        # self.print_head()
        fp = None
        if self.run_opt.is_chief :
            fp = open(self.disp_file, "a")

        cur_batch = self.sess.run(self.global_step)
        is_first_step = True
        self.cur_batch = cur_batch
        for i in range(len(self.lr_list)):
            tmp_lr = self.lr_list[i]
            log.info("system %s, start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e" % 
                 (tmp_lr.get_name(),
                  self.sess.run(self.learning_rate_list[i]),
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
        data_method_list = train_data.get_name()
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
        current_lr = self.sess.run(self.learning_rate_list[method])
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
            sub_loss = self.loss_list[method]
            results = sub_loss.eval(self.sess, feed_dict, natoms)

            for k, v in results.items():
                if k == "natoms":
                    sum_natoms += v
                else:
                    sum_results[k] = sum_results.get(k, 0.) + v * results["natoms"]
        avg_results = {k: v / sum_natoms for k, v in sum_results.items() if not k == "natoms"}
        return avg_results
