#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
import shutil
import time
from typing import (
    Dict,
    List,
)

import google.protobuf.message
import numpy as np
from packaging.version import (
    Version,
)
from tensorflow.python.client import (
    timeline,
)

# load grad of force module
import deepmd.tf.op  # noqa: F401
from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)
from deepmd.tf.common import (
    data_requirement,
    get_precision,
    j_must_have,
)
from deepmd.tf.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    TF_VERSION,
    get_tf_session_config,
    tf,
    tfv2,
)
from deepmd.tf.fit.ener import (
    EnerFitting,
)
from deepmd.tf.model import (
    MultiModel,
)
from deepmd.tf.model.model import (
    Model,
)
from deepmd.tf.utils import random as dp_random
from deepmd.tf.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.tf.utils.errors import (
    GraphTooLargeError,
    GraphWithoutTensorError,
)
from deepmd.tf.utils.graph import (
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.tf.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.tf.utils.sess import (
    run_sess,
)

log = logging.getLogger(__name__)

# nvnmd
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)


def _is_subdir(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    if path == directory:
        return False
    relative = os.path.relpath(path, directory) + os.sep
    return not relative.startswith(os.pardir + os.sep)


class DPTrainer:
    def __init__(self, jdata, run_opt, is_compress=False):
        self.run_opt = run_opt
        self._init_param(jdata)
        self.is_compress = is_compress

    def _init_param(self, jdata):
        # model config
        model_param = j_must_have(jdata, "model")
        if "fitting_key" in model_param:
            model_param["type"] = "multi"

        # nvnmd
        self.nvnmd_param = jdata.get("nvnmd", {})
        nvnmd_cfg.init_from_jdata(self.nvnmd_param)
        if nvnmd_cfg.enable:
            nvnmd_cfg.init_from_deepmd_input(model_param)
            nvnmd_cfg.disp_message()
            nvnmd_cfg.save()

        # init model
        self.model = Model(**model_param)
        self.multi_task_mode = isinstance(self.model, MultiModel)
        self.fitting = self.model.get_fitting()

        def get_lr_and_coef(lr_param):
            scale_by_worker = lr_param.get("scale_by_worker", "linear")
            if scale_by_worker == "linear":
                scale_lr_coef = float(self.run_opt.world_size)
            elif scale_by_worker == "sqrt":
                scale_lr_coef = np.sqrt(self.run_opt.world_size).real
            else:
                scale_lr_coef = 1.0
            lr_type = lr_param.get("type", "exp")
            if lr_type == "exp":
                lr = LearningRateExp(
                    lr_param["start_lr"], lr_param["stop_lr"], lr_param["decay_steps"]
                )
            else:
                raise RuntimeError("unknown learning_rate type " + lr_type)
            return lr, scale_lr_coef

        # learning rate
        if not self.multi_task_mode:
            lr_param = j_must_have(jdata, "learning_rate")
            self.lr, self.scale_lr_coef = get_lr_and_coef(lr_param)
        else:
            self.lr_dict = {}
            self.scale_lr_coef_dict = {}
            lr_param_dict = jdata.get("learning_rate_dict", {})
            for fitting_key in self.fitting:
                lr_param = lr_param_dict.get(fitting_key, {})
                (
                    self.lr_dict[fitting_key],
                    self.scale_lr_coef_dict[fitting_key],
                ) = get_lr_and_coef(lr_param)
        # loss
        # infer loss type by fitting_type
        if not self.multi_task_mode:
            loss_param = jdata.get("loss", {})
            self.loss = self.model.get_loss(loss_param, self.lr)
        else:
            loss_param = jdata.get("loss_dict", {})
            self.loss_dict = self.model.get_loss(loss_param, self.lr_dict)

        # training
        tr_data = jdata["training"]
        self.fitting_weight = tr_data.get("fitting_weight", None)
        if self.multi_task_mode:
            self.fitting_key_list = []
            self.fitting_prob = []
            for fitting_key in self.fitting:
                self.fitting_key_list.append(fitting_key)
                # multi-task mode must have self.fitting_weight
                self.fitting_prob.append(self.fitting_weight[fitting_key])
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.display_in_training = tr_data.get("disp_training", True)
        self.timing_in_training = tr_data.get("time_training", True)
        self.profiling = self.run_opt.is_chief and tr_data.get("profiling", False)
        self.profiling_file = tr_data.get("profiling_file", "timeline.json")
        self.enable_profiler = tr_data.get("enable_profiler", False)
        self.tensorboard = self.run_opt.is_chief and tr_data.get("tensorboard", False)
        self.tensorboard_log_dir = tr_data.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = tr_data.get("tensorboard_freq", 1)
        self.mixed_prec = tr_data.get("mixed_precision", None)
        if self.mixed_prec is not None:
            if (
                self.mixed_prec["compute_prec"] not in ("float16", "bfloat16")
                or self.mixed_prec["output_prec"] != "float32"
            ):
                raise RuntimeError(
                    "Unsupported mixed precision option [output_prec, compute_prec]: [{}, {}], "
                    " Supported: [float32, float16/bfloat16], Please set mixed precision option correctly!".format(
                        self.mixed_prec["output_prec"], self.mixed_prec["compute_prec"]
                    )
                )
        # self.sys_probs = tr_data['sys_probs']
        # self.auto_prob_style = tr_data['auto_prob']
        self.useBN = False
        if not self.multi_task_mode:
            self.numb_fparam = self.model.get_numb_fparam()

            if tr_data.get("validation_data", None) is not None:
                self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
            else:
                self.valid_numb_batch = 1
        else:
            self.numb_fparam_dict = self.model.get_numb_fparam()
            self.valid_numb_batch_dict = {}
            data_dict = tr_data.get("data_dict", None)
            for systems in data_dict:
                if data_dict[systems].get("validation_data", None) is not None:
                    self.valid_numb_batch_dict[systems] = data_dict[systems][
                        "validation_data"
                    ].get("numb_btch", 1)
                else:
                    self.valid_numb_batch_dict[systems] = 1

        # if init the graph with the frozen model
        self.frz_model = None
        self.ckpt_meta = None
        self.model_type = None

    def build(self, data=None, stop_batch=0, origin_type_map=None, suffix=""):
        self.ntypes = self.model.get_ntypes()
        self.stop_batch = stop_batch

        if not self.multi_task_mode:
            if not self.is_compress and data.mixed_type:
                assert isinstance(
                    self.fitting, EnerFitting
                ), "Data in mixed_type format must use ener fitting!"

            if self.numb_fparam > 0:
                log.info("training with %d frame parameter(s)" % self.numb_fparam)
            else:
                log.info("training without frame parameter")
        else:
            assert (
                not self.is_compress
            ), "You should not reach here, multi-task input could not be compressed! "
            self.valid_fitting_key = []
            for fitting_key in data:
                self.valid_fitting_key.append(fitting_key)
                if data[fitting_key].mixed_type:
                    assert isinstance(
                        self.fitting[fitting_key], EnerFitting
                    ), "Data for fitting net {} in mixed_type format must use ener fitting!".format(
                        fitting_key
                    )
                if self.numb_fparam_dict[fitting_key] > 0:
                    log.info(
                        "fitting net %s training with %d frame parameter(s)"
                        % (fitting_key, self.numb_fparam_dict[fitting_key])
                    )
                else:
                    log.info(
                        "fitting net %s training without frame parameter" % fitting_key
                    )

        if not self.is_compress:
            # Usually, the type number of the model should be equal to that of the data
            # However, nt_model > nt_data should be allowed, since users may only want to
            # train using a dataset that only have some of elements
            if not self.multi_task_mode:
                single_data = data
            else:
                single_data = data[next(iter(data.keys()))]
            if self.ntypes < single_data.get_ntypes():
                raise ValueError(
                    "The number of types of the training data is %d, but that of the "
                    "model is only %d. The latter must be no less than the former. "
                    "You may need to reset one or both of them. Usually, the former "
                    "is given by `model/type_map` in the training parameter (if set) "
                    "or the maximum number in the training data. The latter is given "
                    "by `model/descriptor/sel` in the training parameter."
                    % (single_data.get_ntypes(), self.ntypes)
                )
            self.type_map = single_data.get_type_map()
            if not self.multi_task_mode:
                self.batch_size = data.get_batch_size()
            else:
                self.batch_size = {}
                for fitting_key in data:
                    self.batch_size[fitting_key] = data[fitting_key].get_batch_size()
            if self.run_opt.init_mode not in (
                "init_from_model",
                "restart",
                "init_from_frz_model",
                "finetune",
            ):
                # self.saver.restore (in self._init_session) will restore avg and std variables, so data_stat is useless
                # init_from_frz_model will restore data_stat variables in `init_variables` method
                log.info("data stating... (this step may take long time)")
                self.model.data_stat(data)

            # config the init_frz_model command
            if self.run_opt.init_mode == "init_from_frz_model":
                self._init_from_frz_model()
            elif self.run_opt.init_mode == "init_model":
                self._init_from_ckpt(self.run_opt.init_model)
            elif self.run_opt.init_mode == "restart":
                self._init_from_ckpt(self.run_opt.restart)
            elif self.run_opt.init_mode == "finetune":
                self._init_from_pretrained_model(
                    data=data, origin_type_map=origin_type_map
                )

            # neighbor_stat is moved to train.py as duplicated
            # TODO: this is a simple fix but we should have a clear
            #       architecture to call neighbor stat
        else:
            self.model.enable_compression()

        if self.is_compress or self.model_type == "compressed_model":
            tf.constant("compressed_model", name="model_type", dtype=tf.string)
        else:
            tf.constant("original_model", name="model_type", dtype=tf.string)

        if self.mixed_prec is not None:
            self.model.enable_mixed_precision(self.mixed_prec)

        self._build_lr()
        self._build_network(data, suffix)
        self._build_training()

    def _build_lr(self):
        self._extra_train_ops = []
        self.global_step = tf.train.get_or_create_global_step()
        if not self.multi_task_mode:
            self.learning_rate = self.lr.build(self.global_step, self.stop_batch)
        else:
            self.learning_rate_dict = {}

            for fitting_key in self.fitting:
                self.learning_rate_dict[fitting_key] = self.lr_dict[fitting_key].build(
                    self.global_step, self.stop_batch
                )

        log.info("built lr")

    def _build_loss(self):
        if self.stop_batch == 0:
            # l2 is not used if stop_batch is zero
            return None, None
        if not self.multi_task_mode:
            l2_l, l2_more = self.loss.build(
                self.learning_rate,
                self.place_holders["natoms_vec"],
                self.model_pred,
                self.place_holders,
                suffix="test",
            )

            if self.mixed_prec is not None:
                l2_l = tf.cast(l2_l, get_precision(self.mixed_prec["output_prec"]))
        else:
            l2_l, l2_more = {}, {}
            for fitting_key in self.fitting:
                lr = self.learning_rate_dict[fitting_key]
                model = self.model_pred[fitting_key]
                loss_dict = self.loss_dict[fitting_key]

                l2_l[fitting_key], l2_more[fitting_key] = loss_dict.build(
                    lr,
                    self.place_holders["natoms_vec"],
                    model,
                    self.place_holders,
                    suffix=fitting_key,
                )

                if self.mixed_prec is not None:
                    l2_l[fitting_key] = tf.cast(
                        l2_l[fitting_key], get_precision(self.mixed_prec["output_prec"])
                    )

        return l2_l, l2_more

    def _build_network(self, data, suffix=""):
        self.place_holders = {}
        if self.is_compress:
            for kk in ["coord", "box"]:
                self.place_holders[kk] = tf.placeholder(
                    GLOBAL_TF_FLOAT_PRECISION, [None], "t_" + kk
                )
            self._get_place_holders(data_requirement)
        else:
            if not self.multi_task_mode:
                self._get_place_holders(data.get_data_dict())
            else:
                self._get_place_holders(data[next(iter(data.keys()))].get_data_dict())

        self.place_holders["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        self.place_holders["natoms_vec"] = tf.placeholder(
            tf.int32, [self.ntypes + 2], name="t_natoms"
        )
        self.place_holders["default_mesh"] = tf.placeholder(
            tf.int32, [None], name="t_mesh"
        )
        self.place_holders["is_training"] = tf.placeholder(tf.bool)
        self.model_pred = self.model.build(
            self.place_holders["coord"],
            self.place_holders["type"],
            self.place_holders["natoms_vec"],
            self.place_holders["box"],
            self.place_holders["default_mesh"],
            self.place_holders,
            frz_model=self.frz_model,
            ckpt_meta=self.ckpt_meta,
            suffix=suffix,
            reuse=False,
        )

        self.l2_l, self.l2_more = self._build_loss()

        log.info("built network")

    def _build_optimizer(self, fitting_key=None):
        if self.run_opt.is_distrib:
            if fitting_key is None:
                if self.scale_lr_coef > 1.0:
                    log.info("Scale learning rate by coef: %f", self.scale_lr_coef)
                    optimizer = tf.train.AdamOptimizer(
                        self.learning_rate * self.scale_lr_coef
                    )
                else:
                    optimizer = tf.train.AdamOptimizer(self.learning_rate)
                optimizer = self.run_opt._HVD.DistributedOptimizer(optimizer)
            else:
                if self.scale_lr_coef_dict[fitting_key] > 1.0:
                    log.info(
                        "Scale learning rate by coef: %f",
                        self.scale_lr_coef_dict[fitting_key],
                    )
                    optimizer = tf.train.AdamOptimizer(
                        self.learning_rate_dict[fitting_key]
                        * self.scale_lr_coef_dict[fitting_key]
                    )
                else:
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate_dict[fitting_key]
                    )
                optimizer = self.run_opt._HVD.DistributedOptimizer(optimizer)
        else:
            if fitting_key is None:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate_dict[fitting_key]
                )

        if self.mixed_prec is not None:
            _TF_VERSION = Version(TF_VERSION)
            if _TF_VERSION < Version("1.14.0"):
                raise RuntimeError(
                    "TensorFlow version %s is not compatible with the mixed precision setting. Please consider upgrading your TF version!"
                    % TF_VERSION
                )
            elif _TF_VERSION < Version("2.4.0"):
                optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                    optimizer
                )
            else:
                optimizer = tf.mixed_precision.enable_mixed_precision_graph_rewrite(
                    optimizer
                )
        return optimizer

    def _build_training(self):
        if self.stop_batch == 0:
            # self.train_op is not used if stop_batch is zero
            self.train_op = None
            return

        trainable_variables = tf.trainable_variables()

        if not self.multi_task_mode:
            optimizer = self._build_optimizer()
            apply_op = optimizer.minimize(
                loss=self.l2_l,
                global_step=self.global_step,
                var_list=trainable_variables,
                name="train_step",
            )
            train_ops = [apply_op, *self._extra_train_ops]
            self.train_op = tf.group(*train_ops)
        else:
            self.train_op = {}
            for fitting_key in self.fitting:
                optimizer = self._build_optimizer(fitting_key=fitting_key)
                apply_op = optimizer.minimize(
                    loss=self.l2_l[fitting_key],
                    global_step=self.global_step,
                    var_list=trainable_variables,
                    name=f"train_step_{fitting_key}",
                )
                train_ops = [apply_op, *self._extra_train_ops]
                self.train_op[fitting_key] = tf.group(*train_ops)
        log.info("built training")

    def _init_session(self):
        config = get_tf_session_config()
        device, idx = self.run_opt.my_device.split(":", 1)
        if device == "gpu":
            config.gpu_options.visible_device_list = idx
        self.sess = tf.Session(config=config)

        # Initializes or restore global variables
        init_op = tf.global_variables_initializer()
        if self.run_opt.is_chief:
            self.saver = tf.train.Saver(save_relative_paths=True)
            if self.run_opt.init_mode == "init_from_scratch":
                log.info("initialize model from scratch")
                run_sess(self.sess, init_op)
                if not self.is_compress:
                    fp = open(self.disp_file, "w")
                    fp.close()
            elif self.run_opt.init_mode == "init_from_model":
                log.info("initialize from model %s" % self.run_opt.init_model)
                run_sess(self.sess, init_op)
                self.saver.restore(self.sess, self.run_opt.init_model)
                run_sess(self.sess, self.global_step.assign(0))
                fp = open(self.disp_file, "w")
                fp.close()
            elif self.run_opt.init_mode == "restart":
                log.info("restart from model %s" % self.run_opt.restart)
                run_sess(self.sess, init_op)
                self.saver.restore(self.sess, self.run_opt.restart)
            elif self.run_opt.init_mode == "init_from_frz_model":
                log.info("initialize training from the frozen model")
                run_sess(self.sess, init_op)
                fp = open(self.disp_file, "w")
                fp.close()
            elif self.run_opt.init_mode == "finetune":
                log.info("initialize training from the frozen pretrained model")
                run_sess(self.sess, init_op)
                fp = open(self.disp_file, "w")
                fp.close()
            else:
                raise RuntimeError("unknown init mode")
        else:
            run_sess(self.sess, init_op)
            self.saver = None

        # Ensure variable consistency among tasks when training starts
        if self.run_opt.is_distrib:
            bcast_op = self.run_opt._HVD.broadcast_global_variables(0)
            if self.run_opt.is_chief:
                log.info("broadcast global variables to other tasks")
            else:
                log.info("receive global variables from task#0")
            run_sess(self.sess, bcast_op)

    def train(self, train_data=None, valid_data=None):
        # if valid_data is None:  # no validation set specified.
        #     valid_data = train_data  # using training set as validation set.

        stop_batch = self.stop_batch
        self._init_session()

        # Before data shard is enabled, only cheif do evaluation and record it
        # self.print_head()
        fp = None
        if self.run_opt.is_chief:
            fp = open(self.disp_file, "a")

        cur_batch = run_sess(self.sess, self.global_step)
        is_first_step = True
        self.cur_batch = cur_batch
        if not self.multi_task_mode:
            log.info(
                "start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e"
                % (
                    run_sess(self.sess, self.learning_rate),
                    self.lr.value(cur_batch),
                    self.lr.decay_steps_,
                    self.lr.decay_rate_,
                    self.lr.value(stop_batch),
                )
            )
        else:
            for fitting_key in self.fitting:
                log.info(
                    "%s: start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e"
                    % (
                        fitting_key,
                        run_sess(self.sess, self.learning_rate_dict[fitting_key]),
                        self.lr_dict[fitting_key].value(cur_batch),
                        self.lr_dict[fitting_key].decay_steps_,
                        self.lr_dict[fitting_key].decay_rate_,
                        self.lr_dict[fitting_key].value(stop_batch),
                    )
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
                pass  # directory does not exist, this is OK
            except Exception as e:
                # general error when removing directory, warn user
                log.exception(
                    f"Could not remove old tensorboard logging directory: "
                    f"{self.tensorboard_log_dir}. Error: {e}"
                )
            else:
                log.debug("Removing old tensorboard log directory.")
            tb_train_writer = tf.summary.FileWriter(
                self.tensorboard_log_dir + "/train", self.sess.graph
            )
            tb_valid_writer = tf.summary.FileWriter(self.tensorboard_log_dir + "/test")
        else:
            tb_train_writer = None
            tb_valid_writer = None
        if self.enable_profiler:
            # https://www.tensorflow.org/guide/profiler
            tfv2.profiler.experimental.start(self.tensorboard_log_dir)

        train_time = 0
        total_train_time = 0.0
        wall_time_tic = time.time()

        next_batch_train_op = None
        next_fitting_key = None
        next_train_batch_list = None
        next_datasetloader = None

        # dataset loader op
        if not self.multi_task_mode:
            datasetloader = DatasetLoader(train_data)
            data_op = datasetloader.build()
        else:
            datasetloader = {}
            data_op = {}
            for fitting_key in self.fitting:
                datasetloader[fitting_key] = DatasetLoader(train_data[fitting_key])
                data_op[fitting_key] = datasetloader[fitting_key].build()

        while cur_batch < stop_batch:
            # first round validation:
            if is_first_step:
                if not self.multi_task_mode:
                    train_batch = train_data.get_batch()
                    batch_train_op = self.train_op
                else:
                    fitting_idx = dp_random.choice(
                        np.arange(len(self.fitting_key_list)),
                        p=np.array(self.fitting_prob),
                    )
                    fitting_key = self.fitting_key_list[fitting_idx]
                    train_batch = train_data[fitting_key].get_batch()
                    batch_train_op = self.train_op[fitting_key]
            else:
                train_batch = next_datasetloader.get_data_dict(next_train_batch_list)
                batch_train_op = next_batch_train_op
                fitting_key = next_fitting_key
            # for next round
            if not self.multi_task_mode:
                next_datasetloader = datasetloader
                next_batch_train_op = self.train_op
                next_train_batch_op = data_op
            else:
                fitting_idx = dp_random.choice(
                    np.arange(len(self.fitting_key_list)), p=np.array(self.fitting_prob)
                )
                next_fitting_key = self.fitting_key_list[fitting_idx]
                next_datasetloader = datasetloader[next_fitting_key]
                next_batch_train_op = self.train_op[fitting_key]
                next_train_batch_op = data_op[fitting_key]

            if self.display_in_training and is_first_step:
                if self.run_opt.is_chief:
                    if not self.multi_task_mode:
                        valid_batches = (
                            [
                                valid_data.get_batch()
                                for ii in range(self.valid_numb_batch)
                            ]
                            if valid_data is not None
                            else None
                        )
                        self.valid_on_the_fly(
                            fp, [train_batch], valid_batches, print_header=True
                        )
                    else:
                        train_batches = {}
                        valid_batches = {}
                        # valid_numb_batch_dict
                        for fitting_key_ii in train_data:
                            # enumerate fitting key as fitting_key_ii
                            train_batches[fitting_key_ii] = [
                                train_data[fitting_key_ii].get_batch()
                            ]
                            valid_batches[fitting_key_ii] = (
                                [
                                    valid_data[fitting_key_ii].get_batch()
                                    for ii in range(
                                        self.valid_numb_batch_dict[fitting_key_ii]
                                    )
                                ]
                                if fitting_key_ii in valid_data
                                else None
                            )
                        self.valid_on_the_fly(
                            fp,
                            train_batches,
                            valid_batches,
                            print_header=True,
                            fitting_key=fitting_key,
                        )
                is_first_step = False

            if self.timing_in_training:
                tic = time.time()
            train_feed_dict = self.get_feed_dict(train_batch, is_training=True)
            # use tensorboard to visualize the training of deepmd-kit
            # it will takes some extra execution time to generate the tensorboard data
            if self.tensorboard and (cur_batch % self.tensorboard_freq == 0):
                summary, _, next_train_batch_list = run_sess(
                    self.sess,
                    [summary_merged_op, batch_train_op, next_train_batch_op],
                    feed_dict=train_feed_dict,
                    options=prf_options,
                    run_metadata=prf_run_metadata,
                )
                tb_train_writer.add_summary(summary, cur_batch)
            else:
                _, next_train_batch_list = run_sess(
                    self.sess,
                    [batch_train_op, next_train_batch_op],
                    feed_dict=train_feed_dict,
                    options=prf_options,
                    run_metadata=prf_run_metadata,
                )
            if self.timing_in_training:
                toc = time.time()
            if self.timing_in_training:
                train_time += toc - tic
            cur_batch = run_sess(self.sess, self.global_step)
            self.cur_batch = cur_batch

            # on-the-fly validation
            if self.display_in_training and (cur_batch % self.disp_freq == 0):
                if self.timing_in_training:
                    tic = time.time()
                if self.run_opt.is_chief:
                    if not self.multi_task_mode:
                        valid_batches = (
                            [
                                valid_data.get_batch()
                                for ii in range(self.valid_numb_batch)
                            ]
                            if valid_data is not None
                            else None
                        )
                        self.valid_on_the_fly(fp, [train_batch], valid_batches)
                    else:
                        train_batches = {}
                        valid_batches = {}
                        for fitting_key_ii in train_data:
                            train_batches[fitting_key_ii] = [
                                train_data[fitting_key_ii].get_batch()
                            ]
                            valid_batches[fitting_key_ii] = (
                                [
                                    valid_data[fitting_key_ii].get_batch()
                                    for ii in range(
                                        self.valid_numb_batch_dict[fitting_key_ii]
                                    )
                                ]
                                if fitting_key_ii in valid_data
                                else None
                            )
                        self.valid_on_the_fly(
                            fp, train_batches, valid_batches, fitting_key=fitting_key
                        )
                if self.timing_in_training:
                    toc = time.time()
                    test_time = toc - tic
                    wall_time = toc - wall_time_tic
                    log.info(
                        format_training_message(
                            batch=cur_batch,
                            wall_time=wall_time,
                        )
                    )
                    # the first training time is not accurate
                    if cur_batch > self.disp_freq or stop_batch < 2 * self.disp_freq:
                        total_train_time += train_time
                    train_time = 0
                    wall_time_tic = toc
                if (
                    self.save_freq > 0
                    and cur_batch % self.save_freq == 0
                    and self.saver is not None
                ):
                    self.save_checkpoint(cur_batch)
        if (
            self.save_freq == 0 or cur_batch == 0 or cur_batch % self.save_freq != 0
        ) and self.saver is not None:
            self.save_checkpoint(cur_batch)
        if self.run_opt.is_chief:
            fp.close()
        if self.timing_in_training and stop_batch // self.disp_freq > 0:
            if stop_batch >= 2 * self.disp_freq:
                log.info(
                    "average training time: %.4f s/batch (exclude first %d batches)",
                    total_train_time
                    / (stop_batch // self.disp_freq * self.disp_freq - self.disp_freq),
                    self.disp_freq,
                )
            else:
                log.info(
                    "average training time: %.4f s/batch",
                    total_train_time / (stop_batch // self.disp_freq * self.disp_freq),
                )

        if self.profiling and self.run_opt.is_chief:
            fetched_timeline = timeline.Timeline(prf_run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self.profiling_file, "w") as f:
                f.write(chrome_trace)
        if self.enable_profiler and self.run_opt.is_chief:
            tfv2.profiler.experimental.stop()

    def save_checkpoint(self, cur_batch: int):
        try:
            ckpt_prefix = self.saver.save(
                self.sess,
                os.path.join(os.getcwd(), self.save_ckpt),
                global_step=cur_batch,
            )
        except google.protobuf.message.DecodeError as e:
            raise GraphTooLargeError(
                "The graph size exceeds 2 GB, the hard limitation of protobuf."
                " Then a DecodeError was raised by protobuf. You should "
                "reduce the size of your model."
            ) from e
        # make symlinks from prefix with step to that without step to break nothing
        # get all checkpoint files
        symlink_prefix_files(ckpt_prefix, self.save_ckpt)
        log.info("saved checkpoint %s" % self.save_ckpt)

    def get_feed_dict(self, batch, is_training):
        feed_dict = {}
        for kk in batch.keys():
            if kk == "find_type" or kk == "type" or kk == "real_natoms_vec":
                continue
            if "find_" in kk:
                feed_dict[self.place_holders[kk]] = batch[kk]
            else:
                feed_dict[self.place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ["type"]:
            feed_dict[self.place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ["natoms_vec", "default_mesh"]:
            feed_dict[self.place_holders[ii]] = batch[ii]
        feed_dict[self.place_holders["is_training"]] = is_training
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

    def valid_on_the_fly(
        self, fp, train_batches, valid_batches, print_header=False, fitting_key=None
    ):
        train_results = self.get_evaluation_results(train_batches)
        valid_results = self.get_evaluation_results(valid_batches)

        cur_batch = self.cur_batch
        if not self.multi_task_mode:
            current_lr = run_sess(self.sess, self.learning_rate)
        else:
            assert (
                fitting_key is not None
            ), "Fitting key must be assigned in validation!"
            current_lr = None
            # current_lr can be used as the learning rate of descriptor in the future
            current_lr_dict = {}
            for fitting_key_ii in train_batches:
                current_lr_dict[fitting_key_ii] = run_sess(
                    self.sess, self.learning_rate_dict[fitting_key_ii]
                )
        if print_header:
            self.print_header(fp, train_results, valid_results, self.multi_task_mode)
        if not self.multi_task_mode:
            self.print_on_training(
                fp,
                train_results,
                valid_results,
                cur_batch,
                current_lr,
                self.multi_task_mode,
            )
        else:
            assert (
                fitting_key is not None
            ), "Fitting key must be assigned when printing learning rate!"
            self.print_on_training(
                fp,
                train_results,
                valid_results,
                cur_batch,
                current_lr,
                self.multi_task_mode,
                current_lr_dict,
            )

    @staticmethod
    def print_header(fp, train_results, valid_results, multi_task_mode=False):
        print_str = ""
        print_str += "# %5s" % "step"
        if not multi_task_mode:
            if valid_results is not None:
                prop_fmt = "   %11s %11s"
                for k in train_results.keys():
                    print_str += prop_fmt % (k + "_val", k + "_trn")
            else:
                prop_fmt = "   %11s"
                for k in train_results.keys():
                    print_str += prop_fmt % (k + "_trn")
            print_str += "   %8s\n" % "lr"
        else:
            for fitting_key in train_results:
                if valid_results[fitting_key] is not None:
                    prop_fmt = "   %11s %11s"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (k + "_val", k + "_trn")
                else:
                    prop_fmt = "   %11s"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (k + "_trn")
                print_str += "   %8s\n" % (fitting_key + "_lr")
        print_str += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_on_training(
        fp,
        train_results,
        valid_results,
        cur_batch,
        cur_lr,
        multi_task_mode=False,
        cur_lr_dict=None,
    ):
        print_str = ""
        print_str += "%7d" % cur_batch
        if not multi_task_mode:
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
            log.info(
                format_training_message_per_task(
                    batch=cur_batch,
                    task_name="trn",
                    rmse=train_results,
                    learning_rate=cur_lr,
                )
            )
            if valid_results is not None:
                log.info(
                    format_training_message_per_task(
                        batch=cur_batch,
                        task_name="val",
                        rmse=valid_results,
                        learning_rate=None,
                    )
                )
        else:
            for fitting_key in train_results:
                if valid_results[fitting_key] is not None:
                    prop_fmt = "   %11.2e %11.2e"
                    for k in valid_results[fitting_key].keys():
                        # assert k in train_results[fitting_key].keys()
                        print_str += prop_fmt % (
                            valid_results[fitting_key][k],
                            train_results[fitting_key][k],
                        )
                else:
                    prop_fmt = "   %11.2e"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (train_results[fitting_key][k])
                print_str += "   %8.1e\n" % cur_lr_dict[fitting_key]
                log.info(
                    format_training_message_per_task(
                        batch=cur_batch,
                        task_name=f"{fitting_key}_trn",
                        rmse=train_results[fitting_key],
                        learning_rate=cur_lr_dict[fitting_key],
                    )
                )
                if valid_results is not None:
                    log.info(
                        format_training_message_per_task(
                            batch=cur_batch,
                            task_name=f"{fitting_key}_val",
                            rmse=valid_results[fitting_key],
                            learning_rate=None,
                        )
                    )
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def eval_single_list(single_batch_list, loss, sess, get_feed_dict_func, prefix=""):
        if single_batch_list is None:
            return None
        numb_batch = len(single_batch_list)
        sum_results = {}  # sum of losses on all atoms
        sum_natoms = 0
        for i in range(numb_batch):
            batch = single_batch_list[i]
            natoms = batch["natoms_vec"]
            feed_dict = get_feed_dict_func(batch, is_training=False)
            results = loss.eval(sess, feed_dict, natoms)

            for k, v in results.items():
                if k == "natoms":
                    sum_natoms += v
                else:
                    sum_results[k] = sum_results.get(k, 0.0) + v * results["natoms"]
        single_results = {
            prefix + k: v / sum_natoms
            for k, v in sum_results.items()
            if not k == "natoms"
        }
        return single_results

    def get_evaluation_results(self, batch_list):
        if not self.multi_task_mode:
            avg_results = self.eval_single_list(
                batch_list, self.loss, self.sess, self.get_feed_dict
            )
        else:
            avg_results = {}
            for fitting_key in batch_list:
                avg_results[fitting_key] = self.eval_single_list(
                    batch_list[fitting_key],
                    self.loss_dict[fitting_key],
                    self.sess,
                    self.get_feed_dict,
                    prefix=f"{fitting_key}_",
                )
        return avg_results

    def save_compressed(self):
        """Save the compressed graph."""
        self._init_session()
        if self.is_compress:
            self.saver.save(self.sess, os.path.join(os.getcwd(), self.save_ckpt))

    def _get_place_holders(self, data_dict):
        for kk in data_dict.keys():
            if kk == "type":
                continue
            prec = GLOBAL_TF_FLOAT_PRECISION
            if data_dict[kk]["high_prec"]:
                prec = GLOBAL_ENER_FLOAT_PRECISION
            self.place_holders[kk] = tf.placeholder(prec, [None], name="t_" + kk)
            self.place_holders["find_" + kk] = tf.placeholder(
                tf.float32, name="t_find_" + kk
            )

    def _init_from_frz_model(self):
        try:
            graph, graph_def = load_graph_def(self.run_opt.init_frz_model)
        except FileNotFoundError as e:
            # throw runtime error if there's no frozen model
            raise RuntimeError(
                "The input frozen model {} ({}) does not exist! Please check the path of the frozen model. ".format(
                    self.run_opt.init_frz_model,
                    os.path.abspath(self.run_opt.init_frz_model),
                )
            ) from e
        # get the model type from the frozen model(self.run_opt.init_frz_model)
        try:
            t_model_type = get_tensor_by_name_from_graph(graph, "model_type")
        except GraphWithoutTensorError as e:
            # throw runtime error if the frozen_model has no model type information...
            raise RuntimeError(
                "The input frozen model: %s has no 'model_type' information, "
                "which is not supported by the 'dp train init-frz-model' interface. "
                % self.run_opt.init_frz_model
            ) from e
        else:
            self.model_type = bytes.decode(t_model_type)
        if self.model_type == "compressed_model":
            self.frz_model = self.run_opt.init_frz_model
        self.model.init_variables(graph, graph_def, model_type=self.model_type)

    def _init_from_ckpt(self, ckpt_meta: str):
        with tf.Graph().as_default() as graph:
            tf.train.import_meta_graph(f"{ckpt_meta}.meta", clear_devices=True)
        # get the model type from the model
        try:
            t_model_type = get_tensor_by_name_from_graph(graph, "model_type")
        except GraphWithoutTensorError as e:
            self.model_type = "original_model"
        else:
            self.model_type = bytes.decode(t_model_type)
        if self.model_type == "compressed_model":
            self.ckpt_meta = ckpt_meta

    def _init_from_pretrained_model(
        self, data, origin_type_map=None, bias_shift="delta"
    ):
        """Init the embedding net variables with the given frozen model.

        Parameters
        ----------
        data : DeepmdDataSystem
            The training data.
        origin_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        """
        try:
            graph, graph_def = load_graph_def(self.run_opt.finetune)
        except FileNotFoundError as e:
            # throw runtime error if there's no frozen model
            raise RuntimeError(
                "The input frozen pretrained model {} ({}) does not exist! "
                "Please check the path of the frozen pretrained model. ".format(
                    self.run_opt.finetune, os.path.abspath(self.run_opt.finetune)
                )
            ) from e
        # get the model type from the frozen model(self.run_opt.finetune)
        try:
            t_model_type = get_tensor_by_name_from_graph(graph, "model_type")
        except GraphWithoutTensorError as e:
            # throw runtime error if the frozen_model has no model type information...
            raise RuntimeError(
                "The input frozen pretrained model: %s has no 'model_type' information, "
                "which is not supported by the 'dp train finetune' interface. "
                % self.run_opt.finetune
            ) from e
        else:
            self.model_type = bytes.decode(t_model_type)
        assert (
            self.model_type != "compressed_model"
        ), "Compressed models are not supported for finetuning!"
        self.model.init_variables(graph, graph_def, model_type=self.model_type)
        log.info(
            "Changing energy bias in pretrained model for types {}... "
            "(this step may take long time)".format(str(origin_type_map))
        )
        self._change_energy_bias(
            data, self.run_opt.finetune, origin_type_map, bias_shift
        )

    def _change_energy_bias(
        self, data, frozen_model, origin_type_map, bias_shift="delta"
    ):
        full_type_map = data.get_type_map()
        self.model.change_energy_bias(
            data,
            frozen_model,
            origin_type_map,
            full_type_map,
            bias_shift=bias_shift,
        )


class DatasetLoader:
    """Generate an OP that loads the training data from the given DeepmdDataSystem.

    It can be used to load the training data in the training process, so there is
    no waiting time between training steps.

    Parameters
    ----------
    train_data : DeepmdDataSystem
        The training data.

    Examples
    --------
    >>> loader = DatasetLoader(train_data)
    >>> data_op = loader.build()
    >>> with tf.Session() as sess:
    >>>     data_list = sess.run(data_op)
    >>> data_dict = loader.get_data_dict(data_list)
    """

    def __init__(self, train_data: DeepmdDataSystem):
        self.train_data = train_data
        # get the keys of the data
        batch_data = self.train_data.get_batch()
        self.data_keys = batch_data.keys()
        self.data_types = [tf.as_dtype(x.dtype) for x in batch_data.values()]

    def build(self) -> List[tf.Tensor]:
        """Build the OP that loads the training data.

        Returns
        -------
        List[tf.Tensor]
            Tensor of the loaded data.
        """
        train_data = self.train_data

        def get_train_batch() -> List[np.ndarray]:
            batch_data = train_data.get_batch()
            # convert dict to list of arryas
            batch_data = tuple([batch_data[kk] for kk in self.data_keys])
            return batch_data

        return tf.py_func(get_train_batch, [], self.data_types, name="train_data")

    def get_data_dict(self, batch_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Generate a dict of the loaded data.

        Parameters
        ----------
        batch_list : List[np.ndarray]
            The loaded data.

        Returns
        -------
        Dict[str, np.ndarray]
            The dict of the loaded data.
        """
        return dict(zip(self.data_keys, batch_list))
