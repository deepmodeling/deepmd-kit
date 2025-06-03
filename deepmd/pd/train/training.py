# SPDX-License-Identifier: LGPL-3.0-or-later
import functools
import logging
import time
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import (
    fleet,
)
from paddle.framework import (
    core,
)
from paddle.io import (
    DataLoader,
)

from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)
from deepmd.pd.loss import (
    EnergyHessianStdLoss,
    EnergyStdLoss,
    TaskLoss,
)
from deepmd.pd.model.model import (
    get_model,
)
from deepmd.pd.train.wrapper import (
    ModelWrapper,
)
from deepmd.pd.utils import (
    dp_random,
)
from deepmd.pd.utils.dataloader import (
    BufferedIterator,
    get_sampler_from_params,
)
from deepmd.pd.utils.env import (
    CINN,
    DEFAULT_PRECISION,
    DEVICE,
    JIT,
    NUM_WORKERS,
    SAMPLER_RECORD,
    enable_prim,
)
from deepmd.pd.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.pd.utils.stat import (
    make_stat_input,
)
from deepmd.pd.utils.utils import (
    nvprof_context,
    to_numpy_array,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.path import (
    DPH5Path,
)

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        training_data,
        stat_file_path=None,
        validation_data=None,
        init_model=None,
        restart_model=None,
        finetune_model=None,
        force_load=False,
        shared_links=None,
        finetune_links=None,
        init_frz_model=None,
    ) -> None:
        """Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        """
        enable_prim(True)
        if init_model is not None:
            resume_model = init_model
        elif restart_model is not None:
            resume_model = restart_model
        elif finetune_model is not None:
            resume_model = finetune_model
        else:
            resume_model = None
        resuming = resume_model is not None
        self.restart_training = restart_model is not None
        model_params = config["model"]
        training_params = config["training"]
        self.multi_task = "model_dict" in model_params
        self.finetune_links = finetune_links
        self.finetune_update_stat = False
        self.model_keys = (
            list(model_params["model_dict"]) if self.multi_task else ["Default"]
        )
        self.rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self.num_model = len(self.model_keys)

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.ckpt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.max_ckpt_keep = training_params.get("max_ckpt_keep", 5)
        self.display_in_training = training_params.get("disp_training", True)
        self.timing_in_training = training_params.get("time_training", True)
        self.change_bias_after_training = training_params.get(
            "change_bias_after_training", False
        )
        self.lcurve_should_print_header = True

        def get_opt_param(params):
            opt_type = params.get("opt_type", "Adam")
            opt_param = {
                "kf_blocksize": params.get("kf_blocksize", 5120),
                "kf_start_pref_e": params.get("kf_start_pref_e", 1),
                "kf_limit_pref_e": params.get("kf_limit_pref_e", 1),
                "kf_start_pref_f": params.get("kf_start_pref_f", 1),
                "kf_limit_pref_f": params.get("kf_limit_pref_f", 1),
            }
            return opt_type, opt_param

        def get_data_loader(_training_data, _validation_data, _training_params):
            def get_dataloader_and_buffer(_data, _params):
                _sampler = get_sampler_from_params(_data, _params)
                if _sampler is None:
                    log.warning(
                        "Sampler not specified!"
                    )  # None sampler will lead to a premature stop iteration. Replacement should be True in attribute of the sampler to produce expected number of items in one iteration.
                _dataloader = DataLoader(
                    _data,
                    batch_sampler=paddle.io.BatchSampler(
                        sampler=_sampler,
                        drop_last=False,
                    ),
                    num_workers=NUM_WORKERS
                    if dist.is_available()
                    else 0,  # setting to 0 diverges the behavior of its iterator; should be >=1
                    collate_fn=lambda batch: batch[0],  # prevent extra conversion
                )
                _data_buffered = BufferedIterator(iter(_dataloader))
                return _dataloader, _data_buffered

            training_dataloader, training_data_buffered = get_dataloader_and_buffer(
                _training_data, _training_params["training_data"]
            )

            if _validation_data is not None:
                (
                    validation_dataloader,
                    validation_data_buffered,
                ) = get_dataloader_and_buffer(
                    _validation_data, _training_params["validation_data"]
                )
                valid_numb_batch = _training_params["validation_data"].get(
                    "numb_btch", 1
                )
            else:
                validation_dataloader = None
                validation_data_buffered = None
                valid_numb_batch = 1
            return (
                training_dataloader,
                training_data_buffered,
                validation_dataloader,
                validation_data_buffered,
                valid_numb_batch,
            )

        def single_model_stat(
            _model,
            _data_stat_nbatch,
            _training_data,
            _validation_data,
            _stat_file_path,
            _data_requirement,
            finetune_has_new_type=False,
        ):
            _data_requirement += get_additional_data_requirement(_model)
            _training_data.add_data_requirement(_data_requirement)
            if _validation_data is not None:
                _validation_data.add_data_requirement(_data_requirement)

            @functools.lru_cache
            def get_sample():
                sampled = make_stat_input(
                    _training_data.systems,
                    _training_data.dataloaders,
                    _data_stat_nbatch,
                )
                return sampled

            if (not resuming or finetune_has_new_type) and self.rank == 0:
                _model.compute_or_load_stat(
                    sampled_func=get_sample,
                    stat_file_path=_stat_file_path,
                )
                if isinstance(_stat_file_path, DPH5Path):
                    _stat_file_path.root.close()
            return get_sample

        def get_lr(lr_params):
            assert lr_params.get("type", "exp") == "exp", (
                "Only learning rate `exp` is supported!"
            )
            lr_params["stop_steps"] = self.num_steps - self.warmup_steps
            lr_exp = LearningRateExp(**lr_params)
            return lr_exp

        # Optimizer
        if self.multi_task and training_params.get("optim_dict", None) is not None:
            self.optim_dict = training_params.get("optim_dict")
            missing_keys = [
                key for key in self.model_keys if key not in self.optim_dict
            ]
            assert not missing_keys, (
                f"These keys are not in optim_dict: {missing_keys}!"
            )
            self.opt_type = {}
            self.opt_param = {}
            for model_key in self.model_keys:
                self.opt_type[model_key], self.opt_param[model_key] = get_opt_param(
                    self.optim_dict[model_key]
                )
        else:
            self.opt_type, self.opt_param = get_opt_param(training_params)

        # loss_param_tmp for Hessian activation
        loss_param_tmp = None
        if not self.multi_task:
            loss_param_tmp = config["loss"]
        else:
            loss_param_tmp = {
                model_key: config["loss_dict"][model_key]
                for model_key in self.model_keys
            }

        # Model
        self.model = get_model_for_wrapper(
            model_params,
            resuming=resuming,
            _loss_params=loss_param_tmp,
        )

        # Loss
        if not self.multi_task:
            self.loss = get_loss(
                config["loss"],
                config["learning_rate"]["start_lr"],
                len(model_params["type_map"]),
                self.model,
            )
        else:
            self.loss = {}
            for model_key in self.model_keys:
                loss_param = config["loss_dict"][model_key]
                if config.get("learning_rate_dict", None) is not None:
                    lr_param = config["learning_rate_dict"][model_key]["start_lr"]
                else:
                    lr_param = config["learning_rate"]["start_lr"]
                ntypes = len(model_params["model_dict"][model_key]["type_map"])
                self.loss[model_key] = get_loss(
                    loss_param, lr_param, ntypes, self.model[model_key]
                )

        # Data
        if not self.multi_task:
            self.get_sample_func = single_model_stat(
                self.model,
                model_params.get("data_stat_nbatch", 10),
                training_data,
                validation_data,
                stat_file_path,
                self.loss.label_requirement,
                finetune_has_new_type=self.finetune_links["Default"].get_has_new_type()
                if self.finetune_links is not None
                else False,
            )
            (
                self.training_dataloader,
                self.training_data,
                self.validation_dataloader,
                self.validation_data,
                self.valid_numb_batch,
            ) = get_data_loader(training_data, validation_data, training_params)
            training_data.print_summary(
                "training",
                to_numpy_array(self.training_dataloader.batch_sampler.sampler.weights),
            )
            if validation_data is not None:
                validation_data.print_summary(
                    "validation",
                    to_numpy_array(
                        self.validation_dataloader.batch_sampler.sampler.weights
                    ),
                )
        else:
            (
                self.training_dataloader,
                self.training_data,
                self.validation_dataloader,
                self.validation_data,
                self.valid_numb_batch,
                self.get_sample_func,
            ) = {}, {}, {}, {}, {}, {}
            for model_key in self.model_keys:
                self.get_sample_func[model_key] = single_model_stat(
                    self.model[model_key],
                    model_params["model_dict"][model_key].get("data_stat_nbatch", 10),
                    training_data[model_key],
                    validation_data[model_key],
                    stat_file_path[model_key],
                    self.loss[model_key].label_requirement,
                    finetune_has_new_type=self.finetune_links[
                        model_key
                    ].get_has_new_type()
                    if self.finetune_links is not None
                    else False,
                )
                (
                    self.training_dataloader[model_key],
                    self.training_data[model_key],
                    self.validation_dataloader[model_key],
                    self.validation_data[model_key],
                    self.valid_numb_batch[model_key],
                ) = get_data_loader(
                    training_data[model_key],
                    validation_data[model_key],
                    training_params["data_dict"][model_key],
                )

                training_data[model_key].print_summary(
                    f"training in {model_key}",
                    to_numpy_array(
                        self.training_dataloader[
                            model_key
                        ].batch_sampler.sampler.weights
                    ),
                )
                if (
                    validation_data is not None
                    and validation_data[model_key] is not None
                ):
                    validation_data[model_key].print_summary(
                        f"validation in {model_key}",
                        to_numpy_array(
                            self.validation_dataloader[
                                model_key
                            ].batch_sampler.sampler.weights
                        ),
                    )

        # Learning rate
        self.warmup_steps = training_params.get("warmup_steps", 0)
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)
        assert self.num_steps - self.warmup_steps > 0 or self.warmup_steps == 0, (
            "Warm up steps must be less than total training steps!"
        )
        if self.multi_task and config.get("learning_rate_dict", None) is not None:
            self.lr_exp = {}
            for model_key in self.model_keys:
                self.lr_exp[model_key] = get_lr(config["learning_rate_dict"][model_key])
        else:
            self.lr_exp = get_lr(config["learning_rate"])

        # JIT
        if JIT:
            raise NotImplementedError(
                "JIT is not supported yet when training with Paddle"
            )
            self.model = paddle.jit.to_static(self.model)

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # resuming and finetune
        optimizer_state_dict = None
        if resuming:
            log.info(f"Resuming from {resume_model}.")
            state_dict = paddle.load(resume_model)
            if "model" in state_dict:
                optimizer_state_dict = (
                    state_dict["optimizer"] if finetune_model is None else None
                )
                state_dict = state_dict["model"]
            self.start_step = (
                state_dict["_extra_state"]["train_infos"]["step"]
                if self.restart_training
                else 0
            )
            if self.rank == 0:
                if force_load:
                    input_keys = list(state_dict.keys())
                    target_keys = list(self.wrapper.state_dict().keys())
                    missing_keys = [
                        item for item in target_keys if item not in input_keys
                    ]
                    if missing_keys:
                        target_state_dict = self.wrapper.state_dict()
                        slim_keys = []
                        for item in missing_keys:
                            state_dict[item] = target_state_dict[item].clone().detach()
                            new_key = True
                            for slim_key in slim_keys:
                                if slim_key in item:
                                    new_key = False
                                    break
                            if new_key:
                                tmp_keys = ".".join(item.split(".")[:3])
                                slim_keys.append(tmp_keys)
                        slim_keys = [i + ".*" for i in slim_keys]
                        log.warning(
                            f"Force load mode allowed! These keys are not in ckpt and will re-init: {slim_keys}"
                        )
                # update model params in the pretrained model
                if finetune_model is not None:
                    new_state_dict = {}
                    target_state_dict = self.wrapper.state_dict()
                    # pretrained_model
                    pretrained_model = get_model_for_wrapper(
                        state_dict["_extra_state"]["model_params"]
                    )
                    pretrained_model_wrapper = ModelWrapper(pretrained_model)
                    pretrained_model_wrapper.set_state_dict(state_dict)
                    # update type related params
                    for model_key in self.model_keys:
                        finetune_rule_single = self.finetune_links[model_key]
                        _model_key_from = finetune_rule_single.get_model_branch()
                        # skip if updated
                        if (
                            finetune_rule_single.get_finetune_tmap()
                            != pretrained_model_wrapper.model[
                                _model_key_from
                            ].get_type_map()
                        ):
                            model_with_new_type_stat = None
                            if finetune_rule_single.get_has_new_type():
                                self.finetune_update_stat = True
                                model_with_new_type_stat = self.wrapper.model[model_key]
                            pretrained_model_wrapper.model[
                                _model_key_from
                            ].change_type_map(
                                finetune_rule_single.get_finetune_tmap(),
                                model_with_new_type_stat=model_with_new_type_stat,
                            )
                    state_dict = pretrained_model_wrapper.state_dict()

                    def collect_single_finetune_params(
                        _model_key,
                        _finetune_rule_single,
                        _new_state_dict,
                        _origin_state_dict,
                        _random_state_dict,
                    ) -> None:
                        _new_fitting = _finetune_rule_single.get_random_fitting()
                        _model_key_from = _finetune_rule_single.get_model_branch()
                        target_keys = [
                            i
                            for i in _random_state_dict.keys()
                            if i != "_extra_state" and f".{_model_key}." in i
                        ]
                        for item_key in target_keys:
                            if _new_fitting and (".descriptor." not in item_key):
                                # print(f'Keep {item_key} in old model!')
                                _new_state_dict[item_key] = (
                                    _random_state_dict[item_key].clone().detach()
                                )
                            else:
                                new_key = item_key.replace(
                                    f".{_model_key}.", f".{_model_key_from}."
                                )
                                # print(f'Replace {item_key} with {new_key} in pretrained_model!')
                                _new_state_dict[item_key] = (
                                    _origin_state_dict[new_key].clone().detach()
                                )

                    # collect model params from the pretrained model
                    for model_key in self.model_keys:
                        finetune_rule_single = self.finetune_links[model_key]
                        collect_single_finetune_params(
                            model_key,
                            finetune_rule_single,
                            new_state_dict,
                            state_dict,
                            target_state_dict,
                        )
                    state_dict = new_state_dict
                    state_dict["_extra_state"] = self.wrapper.state_dict()[
                        "_extra_state"
                    ]

                self.wrapper.set_state_dict(state_dict)

                # change bias for fine-tuning
                if finetune_model is not None:

                    def single_model_finetune(
                        _model,
                        _finetune_rule_single,
                        _sample_func,
                    ):
                        _model = model_change_out_bias(
                            _model,
                            _sample_func,
                            _bias_adjust_mode="change-by-statistic"
                            if not _finetune_rule_single.get_random_fitting()
                            else "set-by-statistic",
                        )
                        return _model

                    if not self.multi_task:
                        finetune_rule_single = self.finetune_links["Default"]
                        self.model = single_model_finetune(
                            self.model, finetune_rule_single, self.get_sample_func
                        )
                    else:
                        for model_key in self.model_keys:
                            finetune_rule_single = self.finetune_links[model_key]
                            if not finetune_rule_single.get_resuming():
                                log.info(
                                    f"Model branch {model_key} will be fine-tuned. This may take a long time..."
                                )
                                self.model[model_key] = single_model_finetune(
                                    self.model[model_key],
                                    finetune_rule_single,
                                    self.get_sample_func[model_key],
                                )
                            else:
                                log.info(
                                    f"Model branch {model_key} will resume training."
                                )

        if init_frz_model is not None:
            frz_model = paddle.jit.load(init_frz_model)
            self.model.set_state_dict(frz_model.state_dict())

        # Multi-task share params
        if shared_links is not None:
            self.wrapper.share_params(
                shared_links,
                resume=(resuming and not self.finetune_update_stat) or self.rank != 0,
            )

        # TODO add lr warmups for multitask
        # author: iProzd
        def warm_up_linear(step, warmup_steps):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return self.lr_exp.value(step - warmup_steps) / self.lr_exp.start_lr

        # TODO add optimizers for multitask
        # author: iProzd
        if self.opt_type == "Adam":
            self.scheduler = paddle.optimizer.lr.LambdaDecay(
                learning_rate=self.lr_exp.start_lr,
                lr_lambda=lambda step: warm_up_linear(step, self.warmup_steps),
            )
            self.optimizer = paddle.optimizer.Adam(
                learning_rate=self.scheduler, parameters=self.wrapper.parameters()
            )
            if optimizer_state_dict is not None and self.restart_training:
                self.optimizer.set_state_dict(optimizer_state_dict)
                self.scheduler.last_epoch -= 1
        else:
            raise ValueError(f"Not supported optimizer type '{self.opt_type}'")

        # NOTE: to_static + compiler should be before distributed wrapper
        if CINN:
            from paddle import (
                jit,
                static,
            )

            backend = "CINN" if CINN else None
            self.wrapper.forward = jit.to_static(
                backend=backend,
                input_spec=[
                    static.InputSpec([1, -1, 3], "float64", name="coord"),  # coord
                    static.InputSpec([1, -1], "int32", name="atype"),  # atype
                    None,  # spin
                    static.InputSpec([1, 9], "float64", name="box"),  # box
                    static.InputSpec([], "float64", name="cur_lr"),  # cur_lr
                    {
                        "find_box": np.float32(1.0),
                        "find_coord": np.float32(1.0),
                        "find_numb_copy": np.float32(0.0),
                        "numb_copy": static.InputSpec(
                            [1, 1], "int64", name="numb_copy"
                        ),
                        "find_energy": np.float32(1.0),
                        "energy": static.InputSpec([1, 1], "float64", name="energy"),
                        "find_force": np.float32(1.0),
                        "force": static.InputSpec([1, -1, 3], "float64", name="force"),
                        "natoms": static.InputSpec([1, -1], "int32", name="natoms"),
                    },  # label,
                    # None, # task_key
                    # False, # inference_only
                    # False, # do_atomic_virial
                    # None, # fparam
                    # None, # aparam
                ],
                full_graph=True,
            )(self.wrapper.forward)

            log.info(
                "Enable CINN during training, there may be some additional "
                "compilation time in the first traning step."
            )

        if dist.is_available() and dist.is_initialized():
            # DDP will guarantee the model parameters are identical across all processes
            self.wrapper = fleet.distributed_model(
                self.wrapper,
                # find_unused_parameters=True,
            )
            self.optimizer = fleet.distributed_optimizer(self.optimizer)

        # Get model prob for multi-task
        if self.multi_task:
            self.model_prob = np.array([0.0 for key in self.model_keys])
            if training_params.get("model_prob", None) is not None:
                model_prob = training_params["model_prob"]
                for ii, model_key in enumerate(self.model_keys):
                    if model_key in model_prob:
                        self.model_prob[ii] += float(model_prob[model_key])
            else:
                for ii, model_key in enumerate(self.model_keys):
                    self.model_prob[ii] += float(len(self.training_data[model_key]))
            sum_prob = np.sum(self.model_prob)
            assert sum_prob > 0.0, "Sum of model prob must be larger than 0!"
            self.model_prob = self.model_prob / sum_prob

        # Tensorboard
        self.enable_tensorboard = training_params.get("tensorboard", False)
        self.tensorboard_log_dir = training_params.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = training_params.get("tensorboard_freq", 1)
        self.enable_profiler = training_params.get("enable_profiler", False)
        self.profiling = training_params.get("profiling", False)
        self.profiling_file = training_params.get("profiling_file", "timeline.json")

    def run(self) -> None:
        fout = (
            open(
                self.disp_file,
                mode="w" if not self.restart_training else "a",
                buffering=1,
            )
            if self.rank == 0
            else None
        )  # line buffered
        if SAMPLER_RECORD:
            record_file = f"Sample_rank_{self.rank}.txt"
            fout1 = open(record_file, mode="w", buffering=1)
        log.info("Start to train %d steps.", self.num_steps)
        if dist.is_available() and dist.is_initialized():
            log.info(f"Rank: {dist.get_rank()}/{dist.get_world_size()}")
        if self.enable_tensorboard:
            from tensorboardX import (
                SummaryWriter,
            )

            writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        enable_profiling = self.enable_profiler or self.profiling
        if enable_profiling:
            core.nvprof_start()
            core.nvprof_enable_record_event()

        def step(_step_id, task_key="Default") -> None:
            if self.multi_task:
                model_index = dp_random.choice(
                    np.arange(self.num_model, dtype=np.int_),
                    p=self.model_prob,
                )
                task_key = self.model_keys[model_index]
            # Paddle Profiler
            if enable_profiling:
                core.nvprof_nvtx_push(f"Training step {_step_id}")
            if isinstance(self.lr_exp, dict):
                _lr = self.lr_exp[task_key]
            else:
                _lr = self.lr_exp
            cur_lr = _lr.value(_step_id)
            pref_lr = cur_lr
            self.optimizer.clear_grad(set_to_zero=False)

            with nvprof_context(enable_profiling, "Fetching data"):
                input_dict, label_dict, log_dict = self.get_data(
                    is_train=True, task_key=task_key
                )
            if SAMPLER_RECORD:
                print_str = f"Step {_step_id}: sample system{log_dict['sid']}  frame{log_dict['fid']}\n"
                fout1.write(print_str)
                fout1.flush()
            if self.opt_type == "Adam":
                cur_lr = self.scheduler.get_lr()
                if _step_id < self.warmup_steps:
                    pref_lr = _lr.start_lr
                else:
                    pref_lr = cur_lr
                with nvprof_context(enable_profiling, "Forward pass"):
                    model_pred, loss, more_loss = self.wrapper(
                        **input_dict,
                        cur_lr=paddle.full([], pref_lr, DEFAULT_PRECISION),
                        label=label_dict,
                        task_key=task_key,
                    )

                with nvprof_context(enable_profiling, "Backward pass"):
                    loss.backward()

                if self.gradient_max_norm > 0.0:
                    with nvprof_context(enable_profiling, "Gradient clip"):
                        paddle.nn.utils.clip_grad_norm_(
                            self.wrapper.parameters(),
                            self.gradient_max_norm,
                            error_if_nonfinite=True,
                        )

                with nvprof_context(enable_profiling, "Adam update"):
                    self.optimizer.step()
                self.scheduler.step()

            else:
                raise ValueError(f"Not supported optimizer type '{self.opt_type}'")

            # Log and persist
            display_step_id = _step_id + 1
            if self.display_in_training and (
                display_step_id % self.disp_freq == 0 or display_step_id == 1
            ):
                self.wrapper.eval()  # Will set to train mode before fininshing validation

                def log_loss_train(_loss, _more_loss, _task_key="Default"):
                    results = {}
                    rmse_val = {
                        item: _more_loss[item]
                        for item in _more_loss
                        if "l2_" not in item
                    }
                    for item in sorted(rmse_val.keys()):
                        results[item] = rmse_val[item]
                    return results

                def log_loss_valid(_task_key="Default"):
                    single_results = {}
                    sum_natoms = 0
                    if not self.multi_task:
                        valid_numb_batch = self.valid_numb_batch
                    else:
                        valid_numb_batch = self.valid_numb_batch[_task_key]
                    for ii in range(valid_numb_batch):
                        self.optimizer.clear_grad()
                        input_dict, label_dict, _ = self.get_data(
                            is_train=False, task_key=_task_key
                        )
                        if input_dict == {}:
                            # no validation data
                            return {}
                        _, loss, more_loss = self.wrapper(
                            **input_dict,
                            cur_lr=paddle.full([], pref_lr, DEFAULT_PRECISION),
                            label=label_dict,
                            task_key=_task_key,
                        )
                        # more_loss.update({"rmse": math.sqrt(loss)})
                        natoms = int(input_dict["atype"].shape[-1])
                        sum_natoms += natoms
                        for k, v in more_loss.items():
                            if "l2_" not in k:
                                single_results[k] = (
                                    single_results.get(k, 0.0) + v * natoms
                                )
                    results = {k: v / sum_natoms for k, v in single_results.items()}
                    return results

                if not self.multi_task:
                    train_results = log_loss_train(loss, more_loss)
                    valid_results = log_loss_valid()
                    if self.rank == 0:
                        log.info(
                            format_training_message_per_task(
                                batch=display_step_id,
                                task_name="trn",
                                rmse=train_results,
                                learning_rate=cur_lr,
                            )
                        )
                        if valid_results:
                            log.info(
                                format_training_message_per_task(
                                    batch=display_step_id,
                                    task_name="val",
                                    rmse=valid_results,
                                    learning_rate=None,
                                )
                            )
                else:
                    train_results = {_key: {} for _key in self.model_keys}
                    valid_results = {_key: {} for _key in self.model_keys}
                    train_results[task_key] = log_loss_train(
                        loss, more_loss, _task_key=task_key
                    )
                    for _key in self.model_keys:
                        if _key != task_key:
                            self.optimizer.clear_grad()
                            input_dict, label_dict, _ = self.get_data(
                                is_train=True, task_key=_key
                            )
                            _, loss, more_loss = self.wrapper(
                                **input_dict,
                                cur_lr=paddle.full([], pref_lr, DEFAULT_PRECISION),
                                label=label_dict,
                                task_key=_key,
                            )
                            train_results[_key] = log_loss_train(
                                loss, more_loss, _task_key=_key
                            )
                        valid_results[_key] = log_loss_valid(_task_key=_key)
                        if self.rank == 0:
                            log.info(
                                format_training_message_per_task(
                                    batch=display_step_id,
                                    task_name=_key + "_trn",
                                    rmse=train_results[_key],
                                    learning_rate=cur_lr,
                                )
                            )
                            if valid_results[_key]:
                                log.info(
                                    format_training_message_per_task(
                                        batch=display_step_id,
                                        task_name=_key + "_val",
                                        rmse=valid_results[_key],
                                        learning_rate=None,
                                    )
                                )
                self.wrapper.train()

                current_time = time.time()
                train_time = current_time - self.t0
                self.t0 = current_time
                if self.rank == 0 and self.timing_in_training:
                    eta = int(
                        (self.num_steps - display_step_id) / self.disp_freq * train_time
                    )
                    log.info(
                        format_training_message(
                            batch=display_step_id,
                            wall_time=train_time,
                            eta=eta,
                        )
                    )
                # the first training time is not accurate
                if (
                    (_step_id + 1 - self.start_step) > self.disp_freq
                    or self.num_steps - self.start_step < 2 * self.disp_freq
                ):
                    self.total_train_time += train_time

                if fout:
                    if self.lcurve_should_print_header:
                        self.print_header(fout, train_results, valid_results)
                        self.lcurve_should_print_header = False
                    self.print_on_training(
                        fout, display_step_id, cur_lr, train_results, valid_results
                    )

            if (
                ((_step_id + 1) % self.save_freq == 0 and _step_id != self.start_step)
                or (_step_id + 1) == self.num_steps
            ) and (self.rank == 0 or dist.get_rank() == 0):
                # Handle the case if rank 0 aborted and re-assigned
                self.latest_model = Path(self.save_ckpt + f"-{_step_id + 1}.pd")
                self.save_model(self.latest_model, lr=cur_lr, step=_step_id)
                log.info(f"Saved model to {self.latest_model}")
                symlink_prefix_files(self.latest_model.stem, self.save_ckpt)
                with open("checkpoint", "w") as f:
                    f.write(str(self.latest_model))

            # tensorboard
            if self.enable_tensorboard and (
                display_step_id % self.tensorboard_freq == 0 or display_step_id == 1
            ):
                writer.add_scalar(f"{task_key}/lr", cur_lr, display_step_id)
                writer.add_scalar(f"{task_key}/loss", loss.item(), display_step_id)
                for item in more_loss:
                    writer.add_scalar(
                        f"{task_key}/{item}", more_loss[item].item(), display_step_id
                    )

            if enable_profiling:
                core.nvprof_nvtx_pop()

        self.wrapper.train()
        self.t0 = time.time()
        self.total_train_time = 0.0
        for step_id in range(self.start_step, self.num_steps):
            step(step_id)
            if JIT:
                break

        if self.change_bias_after_training and (self.rank == 0 or dist.get_rank() == 0):
            if not self.multi_task:
                self.model = model_change_out_bias(
                    self.model,
                    self.get_sample_func,
                    _bias_adjust_mode="change-by-statistic",
                )
            else:
                for model_key in self.model_keys:
                    self.model[model_key] = model_change_out_bias(
                        self.model[model_key],
                        self.get_sample_func[model_key],
                        _bias_adjust_mode="change-by-statistic",
                    )
            self.latest_model = Path(self.save_ckpt + f"-{self.num_steps}.pd")
            cur_lr = self.lr_exp.value(self.num_steps - 1)
            self.save_model(self.latest_model, lr=cur_lr, step=self.num_steps - 1)
            log.info(f"Saved model to {self.latest_model}")
            symlink_prefix_files(self.latest_model.stem, self.save_ckpt)
            with open("checkpoint", "w") as f:
                f.write(str(self.latest_model))

        if (
            self.rank == 0 or dist.get_rank() == 0
        ):  # Handle the case if rank 0 aborted and re-assigned
            if self.num_steps == 0:
                # when num_steps is 0, the checkpoint is never not saved
                self.latest_model = Path(self.save_ckpt + "-0.pd")
                self.save_model(self.latest_model, lr=0, step=0)
                log.info(f"Saved model to {self.latest_model}")
                symlink_prefix_files(self.latest_model.stem, self.save_ckpt)
                with open("checkpoint", "w") as f:
                    f.write(str(self.latest_model))

            elapsed_batch = self.num_steps - self.start_step
            if self.timing_in_training and elapsed_batch // self.disp_freq > 0:
                if self.start_step >= 2 * self.disp_freq:
                    log.info(
                        "average training time: %.4f s/batch (exclude first %d batches)",
                        self.total_train_time
                        / (
                            elapsed_batch // self.disp_freq * self.disp_freq
                            - self.disp_freq
                        ),
                        self.disp_freq,
                    )
                else:
                    log.info(
                        "average training time: %.4f s/batch",
                        self.total_train_time
                        / (elapsed_batch // self.disp_freq * self.disp_freq),
                    )

            if JIT:
                raise NotImplementedError(
                    "Paddle JIT saving during training is not supported yet."
                )
            log.info(f"Trained model has been saved to: {self.save_ckpt}")

        if fout:
            fout.close()
        if SAMPLER_RECORD:
            fout1.close()
        if self.enable_tensorboard:
            writer.close()
        if enable_profiling:
            core.nvprof_stop()
            log.info(
                "The nsys profiling trace have been saved to *.nsys-rep and *.sqlite "
                "files, which can be viewd in NVIDIA Nsight Systems software"
            )

    def save_model(self, save_path, lr=0.0, step=0) -> None:
        module = (
            self.wrapper._layers
            if dist.is_available() and dist.is_initialized()
            else self.wrapper
        )
        module.train_infos["lr"] = float(lr)
        module.train_infos["step"] = step
        paddle.save(
            {"model": module.state_dict(), "optimizer": self.optimizer.state_dict()},
            str(save_path),
        )
        checkpoint_dir = save_path.parent
        checkpoint_files = [
            f
            for f in checkpoint_dir.glob("*.pd")
            if not f.is_symlink() and f.name.startswith(self.save_ckpt)
        ]
        if len(checkpoint_files) > self.max_ckpt_keep:
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime)
            checkpoint_files[0].unlink()

    def get_data(self, is_train=True, task_key="Default"):
        if not self.multi_task:
            if is_train:
                try:
                    batch_data = next(iter(self.training_data))
                except StopIteration:
                    # Refresh the status of the dataloader to start from a new epoch
                    self.training_data = BufferedIterator(
                        iter(self.training_dataloader)
                    )
                    batch_data = next(iter(self.training_data))
            else:
                if self.validation_data is None:
                    return {}, {}, {}
                try:
                    batch_data = next(iter(self.validation_data))
                except StopIteration:
                    self.validation_data = BufferedIterator(
                        iter(self.validation_dataloader)
                    )
                    batch_data = next(iter(self.validation_data))
        else:
            if is_train:
                try:
                    batch_data = next(iter(self.training_data[task_key]))
                except StopIteration:
                    # Refresh the status of the dataloader to start from a new epoch
                    self.training_data[task_key] = BufferedIterator(
                        iter(self.training_dataloader[task_key])
                    )
                    batch_data = next(iter(self.training_data[task_key]))
            else:
                if self.validation_data[task_key] is None:
                    return {}, {}, {}
                try:
                    batch_data = next(iter(self.validation_data[task_key]))
                except StopIteration:
                    self.validation_data[task_key] = BufferedIterator(
                        iter(self.validation_dataloader[task_key])
                    )
                    batch_data = next(iter(self.validation_data[task_key]))

        for key in batch_data.keys():
            if key == "sid" or key == "fid" or key == "box" or "find_" in key:
                continue
            elif not isinstance(batch_data[key], list):
                if batch_data[key] is not None:
                    batch_data[key] = batch_data[key].to(DEVICE, blocking=False)
            else:
                batch_data[key] = [
                    item.to(DEVICE, blocking=False) for item in batch_data[key]
                ]
        # we may need a better way to classify which are inputs and which are labels
        # now wrapper only supports the following inputs:
        input_keys = [
            "coord",
            "atype",
            "spin",
            "box",
            "fparam",
            "aparam",
        ]
        input_dict = dict.fromkeys(input_keys)
        label_dict = {}
        for item_key in batch_data:
            if item_key in input_keys:
                input_dict[item_key] = batch_data[item_key]
            else:
                if item_key not in ["sid", "fid"]:
                    label_dict[item_key] = batch_data[item_key]
        log_dict = {}
        if "fid" in batch_data:
            log_dict["fid"] = batch_data["fid"]
        log_dict["sid"] = batch_data["sid"]
        return input_dict, label_dict, log_dict

    def print_header(self, fout, train_results, valid_results) -> None:
        train_keys = sorted(train_results.keys())
        print_str = ""
        print_str += "# {:5s}".format("step")
        if not self.multi_task:
            if valid_results:
                prop_fmt = "   %11s %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_val", k + "_trn")
            else:
                prop_fmt = "   %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_trn")
        else:
            for model_key in self.model_keys:
                if valid_results[model_key]:
                    prop_fmt = "   %11s %11s"
                    for k in sorted(train_results[model_key].keys()):
                        print_str += prop_fmt % (
                            k + f"_val_{model_key}",
                            k + f"_trn_{model_key}",
                        )
                else:
                    prop_fmt = "   %11s"
                    for k in sorted(train_results[model_key].keys()):
                        print_str += prop_fmt % (k + f"_trn_{model_key}")
        print_str += "   {:8s}\n".format("lr")
        print_str += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"
        fout.write(print_str)
        fout.flush()

    def print_on_training(
        self, fout, step_id, cur_lr, train_results, valid_results
    ) -> None:
        train_keys = sorted(train_results.keys())
        print_str = ""
        print_str += f"{step_id:7d}"
        if not self.multi_task:
            if valid_results:
                prop_fmt = "   %11.2e %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (valid_results[k], train_results[k])
            else:
                prop_fmt = "   %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (train_results[k])
        else:
            for model_key in self.model_keys:
                if valid_results[model_key]:
                    prop_fmt = "   %11.2e %11.2e"
                    for k in sorted(valid_results[model_key].keys()):
                        print_str += prop_fmt % (
                            valid_results[model_key][k],
                            train_results[model_key][k],
                        )
                else:
                    prop_fmt = "   %11.2e"
                    for k in sorted(train_results[model_key].keys()):
                        print_str += prop_fmt % (train_results[model_key][k])
        print_str += f"   {cur_lr:8.1e}\n"
        fout.write(print_str)
        fout.flush()


def get_additional_data_requirement(_model):
    additional_data_requirement = []
    if _model.get_dim_fparam() > 0:
        fparam_requirement_items = [
            DataRequirementItem(
                "fparam", _model.get_dim_fparam(), atomic=False, must=True
            )
        ]
        additional_data_requirement += fparam_requirement_items
    if _model.get_dim_aparam() > 0:
        aparam_requirement_items = [
            DataRequirementItem(
                "aparam", _model.get_dim_aparam(), atomic=True, must=True
            )
        ]
        additional_data_requirement += aparam_requirement_items
    has_spin = getattr(_model, "has_spin", False)
    if callable(has_spin):
        has_spin = has_spin()
    if has_spin:
        spin_requirement_items = [
            DataRequirementItem("spin", ndof=3, atomic=True, must=True)
        ]
        additional_data_requirement += spin_requirement_items
    return additional_data_requirement


def whether_hessian(loss_params):
    loss_type = loss_params.get("type", "ener")
    return loss_type == "ener" and loss_params.get("start_pref_h", 0.0) > 0.0


def get_loss(loss_params, start_lr, _ntypes, _model):
    loss_type = loss_params.get("type", "ener")
    if whether_hessian(loss_params):
        loss_params["starter_learning_rate"] = start_lr
        return EnergyHessianStdLoss(**loss_params)
    if loss_type == "ener":
        loss_params["starter_learning_rate"] = start_lr
        return EnergyStdLoss(**loss_params)
    else:
        loss_params["starter_learning_rate"] = start_lr
        return TaskLoss.get_class_by_type(loss_type).get_loss(loss_params)


def get_single_model(
    _model_params,
):
    model = get_model(deepcopy(_model_params)).to(DEVICE)
    return model


def get_model_for_wrapper(
    _model_params,
    resuming=False,
    _loss_params=None,
):
    if "model_dict" not in _model_params:
        if _loss_params is not None and whether_hessian(_loss_params):
            _model_params["hessian_mode"] = True
        _model = get_single_model(
            _model_params,
        )
    else:
        _model = {}
        model_keys = list(_model_params["model_dict"])
        do_case_embd, case_embd_index = get_case_embd_config(_model_params)
        for _model_key in model_keys:
            if _loss_params is not None and whether_hessian(_loss_params[_model_key]):
                _model_params["model_dict"][_model_key]["hessian_mode"] = True
            _model[_model_key] = get_single_model(
                _model_params["model_dict"][_model_key],
            )
            if do_case_embd and not resuming:
                # only set case_embd when from scratch multitask training
                _model[_model_key].set_case_embd(case_embd_index[_model_key])
    return _model


def get_case_embd_config(_model_params):
    assert "model_dict" in _model_params, (
        "Only support setting case embedding for multi-task model!"
    )
    model_keys = list(_model_params["model_dict"])
    sorted_model_keys = sorted(model_keys)
    numb_case_embd_list = [
        _model_params["model_dict"][model_key]
        .get("fitting_net", {})
        .get("dim_case_embd", 0)
        for model_key in sorted_model_keys
    ]
    if not all(item == numb_case_embd_list[0] for item in numb_case_embd_list):
        raise ValueError(
            f"All models must have the same dimension of case embedding, while the settings are: {numb_case_embd_list}"
        )
    if numb_case_embd_list[0] == 0:
        return False, {}
    case_embd_index = {
        model_key: idx for idx, model_key in enumerate(sorted_model_keys)
    }
    return True, case_embd_index


def model_change_out_bias(
    _model,
    _sample_func,
    _bias_adjust_mode="change-by-statistic",
):
    old_bias = deepcopy(_model.get_out_bias())
    _model.change_out_bias(
        _sample_func,
        bias_adjust_mode=_bias_adjust_mode,
    )
    new_bias = deepcopy(_model.get_out_bias())

    model_type_map = _model.get_type_map()
    log.info(
        f"Change output bias of {model_type_map!s} "
        f"from {to_numpy_array(old_bias).reshape(-1)!s} "
        f"to {to_numpy_array(new_bias).reshape(-1)!s}."
    )
    return _model
