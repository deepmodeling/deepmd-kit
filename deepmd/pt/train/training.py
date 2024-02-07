# SPDX-License-Identifier: LGPL-3.0-or-later
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
    Dict,
)

import numpy as np
import torch

from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.pt.loss import (
    DenoiseLoss,
    EnergyStdLoss,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.optimizer import (
    KFOptimizerWrapper,
    LKFOptimizer,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    dp_random,
)
from deepmd.pt.utils.dataloader import (
    BufferedIterator,
    get_weighted_sampler,
)
from deepmd.pt.utils.env import (
    DEVICE,
    JIT,
    LOCAL_RANK,
    NUM_WORKERS,
    SAMPLER_RECORD,
)
from deepmd.pt.utils.learning_rate import (
    LearningRateExp,
)

if torch.__version__.startswith("2"):
    import torch._dynamo

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import (
    DataLoader,
)

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        config: Dict[str, Any],
        training_data,
        sampled=None,
        validation_data=None,
        init_model=None,
        restart_model=None,
        finetune_model=None,
        force_load=False,
        shared_links=None,
    ):
        """Construct a DeePMD trainer.

        Args:
        - config: The Dict-like configuration with training options.
        """
        resume_model = init_model if init_model is not None else restart_model
        self.restart_training = restart_model is not None
        model_params = config["model"]
        training_params = config["training"]
        self.multi_task = "model_dict" in model_params
        self.finetune_multi_task = model_params.pop(
            "finetune_multi_task", False
        )  # should use pop for next finetune
        self.model_keys = (
            list(model_params["model_dict"]) if self.multi_task else ["Default"]
        )
        if self.multi_task and sampled is None:
            sampled = {key: None for key in self.model_keys}
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.num_model = len(self.model_keys)

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.ckpt")
        self.save_freq = training_params.get("save_freq", 1000)
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
            if "auto_prob" in _training_params["training_data"]:
                train_sampler = get_weighted_sampler(
                    _training_data, _training_params["training_data"]["auto_prob"]
                )
            elif "sys_probs" in _training_params["training_data"]:
                train_sampler = get_weighted_sampler(
                    _training_data,
                    _training_params["training_data"]["sys_probs"],
                    sys_prob=True,
                )
            else:
                train_sampler = get_weighted_sampler(_training_data, "prob_sys_size")

            if "auto_prob" in _training_params["validation_data"]:
                valid_sampler = get_weighted_sampler(
                    _validation_data, _training_params["validation_data"]["auto_prob"]
                )
            elif "sys_probs" in _training_params["validation_data"]:
                valid_sampler = get_weighted_sampler(
                    _validation_data,
                    _training_params["validation_data"]["sys_probs"],
                    sys_prob=True,
                )
            else:
                valid_sampler = get_weighted_sampler(_validation_data, "prob_sys_size")

            if train_sampler is None or valid_sampler is None:
                log.warning(
                    "Sampler not specified!"
                )  # None sampler will lead to a premature stop iteration. Replacement should be True in attribute of the sampler to produce expected number of items in one iteration.
            training_dataloader = DataLoader(
                _training_data,
                sampler=train_sampler,
                batch_size=None,
                num_workers=NUM_WORKERS,  # setting to 0 diverges the behavior of its iterator; should be >=1
                drop_last=False,
                pin_memory=True,
            )
            training_data_buffered = BufferedIterator(iter(training_dataloader))
            validation_dataloader = DataLoader(
                _validation_data,
                sampler=valid_sampler,
                batch_size=None,
                num_workers=min(NUM_WORKERS, 1),
                drop_last=False,
                pin_memory=True,
            )

            validation_data_buffered = BufferedIterator(iter(validation_dataloader))
            if _training_params.get("validation_data", None) is not None:
                valid_numb_batch = _training_params["validation_data"].get(
                    "numb_btch", 1
                )
            else:
                valid_numb_batch = 1
            return (
                training_dataloader,
                training_data_buffered,
                validation_dataloader,
                validation_data_buffered,
                valid_numb_batch,
            )

        def get_single_model(_model_params, _sampled):
            model = get_model(deepcopy(_model_params)).to(DEVICE)
            if not model_params.get("resuming", False):
                model.compute_or_load_stat(
                    type_map=_model_params["type_map"],
                    sampled=_sampled,
                    stat_file_path_dict=model_params.get("stat_file_path", None),
                )
            return model

        def get_lr(lr_params):
            assert (
                lr_params.get("type", "exp") == "exp"
            ), "Only learning rate `exp` is supported!"
            lr_params["stop_steps"] = self.num_steps - self.warmup_steps
            lr_exp = LearningRateExp(**lr_params)
            return lr_exp

        def get_loss(loss_params, start_lr, _ntypes):
            loss_type = loss_params.get("type", "ener")
            if loss_type == "ener":
                loss_params["starter_learning_rate"] = start_lr
                return EnergyStdLoss(**loss_params)
            elif loss_type == "denoise":
                loss_params["ntypes"] = _ntypes
                return DenoiseLoss(**loss_params)
            else:
                raise NotImplementedError

        # Optimizer
        if self.multi_task and training_params.get("optim_dict", None) is not None:
            self.optim_dict = training_params.get("optim_dict")
            missing_keys = [
                key for key in self.model_keys if key not in self.optim_dict
            ]
            assert (
                not missing_keys
            ), f"These keys are not in optim_dict: {missing_keys}!"
            self.opt_type = {}
            self.opt_param = {}
            for model_key in self.model_keys:
                self.opt_type[model_key], self.opt_param[model_key] = get_opt_param(
                    self.optim_dict[model_key]
                )
        else:
            self.opt_type, self.opt_param = get_opt_param(training_params)

        # Data + Model
        dp_random.seed(training_params["seed"])
        if not self.multi_task:
            (
                self.training_dataloader,
                self.training_data,
                self.validation_dataloader,
                self.validation_data,
                self.valid_numb_batch,
            ) = get_data_loader(training_data, validation_data, training_params)
            self.model = get_single_model(model_params, sampled)
        else:
            (
                self.training_dataloader,
                self.training_data,
                self.validation_dataloader,
                self.validation_data,
                self.valid_numb_batch,
                self.model,
            ) = {}, {}, {}, {}, {}, {}
            for model_key in self.model_keys:
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
                self.model[model_key] = get_single_model(
                    model_params["model_dict"][model_key], sampled[model_key]
                )

        # Learning rate
        self.warmup_steps = training_params.get("warmup_steps", 0)
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)
        assert (
            self.num_steps - self.warmup_steps > 0
        ), "Warm up steps must be less than total training steps!"
        if self.multi_task and config.get("learning_rate_dict", None) is not None:
            self.lr_exp = {}
            for model_key in self.model_keys:
                self.lr_exp[model_key] = get_lr(config["learning_rate_dict"][model_key])
        else:
            self.lr_exp = get_lr(config["learning_rate"])

        # Loss
        if not self.multi_task:
            self.loss = get_loss(
                config["loss"],
                config["learning_rate"]["start_lr"],
                len(model_params["type_map"]),
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
                self.loss[model_key] = get_loss(loss_param, lr_param, ntypes)

        # JIT
        if JIT:
            self.model = torch.jit.script(self.model)

        # Model Wrapper
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # resuming and finetune
        optimizer_state_dict = None
        if model_params["resuming"]:
            ntest = model_params.get("data_bias_nsample", 1)
            origin_model = (
                finetune_model if finetune_model is not None else resume_model
            )
            log.info(f"Resuming from {origin_model}.")
            state_dict = torch.load(origin_model, map_location=DEVICE)
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
                elif self.finetune_multi_task:
                    new_state_dict = {}
                    model_branch_chosen = model_params.pop("model_branch_chosen")
                    new_fitting = model_params.pop("new_fitting", False)
                    target_state_dict = self.wrapper.state_dict()
                    target_keys = [
                        i for i in target_state_dict.keys() if i != "_extra_state"
                    ]
                    for item_key in target_keys:
                        if new_fitting and ".fitting_net." in item_key:
                            # print(f'Keep {item_key} in old model!')
                            new_state_dict[item_key] = (
                                target_state_dict[item_key].clone().detach()
                            )
                        else:
                            new_key = item_key.replace(
                                ".Default.", f".{model_branch_chosen}."
                            )
                            # print(f'Replace {item_key} with {new_key} in pretrained_model!')
                            new_state_dict[item_key] = (
                                state_dict[new_key].clone().detach()
                            )
                    state_dict = new_state_dict
                if finetune_model is not None:
                    state_dict["_extra_state"] = self.wrapper.state_dict()[
                        "_extra_state"
                    ]

                self.wrapper.load_state_dict(state_dict)
                # finetune
                if finetune_model is not None and model_params["fitting_net"].get(
                    "type", "ener"
                ) in ["ener", "direct_force_ener", "atten_vec_lcc"]:
                    old_type_map, new_type_map = (
                        model_params["type_map"],
                        model_params["new_type_map"],
                    )
                    self.model.fitting_net.change_energy_bias(
                        config,
                        self.model,
                        old_type_map,
                        new_type_map,
                        ntest=ntest,
                        bias_shift=model_params.get("bias_shift", "delta"),
                    )

        # Set trainable params
        self.wrapper.set_trainable_params()

        # Multi-task share params
        if shared_links is not None:
            self.wrapper.share_params(shared_links, resume=model_params["resuming"])

        if dist.is_initialized():
            torch.cuda.set_device(LOCAL_RANK)
            # DDP will guarantee the model parameters are identical across all processes
            self.wrapper = DDP(
                self.wrapper,
                device_ids=[LOCAL_RANK],
                find_unused_parameters=True,
                output_device=LOCAL_RANK,
            )

        # TODO ZD add lr warmups for multitask
        def warm_up_linear(step, warmup_steps):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return self.lr_exp.value(step - warmup_steps) / self.lr_exp.start_lr

        # TODO ZD add optimizers for multitask
        if self.opt_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.wrapper.parameters(), lr=self.lr_exp.start_lr
            )
            if optimizer_state_dict is not None and self.restart_training:
                self.optimizer.load_state_dict(optimizer_state_dict)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lambda step: warm_up_linear(step + self.start_step, self.warmup_steps),
            )
        elif self.opt_type == "LKF":
            self.optimizer = LKFOptimizer(
                self.wrapper.parameters(), 0.98, 0.99870, self.opt_param["kf_blocksize"]
            )
        else:
            raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

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

    def run(self):
        fout = (
            open(self.disp_file, mode="w", buffering=1) if self.rank == 0 else None
        )  # line buffered
        if SAMPLER_RECORD:
            record_file = f"Sample_rank_{self.rank}.txt"
            fout1 = open(record_file, mode="w", buffering=1)
        log.info("Start to train %d steps.", self.num_steps)
        if dist.is_initialized():
            log.info(f"Rank: {dist.get_rank()}/{dist.get_world_size()}")
        if self.enable_tensorboard:
            from torch.utils.tensorboard import (
                SummaryWriter,
            )

            writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        if self.enable_profiler:
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.tensorboard_log_dir
                ),
                record_shapes=True,
                with_stack=True,
            )
            prof.start()

        def step(_step_id, task_key="Default"):
            # PyTorch Profiler
            if self.enable_profiler:
                prof.step()
            self.wrapper.train()
            if isinstance(self.lr_exp, dict):
                _lr = self.lr_exp[task_key]
            else:
                _lr = self.lr_exp
            cur_lr = _lr.value(_step_id)
            pref_lr = cur_lr
            self.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict, log_dict = self.get_data(
                is_train=True, task_key=task_key
            )
            if SAMPLER_RECORD:
                print_str = f"Step {_step_id}: sample system{log_dict['sid']}  frame{log_dict['fid']}\n"
                fout1.write(print_str)
                fout1.flush()
            if self.opt_type == "Adam":
                cur_lr = self.scheduler.get_last_lr()[0]
                if _step_id < self.warmup_steps:
                    pref_lr = _lr.start_lr
                else:
                    pref_lr = cur_lr
                model_pred, loss, more_loss = self.wrapper(
                    **input_dict, cur_lr=pref_lr, label=label_dict, task_key=task_key
                )
                loss.backward()
                if self.gradient_max_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.wrapper.parameters(), self.gradient_max_norm
                    )
                    if not torch.isfinite(grad_norm).all():
                        # check local gradnorm single GPU case, trigger NanDetector
                        raise FloatingPointError("gradients are Nan/Inf")
                self.optimizer.step()
                self.scheduler.step()
            elif self.opt_type == "LKF":
                if isinstance(self.loss, EnergyStdLoss):
                    KFOptWrapper = KFOptimizerWrapper(
                        self.wrapper, self.optimizer, 24, 6, dist.is_initialized()
                    )
                    pref_e = self.opt_param["kf_start_pref_e"] * (
                        self.opt_param["kf_limit_pref_e"]
                        / self.opt_param["kf_start_pref_e"]
                    ) ** (_step_id / self.num_steps)
                    _ = KFOptWrapper.update_energy(
                        input_dict, label_dict["energy"], pref_e
                    )
                    pref_f = self.opt_param["kf_start_pref_f"] * (
                        self.opt_param["kf_limit_pref_f"]
                        / self.opt_param["kf_start_pref_f"]
                    ) ** (_step_id / self.num_steps)
                    p_energy, p_force = KFOptWrapper.update_force(
                        input_dict, label_dict["force"], pref_f
                    )
                    # [coord, atype, natoms, mapping, shift, nlist, box]
                    model_pred = {"energy": p_energy, "force": p_force}
                    module = (
                        self.wrapper.module if dist.is_initialized() else self.wrapper
                    )
                    loss, more_loss = module.loss[task_key](
                        model_pred,
                        label_dict,
                        int(input_dict["atype"].shape[-1]),
                        learning_rate=pref_lr,
                    )
                elif isinstance(self.loss, DenoiseLoss):
                    KFOptWrapper = KFOptimizerWrapper(
                        self.wrapper, self.optimizer, 24, 6, dist.is_initialized()
                    )
                    module = (
                        self.wrapper.module if dist.is_initialized() else self.wrapper
                    )
                    model_pred = KFOptWrapper.update_denoise_coord(
                        input_dict,
                        label_dict["clean_coord"],
                        1,
                        module.loss[task_key].mask_loss_coord,
                        label_dict["coord_mask"],
                    )
                    loss, more_loss = module.loss[task_key](
                        model_pred,
                        label_dict,
                        input_dict["natoms"],
                        learning_rate=pref_lr,
                    )
            else:
                raise ValueError("Not supported optimizer type '%s'" % self.opt_type)

            # Log and persist
            if _step_id % self.disp_freq == 0:
                self.wrapper.eval()
                msg = f"step={_step_id}, lr={cur_lr:.2e}"

                def log_loss_train(_loss, _more_loss, _task_key="Default"):
                    results = {}
                    if not self.multi_task:
                        suffix = ""
                    else:
                        suffix = f"_{_task_key}"
                    _msg = f"loss{suffix}={_loss:.4f}"
                    rmse_val = {
                        item: _more_loss[item]
                        for item in _more_loss
                        if "l2_" not in item
                    }
                    for item in sorted(rmse_val.keys()):
                        _msg += f", {item}_train{suffix}={rmse_val[item]:.4f}"
                        results[item] = rmse_val[item]
                    return _msg, results

                def log_loss_valid(_task_key="Default"):
                    single_results = {}
                    sum_natoms = 0
                    if not self.multi_task:
                        suffix = ""
                        valid_numb_batch = self.valid_numb_batch
                    else:
                        suffix = f"_{_task_key}"
                        valid_numb_batch = self.valid_numb_batch[_task_key]
                    for ii in range(valid_numb_batch):
                        self.optimizer.zero_grad()
                        input_dict, label_dict, _ = self.get_data(
                            is_train=False, task_key=_task_key
                        )
                        _, loss, more_loss = self.wrapper(
                            **input_dict,
                            cur_lr=pref_lr,
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
                    _msg = ""
                    for item in sorted(results.keys()):
                        _msg += f", {item}_valid{suffix}={results[item]:.4f}"
                    return _msg, results

                if not self.multi_task:
                    temp_msg, train_results = log_loss_train(loss, more_loss)
                    msg += "\n" + temp_msg
                    temp_msg, valid_results = log_loss_valid()
                    msg += temp_msg
                else:
                    train_results = {_key: {} for _key in self.model_keys}
                    valid_results = {_key: {} for _key in self.model_keys}
                    train_msg = {}
                    valid_msg = {}
                    train_msg[task_key], train_results[task_key] = log_loss_train(
                        loss, more_loss, _task_key=task_key
                    )
                    for _key in self.model_keys:
                        if _key != task_key:
                            self.optimizer.zero_grad()
                            input_dict, label_dict, _ = self.get_data(
                                is_train=True, task_key=_key
                            )
                            _, loss, more_loss = self.wrapper(
                                **input_dict,
                                cur_lr=pref_lr,
                                label=label_dict,
                                task_key=_key,
                            )
                            train_msg[_key], train_results[_key] = log_loss_train(
                                loss, more_loss, _task_key=_key
                            )
                        valid_msg[_key], valid_results[_key] = log_loss_valid(
                            _task_key=_key
                        )
                        msg += "\n" + train_msg[_key]
                        msg += valid_msg[_key]

                train_time = time.time() - self.t0
                self.t0 = time.time()
                msg += f", speed={train_time:.2f} s/{self.disp_freq if _step_id else 1} batches"
                log.info(msg)

                if fout:
                    if self.lcurve_should_print_header:
                        self.print_header(fout, train_results, valid_results)
                        self.lcurve_should_print_header = False
                    self.print_on_training(
                        fout, _step_id, cur_lr, train_results, valid_results
                    )

            if (
                ((_step_id + 1) % self.save_freq == 0 and _step_id != self.start_step)
                or (_step_id + 1) == self.num_steps
            ) and (self.rank == 0 or dist.get_rank() == 0):
                # Handle the case if rank 0 aborted and re-assigned
                self.latest_model = Path(self.save_ckpt + f"-{_step_id + 1}.pt")

                module = self.wrapper.module if dist.is_initialized() else self.wrapper
                self.save_model(self.latest_model, lr=cur_lr, step=_step_id)
                log.info(f"Saved model to {self.latest_model}")
                symlink_prefix_files(self.latest_model.stem, self.save_ckpt)
                with open("checkpoint", "w") as f:
                    f.write(str(self.latest_model))

            # tensorboard
            if self.enable_tensorboard and _step_id % self.tensorboard_freq == 0:
                writer.add_scalar(f"{task_key}/lr", cur_lr, _step_id)
                writer.add_scalar(f"{task_key}/loss", loss, _step_id)
                for item in more_loss:
                    writer.add_scalar(f"{task_key}/{item}", more_loss[item], _step_id)

        self.t0 = time.time()
        for step_id in range(self.num_steps):
            if step_id < self.start_step:
                continue
            if self.multi_task:
                chosen_index_list = dp_random.choice(
                    np.arange(self.num_model),
                    p=np.array(self.model_prob),
                    size=self.world_size,
                    replace=True,
                )
                assert chosen_index_list.size == self.world_size
                model_index = chosen_index_list[self.rank]
                model_key = self.model_keys[model_index]
            else:
                model_key = "Default"
            step(step_id, model_key)
            if JIT:
                break

        if (
            self.rank == 0 or dist.get_rank() == 0
        ):  # Handle the case if rank 0 aborted and re-assigned
            if JIT:
                pth_model_path = (
                    "frozen_model.pth"  # We use .pth to denote the frozen model
                )
                self.model.save(pth_model_path)
                log.info(
                    f"Frozen model for inferencing has been saved to {pth_model_path}"
                )
            log.info(f"Trained model has been saved to: {self.save_ckpt}")

        if fout:
            fout.close()
        if SAMPLER_RECORD:
            fout1.close()
        if self.enable_tensorboard:
            writer.close()
        if self.enable_profiler:
            prof.stop()

    def save_model(self, save_path, lr=0.0, step=0):
        module = self.wrapper.module if dist.is_initialized() else self.wrapper
        module.train_infos["lr"] = lr
        module.train_infos["step"] = step
        torch.save(
            {"model": module.state_dict(), "optimizer": self.optimizer.state_dict()},
            save_path,
        )

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
                try:
                    batch_data = next(iter(self.validation_data[task_key]))
                except StopIteration:
                    self.validation_data[task_key] = BufferedIterator(
                        iter(self.validation_dataloader[task_key])
                    )
                    batch_data = next(iter(self.validation_data[task_key]))

        for key in batch_data.keys():
            if key == "sid" or key == "fid":
                continue
            elif not isinstance(batch_data[key], list):
                if batch_data[key] is not None:
                    batch_data[key] = batch_data[key].to(DEVICE)
            else:
                batch_data[key] = [item.to(DEVICE) for item in batch_data[key]]
        input_dict = {}
        for item in [
            "coord",
            "atype",
            "box",
        ]:
            if item in batch_data:
                input_dict[item] = batch_data[item]
            else:
                input_dict[item] = None
        label_dict = {}
        for item in [
            "energy",
            "force",
            "virial",
            "clean_coord",
            "clean_type",
            "coord_mask",
            "type_mask",
        ]:
            if item in batch_data:
                label_dict[item] = batch_data[item]
        log_dict = {}
        if "fid" in batch_data:
            log_dict["fid"] = batch_data["fid"]
        log_dict["sid"] = batch_data["sid"]
        return input_dict, label_dict, log_dict

    def print_header(self, fout, train_results, valid_results):
        train_keys = sorted(train_results.keys())
        print_str = ""
        print_str += "# %5s" % "step"
        if not self.multi_task:
            if valid_results is not None:
                prop_fmt = "   %11s %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_val", k + "_trn")
            else:
                prop_fmt = "   %11s"
                for k in train_keys:
                    print_str += prop_fmt % (k + "_trn")
        else:
            for model_key in self.model_keys:
                if valid_results[model_key] is not None:
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
        print_str += "   %8s\n" % "lr"
        fout.write(print_str)
        fout.flush()

    def print_on_training(self, fout, step_id, cur_lr, train_results, valid_results):
        train_keys = sorted(train_results.keys())
        print_str = ""
        print_str += "%7d" % step_id
        if not self.multi_task:
            if valid_results is not None:
                prop_fmt = "   %11.2e %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (valid_results[k], train_results[k])
            else:
                prop_fmt = "   %11.2e"
                for k in train_keys:
                    print_str += prop_fmt % (train_results[k])
        else:
            for model_key in self.model_keys:
                if valid_results[model_key] is not None:
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
        print_str += "   %8.1e\n" % cur_lr
        fout.write(print_str)
        fout.flush()
