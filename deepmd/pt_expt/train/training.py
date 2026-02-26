# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training loop for the pt_expt backend.

Uses ``DeepmdDataSystem`` (numpy-based batch provider) instead of the
pt backend's ``DpLoaderSet`` + ``DataLoader``.  NumPy batches are
converted to torch tensors at the boundary.
"""

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
import torch

from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.loggers.training import (
    format_training_message_per_task,
)
from deepmd.pt.loss import (
    EnergyHessianStdLoss,
    EnergyStdLoss,
    TaskLoss,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.pt_expt.utils.stat import (
    make_stat_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: loss factory (reused from pt)
# ---------------------------------------------------------------------------


def get_loss(
    loss_params: dict[str, Any],
    start_lr: float,
    _ntypes: int,
    _model: Any,
) -> TaskLoss:
    loss_type = loss_params.get("type", "ener")
    if loss_type == "ener" and loss_params.get("start_pref_h", 0.0) > 0.0:
        loss_params["starter_learning_rate"] = start_lr
        return EnergyHessianStdLoss(**loss_params)
    elif loss_type == "ener":
        loss_params["starter_learning_rate"] = start_lr
        return EnergyStdLoss(**loss_params)
    else:
        loss_params["starter_learning_rate"] = start_lr
        return TaskLoss.get_class_by_type(loss_type).get_loss(loss_params)


def get_additional_data_requirement(_model: Any) -> list[DataRequirementItem]:
    additional_data_requirement: list[DataRequirementItem] = []
    if _model.get_dim_fparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "fparam",
                _model.get_dim_fparam(),
                atomic=False,
                must=False,
                default=0.0,
            )
        )
    if _model.get_dim_aparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "aparam", _model.get_dim_aparam(), atomic=True, must=True
            )
        )
    return additional_data_requirement


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Training driver for the pt_expt backend.

    Uses ``DeepmdDataSystem`` for data loading (numpy batches converted
    to torch tensors at the boundary).  Single-task, single-GPU only.

    Parameters
    ----------
    config : dict
        Full training configuration.
    training_data : DeepmdDataSystem
        Training data.
    stat_file_path : DPPath or None
        Path for saving / loading statistics.
    validation_data : DeepmdDataSystem or None
        Validation data.
    init_model : str or None
        Path to a checkpoint to initialise weights from.
    restart_model : str or None
        Path to a checkpoint to *restart* training from (restores step + optimiser).
    """

    def __init__(
        self,
        config: dict[str, Any],
        training_data: DeepmdDataSystem,
        stat_file_path: DPPath | None = None,
        validation_data: DeepmdDataSystem | None = None,
        init_model: str | None = None,
        restart_model: str | None = None,
    ) -> None:
        resume_model = init_model or restart_model
        resuming = resume_model is not None
        self.restart_training = restart_model is not None

        model_params = config["model"]
        training_params = config["training"]

        # Iteration config
        self.num_steps = training_params["numb_steps"]
        self.disp_file = training_params.get("disp_file", "lcurve.out")
        self.disp_freq = training_params.get("disp_freq", 1000)
        self.save_ckpt = training_params.get("save_ckpt", "model.ckpt")
        self.save_freq = training_params.get("save_freq", 1000)
        self.display_in_training = training_params.get("disp_training", True)
        self.timing_in_training = training_params.get("time_training", True)
        self.lcurve_should_print_header = True

        # Model ---------------------------------------------------------------
        self.model = get_model(deepcopy(model_params)).to(DEVICE)

        # Loss ----------------------------------------------------------------
        self.loss = get_loss(
            config.get("loss", {}),
            config["learning_rate"]["start_lr"],
            len(model_params["type_map"]),
            self.model,
        )

        # Data requirements ---------------------------------------------------
        data_requirement = self.loss.label_requirement
        data_requirement += get_additional_data_requirement(self.model)
        training_data.add_data_requirements(data_requirement)
        if validation_data is not None:
            validation_data.add_data_requirements(data_requirement)

        self.training_data = training_data
        self.validation_data = validation_data
        self.valid_numb_batch = training_params.get("validation_data", {}).get(
            "numb_btch", 1
        )

        # Statistics ----------------------------------------------------------
        data_stat_nbatch = model_params.get("data_stat_nbatch", 10)

        @functools.lru_cache
        def get_sample() -> list[dict[str, np.ndarray]]:
            return make_stat_input(training_data, data_stat_nbatch)

        if not resuming:
            self.model.compute_or_load_stat(
                sampled_func=get_sample,
                stat_file_path=stat_file_path,
            )

        # Learning rate -------------------------------------------------------
        lr_params = config["learning_rate"].copy()
        lr_params["num_steps"] = self.num_steps
        self.lr_schedule = LearningRateExp(**lr_params)

        # Gradient clipping
        self.gradient_max_norm = training_params.get("gradient_max_norm", 0.0)

        # Model wrapper -------------------------------------------------------
        self.wrapper = ModelWrapper(self.model, self.loss, model_params=model_params)
        self.start_step = 0

        # Optimiser -----------------------------------------------------------
        opt_type = training_params.get("opt_type", "Adam")
        initial_lr = float(self.lr_schedule.value(self.start_step))

        if opt_type == "Adam":
            self.optimizer = torch.optim.Adam(self.wrapper.parameters(), lr=initial_lr)
        elif opt_type == "AdamW":
            weight_decay = training_params.get("weight_decay", 0.001)
            self.optimizer = torch.optim.AdamW(
                self.wrapper.parameters(),
                lr=initial_lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: self.lr_schedule.value(step + self.start_step) / initial_lr,
            last_epoch=self.start_step - 1,
        )

        # Resume --------------------------------------------------------------
        if resuming:
            log.info(f"Resuming from {resume_model}.")
            state_dict = torch.load(
                resume_model, map_location=DEVICE, weights_only=True
            )
            if "model" in state_dict:
                optimizer_state_dict = (
                    state_dict["optimizer"] if self.restart_training else None
                )
                state_dict = state_dict["model"]
            else:
                optimizer_state_dict = None

            self.start_step = (
                state_dict["_extra_state"]["train_infos"]["step"]
                if self.restart_training
                else 0
            )
            self.wrapper.load_state_dict(state_dict)
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)
                # rebuild scheduler from the resumed step
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer,
                    lambda step: (
                        self.lr_schedule.value(step + self.start_step) / initial_lr
                    ),
                    last_epoch=self.start_step - 1,
                )

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def get_data(
        self,
        is_train: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetch a batch and split into input / label dicts.

        Returns
        -------
        input_dict, label_dict
        """
        data_sys = self.training_data if is_train else self.validation_data
        if data_sys is None:
            return {}, {}

        batch = data_sys.get_batch()  # numpy dict

        input_keys = {"coord", "box", "fparam", "aparam"}
        input_dict: dict[str, Any] = {}
        label_dict: dict[str, Any] = {}

        natoms = batch["type"].shape[-1]

        for key, val in batch.items():
            if key == "type":
                # rename to atype; convert to int64 tensor
                input_dict["atype"] = torch.from_numpy(val.astype(np.int64)).to(DEVICE)
            elif key == "coord":
                # reshape from [nf, natoms*3] → [nf, natoms, 3]
                t = torch.from_numpy(val.copy()).to(
                    dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                )
                # requires_grad needed for force computation via autograd
                input_dict["coord"] = t.reshape(-1, natoms, 3).requires_grad_(True)
            elif key == "box":
                if val is not None:
                    t = torch.from_numpy(val).to(
                        dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                    )
                    input_dict["box"] = t.reshape(-1, 3, 3)
                else:
                    input_dict["box"] = None
            elif key in input_keys:
                if val is not None and isinstance(val, np.ndarray):
                    input_dict[key] = torch.from_numpy(val).to(
                        dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                    )
            elif key in ("natoms_vec", "default_mesh", "sid", "fid"):
                continue
            elif "find_" in key:
                # find_energy, find_force, … — keep as float scalar
                label_dict[key] = float(val) if not isinstance(val, float) else val
            elif key == "force":
                # [nf, natoms*3] → [nf, natoms, 3]
                t = torch.from_numpy(val).to(
                    dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                )
                label_dict["force"] = t.reshape(-1, natoms, 3)
            else:
                if isinstance(val, np.ndarray):
                    label_dict[key] = torch.from_numpy(val).to(
                        dtype=GLOBAL_PT_FLOAT_PRECISION, device=DEVICE
                    )
                else:
                    label_dict[key] = val

        return input_dict, label_dict

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        self.wrapper.train_infos["step"] = step
        state = {
            "model": self.wrapper.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        ckpt_path = f"{self.save_ckpt}-{step}.pt"
        torch.save(state, ckpt_path)
        # symlink latest
        latest = Path(f"{self.save_ckpt}.pt")
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ckpt_path)
        log.info(f"Saved checkpoint to {ckpt_path}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        fout = open(
            self.disp_file,
            mode="w" if not self.restart_training else "a",
            buffering=1,
        )
        log.info("Start to train %d steps.", self.num_steps)

        self.wrapper.train()

        for step_id in range(self.start_step, self.num_steps):
            cur_lr = float(self.lr_schedule.value(step_id))

            if self.timing_in_training:
                t_start = time.time()

            # --- forward / backward ---
            self.optimizer.zero_grad(set_to_none=True)
            input_dict, label_dict = self.get_data(is_train=True)

            cur_lr_sched = self.scheduler.get_last_lr()[0]
            model_pred, loss, more_loss = self.wrapper(
                **input_dict, cur_lr=cur_lr_sched, label=label_dict
            )
            loss.backward()

            if self.gradient_max_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.wrapper.parameters(), self.gradient_max_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            if self.timing_in_training:
                t_end = time.time()

            # --- display ---
            display_step_id = step_id + 1
            if self.display_in_training and (
                display_step_id % self.disp_freq == 0 or display_step_id == 1
            ):
                self.wrapper.eval()

                train_results = {k: v for k, v in more_loss.items() if "l2_" not in k}

                # validation
                valid_results: dict[str, Any] = {}
                if self.validation_data is not None:
                    sum_natoms = 0
                    for _ii in range(self.valid_numb_batch):
                        val_input, val_label = self.get_data(is_train=False)
                        if not val_input:
                            break
                        _, _vloss, _vmore = self.wrapper(
                            **val_input, cur_lr=cur_lr_sched, label=val_label
                        )
                        natoms = int(val_input["atype"].shape[-1])
                        sum_natoms += natoms
                        for k, v in _vmore.items():
                            if "l2_" not in k:
                                valid_results[k] = (
                                    valid_results.get(k, 0.0) + v * natoms
                                )
                    if sum_natoms > 0:
                        valid_results = {
                            k: v / sum_natoms for k, v in valid_results.items()
                        }

                # log
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

                # lcurve file
                if self.lcurve_should_print_header:
                    self.print_header(fout, train_results, valid_results)
                    self.lcurve_should_print_header = False
                self.print_on_training(
                    fout, display_step_id, cur_lr, train_results, valid_results
                )

                self.wrapper.train()

            # --- checkpoint ---
            if display_step_id % self.save_freq == 0:
                self.save_checkpoint(display_step_id)

        # final save
        self.save_checkpoint(self.num_steps)
        fout.close()
        log.info("Training finished.")

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def print_header(
        self,
        fout: Any,
        train_results: dict[str, Any],
        valid_results: dict[str, Any],
    ) -> None:
        train_keys = sorted(train_results.keys())
        header = "# {:5s}".format("step")
        if valid_results:
            for k in train_keys:
                header += f"   {k + '_val':>11s} {k + '_trn':>11s}"
        else:
            for k in train_keys:
                header += f"   {k + '_trn':>11s}"
        header += "   {:8s}\n".format("lr")
        fout.write(header)
        fout.flush()

    def print_on_training(
        self,
        fout: Any,
        step_id: int,
        cur_lr: float,
        train_results: dict,
        valid_results: dict,
    ) -> None:
        train_keys = sorted(train_results.keys())
        line = f"{step_id:7d}"
        if valid_results:
            for k in train_keys:
                line += f"   {valid_results.get(k, float('nan')):11.2e} {train_results[k]:11.2e}"
        else:
            for k in train_keys:
                line += f"   {train_results[k]:11.2e}"
        line += f"   {cur_lr:8.1e}\n"
        fout.write(line)
        fout.flush()
