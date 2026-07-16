#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Local training utilities for the JAX backend."""

import functools
import logging
import os
import platform
import shutil
import time
from collections.abc import (
    Mapping,
)
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

import numpy as np
import optax
import orbax.checkpoint as ocp
from packaging.version import (
    Version,
)

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
    AbstractTrainer,
    RankContext,
    TrainerConfig,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
    change_model_out_bias_by_task,
)
from deepmd.dpmodel.train.validation import (
    resolve_best_checkpoint_dir,
)
from deepmd.dpmodel.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.dpmodel.utils.multi_task import (
    apply_shared_links,
    set_descriptor_component,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.dpmodel.utils.training_utils import (
    resolve_model_prob,
)
from deepmd.jax.env import (
    flax_version,
    jax,
    jnp,
    nnx,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.jax.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.jax.utils.serialization import (
    _drop_zero_size_array_leaves,
    serialize_from_file,
)
from deepmd.utils.argcheck import (
    resolve_full_validation_start_step,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)
from deepmd.utils.finetune import (
    warn_configuration_mismatch_during_finetune,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

log = logging.getLogger(__name__)


class DPTrainer(AbstractTrainer):
    """Train JAX DeePMD models on local devices."""

    def __init__(
        self,
        jdata: dict,
        init_model: str | None = None,
        restart: str | None = None,
        finetune_model: str | None = None,
        finetune_links: dict[str, Any] | None = None,
        shared_links: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the trainer from input data and optional checkpoints."""
        if finetune_model is not None and (
            init_model is not None or restart is not None
        ):
            raise ValueError(
                "finetune_model cannot be combined with init_model or restart."
            )
        self.init_model = init_model
        self.restart = restart
        self.finetune_model = finetune_model
        self.finetune_links = finetune_links
        self.shared_links = shared_links
        self.restart_training = restart is not None
        self.training_param = jdata["training"]
        self.validating_param = jdata.get("validating", {}) or {}
        self.num_steps = self.training_param["numb_steps"]
        self.start_step = 0

        self.model_def_script = jdata["model"]
        self.multi_task = "model_dict" in self.model_def_script
        if (
            self.multi_task
            and self.shared_links is None
            and self.model_def_script.get("shared_dict")
        ):
            self.model_def_script, self.shared_links = preprocess_shared_params(
                self.model_def_script
            )
        self.model_keys = (
            list(self.model_def_script["model_dict"])
            if self.multi_task
            else [DEFAULT_TASK_KEY]
        )
        self.model_params_by_task = self._model_params_by_task(self.model_def_script)

        if init_model is not None or restart is not None:
            checkpoint_path = init_model if init_model is not None else restart
            assert checkpoint_path is not None
            checkpoint_data = serialize_from_file(checkpoint_path)
            checkpoint_multi_task = "model_dict" in checkpoint_data["model_def_script"]
            if checkpoint_multi_task != self.multi_task:
                raise ValueError(
                    "JAX init/restart checkpoint task layout does not match input config."
                )
            checkpoint_keys = list(
                checkpoint_data["model_def_script"].get(
                    "model_dict", {DEFAULT_TASK_KEY: None}
                )
            )
            if checkpoint_keys != self.model_keys:
                raise ValueError(
                    "JAX init/restart checkpoint task keys do not match input config."
                )
            self.models = self._deserialize_models(checkpoint_data)
            if restart is not None:
                self.model_def_script = checkpoint_data["model_def_script"]
                self.model_params_by_task = self._model_params_by_task(
                    self.model_def_script
                )
                self.start_step = int(
                    checkpoint_data.get("model_def_script", {}).get(
                        "current_step",
                        checkpoint_data.get("@variables", {}).get("current_step", 0),
                    )
                )
        else:
            self.models = {
                model_key: get_model(deepcopy(self.model_params_by_task[model_key]))
                for model_key in self.model_keys
            }
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]

        learning_rate_param = jdata["learning_rate"]
        self.lr = self._get_lr_and_coef(learning_rate_param)
        self.losses = self._build_losses(jdata, learning_rate_param)
        self.loss = self.losses if self.multi_task else self.losses[DEFAULT_TASK_KEY]
        self.data_requirements_by_task = {
            model_key: list(self.losses[model_key].label_requirement)
            for model_key in self.model_keys
        }

        self.valid_numb_batch_by_task = self._valid_numb_batch_by_task()
        self.valid_numb_batch = (
            self.valid_numb_batch_by_task
            if self.multi_task
            else self.valid_numb_batch_by_task[DEFAULT_TASK_KEY]
        )

        tr_data = self.training_param
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.max_ckpt_keep = tr_data.get("max_ckpt_keep", 5)
        self.display_in_training = tr_data.get("disp_training", True)
        self.timing_in_training = tr_data.get("time_training", True)
        self.profiling = tr_data.get("profiling", False)
        self.profiling_file = tr_data.get("profiling_file", "timeline.json")
        self.enable_profiler = tr_data.get("enable_profiler", False)
        self.tensorboard = tr_data.get("tensorboard", False)
        self.tensorboard_log_dir = tr_data.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = tr_data.get("tensorboard_freq", 1)
        self.mixed_prec = tr_data.get("mixed_precision", None)
        self.change_bias_after_training = bool(
            tr_data.get("change_bias_after_training", False)
        )
        self.numb_fparam = (
            {key: model.get_dim_fparam() for key, model in self.models.items()}
            if self.multi_task
            else self.models[DEFAULT_TASK_KEY].get_dim_fparam()
        )

        self.frz_model = None
        self.ckpt_meta = None
        self.model_type = None
        self.optimizers: dict[str, nnx.Optimizer] = {}
        self.optimizer: nnx.Optimizer | None = None
        self._train_step_impls: dict[str, Any] = {}
        self._loss_fn_more_loss: dict[str, Any] = {}
        self._sample_funcs: dict[str, Any] = {}
        self.model_prob: np.ndarray | None = None
        self.full_validator: Any | None = None

        super().__init__(
            TrainerConfig.from_training_params(
                tr_data,
                num_steps=self.num_steps,
                start_step=self.start_step,
                restart_training=self.restart is not None,
            ),
            rank_context=RankContext(
                rank=int(jax.process_index()),
                world_size=int(jax.process_count()),
            ),
        )

    @staticmethod
    def _model_params_by_task(
        model_params: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        if "model_dict" in model_params:
            return {
                model_key: model_params["model_dict"][model_key]
                for model_key in model_params["model_dict"]
            }
        return {DEFAULT_TASK_KEY: model_params}

    @staticmethod
    def _deserialize_models(model_data: dict[str, Any]) -> dict[str, BaseModel]:
        if "model_dict" in model_data["model_def_script"]:
            return {
                model_key: BaseModel.deserialize(
                    model_data["model"]["model_dict"][model_key]
                )
                for model_key in model_data["model_def_script"]["model_dict"]
            }
        return {DEFAULT_TASK_KEY: BaseModel.deserialize(model_data["model"])}

    def _get_lr_and_coef(self, lr_param: dict[str, Any]) -> LearningRateExp:
        lr_type = lr_param.get("type", "exp")
        if lr_type == "exp":
            return LearningRateExp(**lr_param, num_steps=self.num_steps)
        raise RuntimeError("unknown learning_rate type " + lr_type)

    def _build_losses(
        self,
        jdata: dict[str, Any],
        learning_rate_param: dict[str, Any],
    ) -> dict[str, EnergyLoss]:
        losses: dict[str, EnergyLoss] = {}
        for model_key in self.model_keys:
            loss_param = deepcopy(
                jdata["loss_dict"][model_key]
                if self.multi_task
                else jdata.get("loss", {})
            )
            loss_param["starter_learning_rate"] = learning_rate_param["start_lr"]
            loss_type = loss_param.get("type", "ener")
            if loss_type != "ener":
                raise RuntimeError("unknown loss type " + loss_type)
            losses[model_key] = EnergyLoss.get_loss(loss_param)
        return losses

    def _valid_numb_batch_by_task(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for model_key in self.model_keys:
            if self.multi_task:
                valid_params = (
                    self.training_param["data_dict"][model_key].get(
                        "validation_data", {}
                    )
                    or {}
                )
            else:
                valid_params = self.training_param.get("validation_data", {}) or {}
            result[model_key] = max(int(valid_params.get("numb_btch", 1)), 1)
        return result

    def _create_full_validator(
        self,
        *,
        validating_params: dict[str, Any],
        validation_data: DeepmdDataSystem | None,
    ) -> Any | None:
        """Create the runtime full validator when it is active."""
        if not self._is_validation_requested(validating_params, "full_validation"):
            return None
        self._raise_if_full_validation_unsupported(validation_data)
        if validation_data is None:
            raise RuntimeError(
                "validation_data must be available after full validation checks."
            )
        from deepmd.jax.train.validation import (
            JAXFullValidator,
        )

        return JAXFullValidator(
            validating_params=validating_params,
            validation_data=validation_data,
            model=self.models[DEFAULT_TASK_KEY],
            state_store=self.model_def_script,
            num_steps=self.num_steps,
            rank=int(jax.process_index()),
            restart_training=self.restart_training,
            checkpoint_dir=resolve_best_checkpoint_dir(
                validating_params, self.save_ckpt
            ),
        )

    def _is_validation_requested(
        self,
        validating_params: dict[str, Any],
        flag_name: str,
    ) -> bool:
        """Check whether a full validation flow can trigger during this run."""
        if not validating_params.get(flag_name, False):
            return False
        start_step = resolve_full_validation_start_step(
            validating_params.get("full_val_start", 0.5),
            self.num_steps,
        )
        return start_step is not None and start_step <= self.num_steps

    def _raise_if_full_validation_unsupported(
        self,
        validation_data: DeepmdDataSystem | None,
    ) -> None:
        """Validate runtime full validation constraints."""
        if self.multi_task:
            raise ValueError(
                "validating.full_validation only supports single-task energy "
                "training; multi-task training is not supported."
            )

        if not isinstance(self.loss, EnergyLoss):
            raise ValueError(
                "validating.full_validation only supports single-task energy training."
            )

        if validation_data is None:
            raise ValueError(
                "validating.full_validation requires `training.validation_data` "
                "to be configured."
            )

    @property
    def data_requirements(self) -> list[DataRequirementItem]:
        """Labels required by the configured loss for single-task callers."""
        return self.data_requirements_by_task[DEFAULT_TASK_KEY]

    def set_min_nbor_dist(
        self,
        min_nbor_dist: float | Mapping[str, float | None] | None,
    ) -> None:
        """Attach neighbor-stat minimum distances to task models."""
        if min_nbor_dist is None:
            return
        if isinstance(min_nbor_dist, Mapping):
            for model_key, value in min_nbor_dist.items():
                if value is not None and model_key in self.models:
                    self.models[model_key].min_nbor_dist = value
            return
        self.models[DEFAULT_TASK_KEY].min_nbor_dist = min_nbor_dist

    def train(
        self,
        train_data: DeepmdDataSystem | Mapping[str, DeepmdDataSystem],
        valid_data: DeepmdDataSystem
        | Mapping[str, DeepmdDataSystem | None]
        | None = None,
    ) -> None:
        """Run the training loop with optional validation data."""
        train_data_by_task = self._normalize_data_map(train_data)
        valid_data_by_task = self._normalize_data_map(valid_data, optional=True)
        self._setup_training(train_data_by_task, valid_data_by_task)
        tasks = TrainingTaskCollection(
            [
                TrainingTask(
                    key=model_key,
                    training_data=train_data_by_task[model_key],
                    validation_data=valid_data_by_task[model_key],
                    valid_numb_batch=self.valid_numb_batch_by_task[model_key],
                    data_requirements=self.data_requirements_by_task[model_key],
                )
                for model_key in self.model_keys
            ],
            probabilities=self.model_prob,
        )
        self.run(tasks)

    def _normalize_data_map(
        self,
        data: Any,
        *,
        optional: bool = False,
    ) -> dict[str, Any]:
        if isinstance(data, Mapping):
            return {model_key: data.get(model_key) for model_key in self.model_keys}
        if optional and data is None:
            return dict.fromkeys(self.model_keys)
        return {DEFAULT_TASK_KEY: data}

    def _setup_training(
        self,
        train_data_by_task: Mapping[str, DeepmdDataSystem],
        valid_data_by_task: Mapping[str, DeepmdDataSystem | None],
    ) -> None:
        """Initialize statistics, fine-tuning, optimizers, and JIT functions."""
        for model_key in self.model_keys:
            train_data_by_task[model_key].add_data_requirements(
                self.data_requirements_by_task[model_key]
            )
            if valid_data_by_task[model_key] is not None:
                valid_data_by_task[model_key].add_data_requirements(
                    self.data_requirements_by_task[model_key]
                )

        if self.multi_task:
            self.model_prob = resolve_model_prob(
                self.model_keys,
                self.training_param.get("model_prob"),
                dict(train_data_by_task),
            )

        for model_key in self.model_keys:
            self._sample_funcs[model_key] = self._make_sample_func(
                train_data_by_task[model_key],
                self.model_params_by_task[model_key].get("data_stat_nbatch", 10),
            )

        if self.init_model is None and self.restart is None:
            for model_key in self.model_keys:
                finetune_has_new_type = (
                    self.finetune_model is not None
                    and self.finetune_links is not None
                    and model_key in self.finetune_links
                    and self.finetune_links[model_key].get_has_new_type()
                )
                if self.finetune_model is None or finetune_has_new_type:
                    self.models[model_key].atomic_model.compute_or_load_stat(
                        self._sample_funcs[model_key]
                    )

        if self.finetune_model is not None:
            self._apply_finetune()

        self._share_model_params(
            resume=self.init_model is not None
            or self.restart is not None
            or self.finetune_model is not None
        )

        for model_key in self.model_keys:
            tx = optax.chain(
                optax.scale_by_adam(),
                _scale_by_global_learning_rate(),
            )
            self.optimizers[model_key] = nnx.Optimizer(
                self.models[model_key], tx, wrt=nnx.Param
            )
            (
                self._train_step_impls[model_key],
                self._loss_fn_more_loss[model_key],
            ) = self._make_step_functions(self.losses[model_key])
        self.optimizer = (
            self.optimizers[DEFAULT_TASK_KEY] if not self.multi_task else None
        )
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]
        self.full_validator = self._create_full_validator(
            validating_params=self.validating_param,
            validation_data=valid_data_by_task[DEFAULT_TASK_KEY]
            if not self.multi_task
            else None,
        )

    @staticmethod
    def _make_sample_func(
        train_data: DeepmdDataSystem,
        data_stat_nbatch: int,
    ) -> Any:
        @functools.lru_cache
        def sample() -> list[dict[str, Any]]:
            stat_data = make_stat_input(train_data, data_stat_nbatch)
            return [
                {
                    key: jnp.asarray(value) if isinstance(value, np.ndarray) else value
                    for key, value in single_data.items()
                }
                for single_data in stat_data
            ]

        return sample

    def _apply_finetune(self) -> None:
        if self.finetune_model is None or self.finetune_links is None:
            return
        pretrained_data = serialize_from_file(self.finetune_model)
        pretrained_params = pretrained_data["model_def_script"]
        pretrained_models = self._deserialize_models(pretrained_data)
        for model_key in self.model_keys:
            finetune_rule = self.finetune_links[model_key]
            source_key = finetune_rule.get_model_branch()
            if source_key not in pretrained_models:
                raise ValueError(
                    f"Pretrained model branch {source_key!r} is not available."
                )
            source_model = pretrained_models[source_key]
            if finetune_rule.get_finetune_tmap() != source_model.get_type_map():
                model_with_new_type_stat = (
                    self.models[model_key] if finetune_rule.get_has_new_type() else None
                )
                source_model.change_type_map(
                    finetune_rule.get_finetune_tmap(),
                    model_with_new_type_stat=model_with_new_type_stat,
                )
            self._warn_finetune_config_mismatch(
                model_key, source_key, pretrained_params
            )
            self.models[model_key] = self._copy_finetune_state(
                self.models[model_key],
                source_model,
                random_fitting=finetune_rule.get_random_fitting(),
            )
            if finetune_rule.get_resuming():
                log.info("Model branch %s will resume training.", model_key)
                continue
            bias_mode = (
                "change-by-statistic"
                if not finetune_rule.get_random_fitting()
                else "set-by-statistic"
            )
            self.models[model_key].change_out_bias(
                self._sample_funcs[model_key],
                bias_adjust_mode=bias_mode,
            )

    def _share_model_params(self, *, resume: bool = False) -> None:
        """Apply multi-task shared_dict links to JAX model branches."""
        if not self.multi_task or not self.shared_links:
            return
        data_stat_protect = np.array(
            [
                self.model_params_by_task[model_key].get("data_stat_protect", 1e-2)
                for model_key in self.model_keys
            ]
        )
        if not np.allclose(data_stat_protect, data_stat_protect[0]):
            raise ValueError(
                "Model key 'data_stat_protect' must be the same in each branch when multitask!"
            )
        if self.model_prob is None:
            model_prob = np.ones(len(self.model_keys), dtype=float) / len(
                self.model_keys
            )
        else:
            model_prob = self.model_prob
        share_jax_model_params(
            self.models,
            self.shared_links,
            model_key_prob_map=dict(zip(self.model_keys, model_prob, strict=True)),
            data_stat_protect=float(data_stat_protect[0]),
            resume=resume,
        )

    def _warn_finetune_config_mismatch(
        self,
        model_key: str,
        source_key: str,
        pretrained_params: dict[str, Any],
    ) -> None:
        input_model_params = self.model_params_by_task[model_key]
        branch_pretrained_params = (
            pretrained_params["model_dict"][source_key]
            if "model_dict" in pretrained_params
            else pretrained_params
        )
        if (
            "descriptor" in input_model_params
            and "descriptor" in branch_pretrained_params
        ):
            warn_configuration_mismatch_during_finetune(
                input_model_params["descriptor"],
                branch_pretrained_params["descriptor"],
                source_key,
            )

    @staticmethod
    def _copy_finetune_state(
        target_model: BaseModel,
        source_model: BaseModel,
        *,
        random_fitting: bool,
    ) -> BaseModel:
        graphdef, target_state = nnx.split(target_model)
        _, source_state = nnx.split(source_model)
        copied = _copy_matching_state_tree(
            target_state.to_pure_dict(),
            source_state.to_pure_dict(),
            random_fitting=random_fitting,
        )
        target_state.replace_by_pure_dict(copied)
        return nnx.merge(graphdef, target_state)

    def _make_step_functions(self, loss_obj: EnergyLoss) -> tuple[Any, Any]:
        def loss_fn(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> jnp.ndarray:
            model_dict = _evaluate_model_dict(
                model, extended_coord, extended_atype, nlist, mapping, fp, ap
            )
            model_dict = _match_label_shapes(model_dict, label_dict)
            loss, _ = loss_obj(
                learning_rate=lr,
                natoms=label_dict["type"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return loss

        @nnx.jit
        def loss_fn_more_loss(
            model: BaseModel,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> dict[str, jnp.ndarray]:
            model_dict = _evaluate_model_dict(
                model, extended_coord, extended_atype, nlist, mapping, fp, ap
            )
            model_dict = _match_label_shapes(model_dict, label_dict)
            _, more_loss = loss_obj(
                learning_rate=lr,
                natoms=label_dict["type"].shape[1],
                model_dict=model_dict,
                label_dict=label_dict,
            )
            return more_loss

        @nnx.jit
        def train_step(
            model: BaseModel,
            optimizer: nnx.Optimizer,
            lr: float,
            label_dict: dict[str, jnp.ndarray],
            extended_coord: jnp.ndarray,
            extended_atype: jnp.ndarray,
            nlist: jnp.ndarray,
            mapping: jnp.ndarray | None,
            fp: jnp.ndarray | None,
            ap: jnp.ndarray | None,
        ) -> None:
            grads = nnx.grad(loss_fn)(
                model,
                lr,
                label_dict,
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fp,
                ap,
            )
            if Version(flax_version) >= Version("0.11.0"):
                optimizer.update(model, grads, learning_rate=lr)
            else:
                _legacy_optimizer_update(optimizer, grads, lr)

        return train_step, loss_fn_more_loss

    def select_task(self, tasks: TrainingTaskCollection) -> TrainingTask:
        """Select a task using DeePMD's seeded random helper."""
        if len(tasks) == 1:
            return tasks[tasks.keys[0]]
        from deepmd.utils import random as dp_random

        model_index = dp_random.choice(
            np.arange(len(tasks), dtype=np.int_),
            p=tasks.probabilities,
        )
        return tasks[tasks.keys[int(model_index)]]

    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        """Run one JAX optimizer step."""
        task_key = task.key
        if task_key not in self.optimizers or task_key not in self._train_step_impls:
            raise RuntimeError("JAX trainer has not been initialized.")
        prepared = self._prepare_batch(task.training_data.get_batch(), task_key)
        self._train_step_impls[task_key](
            self.models[task_key],
            self.optimizers[task_key],
            self.lr.value(step),
            *prepared,
        )
        return TrainStepResult(task_key=task_key, step=step, payload=prepared)

    def evaluate_training(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float]:
        """Evaluate training loss terms for display."""
        prepared = (
            step_result.payload
            if step_result is not None and step_result.task_key == task.key
            else self._prepare_batch(task.training_data.get_batch(), task.key)
        )
        return self._evaluate_prepared_batch(task.key, step, prepared)

    def evaluate_validation(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float] | None:
        """Evaluate validation loss terms for display."""
        if task.validation_data is None:
            return None
        valid_more_loss_list = [
            self._evaluate_prepared_batch(
                task.key,
                step,
                self._prepare_batch(task.validation_data.get_batch(), task.key),
            )
            for _ in range(task.valid_numb_batch)
        ]
        return {
            key: sum(loss[key] for loss in valid_more_loss_list)
            / len(valid_more_loss_list)
            for key in valid_more_loss_list[0]
        }

    def learning_rate(self, step: int) -> float:
        """Return the configured learning rate for a zero-based step."""
        return float(self.lr.value(step))

    def save_checkpoint(self, step: int) -> None:
        """Persist a JAX checkpoint for a one-based step."""
        self._save_checkpoint(step)

    def run(self, tasks: TrainingTaskCollection) -> None:
        """Run JAX training through the backend-independent trainer loop."""
        log.info("Start to train %d steps.", self.num_steps)
        wall_start = time.time()
        super().run(tasks)
        if self.change_bias_after_training and self.num_steps > self.start_step:
            self._change_bias_after_training()
            if self.rank_context.is_chief:
                self.save_checkpoint(self.num_steps)
        log.info("Training finished. Total wall time: %.2fs", time.time() - wall_start)

    def _change_bias_after_training(self) -> None:
        if self.rank_context.is_chief:
            change_model_out_bias_by_task(
                self.models,
                self._sample_funcs,
                self.model_keys,
                bias_adjust_mode="change-by-statistic",
            )
        if self.rank_context.world_size <= 1:
            return
        from jax.experimental import (
            multihost_utils,
        )

        for model_key in self.model_keys:
            _, state = nnx.split(self.models[model_key])
            state = multihost_utils.broadcast_one_to_all(
                state.to_pure_dict(),
                is_source=self.rank_context.is_chief,
            )
            nnx.update(self.models[model_key], state)

    def run_full_validation(
        self,
        *,
        step: int,
        display_step: int,
        learning_rate: float,
    ) -> None:
        """Run optional full validation for one step."""
        if self.full_validator is None:
            return None
        self.full_validator.run(
            step_id=display_step,
            display_step=display_step,
            lr=learning_rate,
            save_checkpoint=self._save_full_validation_checkpoint,
        )
        return None

    def _prepare_batch(
        self,
        batch_data: dict[str, np.ndarray | np.floating],
        task_key: str,
    ) -> tuple[
        dict[str, jnp.ndarray | bool],
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray | None,
        jnp.ndarray | None,
        jnp.ndarray | None,
    ]:
        """Convert one data-system batch into JAX model inputs."""
        model = self.models[task_key]
        jax_data = convert_numpy_data_to_jax_data(batch_data)
        extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
            rcut=model.get_rcut(),
            sel=model.get_sel(),
            coord=jax_data["coord"],
            atype=jax_data["type"],
            box=jax_data["box"] if jax_data["find_box"] else None,
            fparam=jax_data.get("fparam", None),
            aparam=jax_data.get("aparam", None),
            pair_excl=getattr(model.atomic_model, "pair_excl", None),
            conservative_nlist=type(model.get_descriptor()).__name__ == "DescrptDPA4",
        )
        return jax_data, extended_coord, extended_atype, nlist, mapping, fp, ap

    def _evaluate_prepared_batch(
        self,
        task_key: str,
        step: int,
        prepared: tuple[
            dict[str, jnp.ndarray | bool],
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray | None,
            jnp.ndarray | None,
            jnp.ndarray | None,
        ],
    ) -> dict[str, float]:
        if task_key not in self._loss_fn_more_loss:
            raise RuntimeError("JAX trainer has not been initialized.")
        more_loss = self._loss_fn_more_loss[task_key](
            self.models[task_key],
            self.lr.value(step),
            *prepared,
        )
        return {key: float(value) for key, value in more_loss.items()}

    def _save_checkpoint(self, step: int) -> None:
        """Save a JAX checkpoint and update the stable checkpoint pointer."""
        ckpt_path = Path(f"{self.save_ckpt}-{step}.jax")
        self._write_checkpoint(ckpt_path, step=step)
        log.info(f"Trained model has been saved to: {ckpt_path!s}")
        _link_checkpoint(ckpt_path, Path(f"{self.save_ckpt}.jax"))
        self._cleanup_old_checkpoints()
        # Write the pointer next to the checkpoint prefix, with a value relative
        # to that directory (basename only). The freeze entrypoint looks for the
        # pointer inside the folder it is given and resolves the value relative
        # to it, so a directory-valued save_ckpt would otherwise be unresolvable.
        ckpt_dir = Path(self.save_ckpt).parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "checkpoint", "w") as fp:
            fp.write(f"{Path(self.save_ckpt).name}.jax")

    def _save_full_validation_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        """Save a full-validation-selected JAX checkpoint."""
        del lr
        self._write_checkpoint(save_path, step=step)

    def _write_checkpoint(self, ckpt_path: Path, *, step: int) -> None:
        """Write a JAX checkpoint directory to an explicit path."""
        if self.multi_task:
            state = {
                "models": {
                    model_key: nnx.split(model)[1].to_pure_dict()
                    for model_key, model in self.models.items()
                }
            }
        else:
            _, single_state = nnx.split(self.models[DEFAULT_TASK_KEY])
            state = single_state.to_pure_dict()
        state = _drop_zero_size_array_leaves(state)
        if ckpt_path.is_dir():
            shutil.rmtree(ckpt_path)
        model_def_script_cpy = deepcopy(self.model_def_script)
        model_def_script_cpy["current_step"] = step
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "model_def_script")
        ) as checkpointer:
            checkpointer.save(
                ckpt_path.absolute(),
                ocp.args.Composite(
                    state=ocp.args.StandardSave(state),
                    model_def_script=ocp.args.JsonSave(model_def_script_cpy),
                ),
            )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoint directories beyond the retention limit."""
        if self.max_ckpt_keep <= 0:
            return
        ckpt_parent = Path(self.save_ckpt).parent
        ckpt_prefix = Path(self.save_ckpt).name
        checkpoints = []
        for path in ckpt_parent.glob(f"{ckpt_prefix}-*.jax"):
            if not path.is_dir() or path.is_symlink():
                continue
            step_text = path.name.removeprefix(f"{ckpt_prefix}-").removesuffix(".jax")
            if step_text.isdigit():
                checkpoints.append((int(step_text), path))
        for _, path in sorted(checkpoints)[: -self.max_ckpt_keep]:
            shutil.rmtree(path)


def _evaluate_model_dict(
    model: BaseModel,
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: jnp.ndarray | None,
    fp: jnp.ndarray | None,
    ap: jnp.ndarray | None,
) -> dict[str, jnp.ndarray]:
    model_dict_lower = model.call_common_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fp,
        ap,
    )
    model_dict = communicate_extended_output(
        model_dict_lower,
        model.model_output_def(),
        mapping,
        do_atomic_virial=False,
    )
    model_dict["atom_energy"] = model_dict["energy"]
    model_dict["energy"] = model_dict["energy_redu"]
    force = model_dict["energy_derv_r"].squeeze(-2)
    if force.ndim == 2 or (force.ndim == 3 and force.shape[-1] != 3):
        force = jnp.reshape(force, (force.shape[0], -1, 3))
    model_dict["force"] = force
    model_dict["virial"] = model_dict["energy_derv_c_redu"].squeeze(-2)
    return model_dict


def _match_label_shapes(
    model_dict: dict[str, jnp.ndarray],
    label_dict: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Match equivalent flattened model outputs to label tensor shapes."""
    force_hat = model_dict.get("force")
    force = label_dict.get("force")
    if (
        force_hat is not None
        and force is not None
        and force_hat.shape != force.shape
        and force_hat.size == force.size
    ):
        model_dict = dict(model_dict)
        model_dict["force"] = jnp.reshape(force_hat, force.shape)
    return model_dict


def share_jax_model_params(
    models: dict[str, BaseModel],
    shared_links: dict[str, Any],
    *,
    model_key_prob_map: dict[str, float],
    data_stat_protect: float = 1e-2,
    resume: bool = False,
) -> None:
    """Share JAX model parameters following ``preprocess_shared_params`` links."""
    apply_shared_links(
        models,
        shared_links,
        share_descriptor=_share_jax_descriptor,
        share_fitting=_share_jax_fitting,
        model_key_prob_map=model_key_prob_map,
        data_stat_protect=data_stat_protect,
        resume=resume,
        logger=log,
    )


def _share_jax_descriptor(
    link_model: BaseModel,
    link_type: str,
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    resume: bool,
) -> None:
    _share_descriptor_component(
        base_class,
        link_class,
        shared_level,
        model_prob=model_prob,
        resume=resume,
    )
    if shared_level == 0:
        set_descriptor_component(link_model, link_type, base_class)


def _share_jax_fitting(
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    protection: float,
    resume: bool,
) -> None:
    _share_fitting_component(
        base_class,
        link_class,
        shared_level,
        model_prob=model_prob,
        protection=protection,
        resume=resume,
    )


def _share_descriptor_component(
    base_class: Any,
    link_class: Any,
    shared_level: int,
    *,
    model_prob: float,
    resume: bool,
) -> None:
    if type(link_class) is not type(base_class):
        raise AssertionError("Only descriptors of the same type can share params!")
    if shared_level == 0:
        if not resume:
            _merge_descriptor_stats(base_class, link_class, model_prob)
        return
    if shared_level == 1 and hasattr(base_class, "type_embedding"):
        link_class.type_embedding = base_class.type_embedding
        return
    raise NotImplementedError(
        f"JAX shared_dict does not support descriptor shared_level={shared_level} "
        f"for {type(base_class).__name__}."
    )


def _merge_descriptor_stats(
    base_class: Any,
    link_class: Any,
    model_prob: float,
) -> None:
    from deepmd.dpmodel.utils.env_mat_stat import (
        merge_env_stat,
    )

    merge_env_stat(base_class, link_class, model_prob)
    for attr in (
        "se_atten",
        "seat",
        "se_ttebd",
        "repinit",
        "repinit_three_body",
        "repformers",
        "repflows",
    ):
        if hasattr(base_class, attr) and hasattr(link_class, attr):
            _merge_descriptor_stats(
                getattr(base_class, attr),
                getattr(link_class, attr),
                model_prob,
            )
    if hasattr(base_class, "descrpt_list") and hasattr(link_class, "descrpt_list"):
        for base_item, link_item in zip(
            base_class.descrpt_list,
            link_class.descrpt_list,
            strict=True,
        ):
            _merge_descriptor_stats(base_item, link_item, model_prob)


def _share_fitting_component(
    base_class: Any,
    link_class: Any,
    shared_level: int,
    *,
    model_prob: float,
    protection: float,
    resume: bool,
) -> None:
    if type(link_class) is not type(base_class):
        raise AssertionError("Only fitting nets of the same type can share params!")
    if shared_level != 0:
        raise NotImplementedError(
            f"JAX shared_dict does not support fitting_net shared_level={shared_level}."
        )
    if not resume:
        _merge_fitting_param_stats(
            base_class,
            link_class,
            model_prob=model_prob,
            protection=protection,
        )
    link_class.nets = base_class.nets
    for attr in (
        "fparam_avg",
        "fparam_inv_std",
        "aparam_avg",
        "aparam_inv_std",
        "default_fparam_tensor",
    ):
        if getattr(base_class, attr, None) is not None:
            setattr(link_class, attr, getattr(base_class, attr))


def _merge_fitting_param_stats(
    base_class: Any,
    link_class: Any,
    *,
    model_prob: float,
    protection: float,
) -> None:
    _merge_one_fitting_stat(
        base_class,
        link_class,
        name="fparam",
        avg_attr="fparam_avg",
        inv_std_attr="fparam_inv_std",
        numb_attr="numb_fparam",
        model_prob=model_prob,
        protection=protection,
    )
    _merge_one_fitting_stat(
        base_class,
        link_class,
        name="aparam",
        avg_attr="aparam_avg",
        inv_std_attr="aparam_inv_std",
        numb_attr="numb_aparam",
        model_prob=model_prob,
        protection=protection,
    )


def _merge_one_fitting_stat(
    base_class: Any,
    link_class: Any,
    *,
    name: str,
    avg_attr: str,
    inv_std_attr: str,
    numb_attr: str,
    model_prob: float,
    protection: float,
) -> None:
    if getattr(base_class, numb_attr, 0) <= 0:
        return
    base_stats = base_class.get_param_stats().get(name, [])
    link_stats = link_class.get_param_stats().get(name, [])
    if not base_stats or not link_stats:
        return
    if len(base_stats) != getattr(base_class, numb_attr):
        raise AssertionError(f"{name} statistics length mismatch!")
    merged = [
        base_stats[ii] + link_stats[ii] * model_prob
        for ii in range(getattr(base_class, numb_attr))
    ]
    avg = np.array([stat.compute_avg() for stat in merged], dtype=np.float64)
    inv_std = 1.0 / np.array(
        [stat.compute_std(protection=protection) for stat in merged],
        dtype=np.float64,
    )
    setattr(base_class, avg_attr, _as_backend_array(getattr(base_class, avg_attr), avg))
    setattr(
        base_class,
        inv_std_attr,
        _as_backend_array(getattr(base_class, inv_std_attr), inv_std),
    )
    base_class._param_stats[name] = merged


def _as_backend_array(reference: Any, value: np.ndarray) -> Any:
    import array_api_compat

    ref_value = getattr(reference, "value", reference)
    xp = array_api_compat.array_namespace(ref_value)
    return xp.asarray(
        value,
        dtype=ref_value.dtype,
        device=array_api_compat.device(ref_value),
    )


def _init_empty_state(params: Any) -> optax.EmptyState:
    """Initialize an empty Optax state without requiring optax.init_empty_state.

    Older Optax releases expose ``EmptyState`` but not the convenience helper
    ``init_empty_state``. Constructing the state locally keeps this transform
    compatible with the Optax versions selected by the current JAX/Flax pins.
    """
    del params
    return optax.EmptyState()


def _scale_by_global_learning_rate() -> optax.GradientTransformationExtraArgs:
    """Scale optimizer updates by the learning rate from the global step."""

    def update_fn(
        updates: Any,
        state: optax.EmptyState,
        params: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, optax.EmptyState]:
        del params
        learning_rate = kwargs["learning_rate"]
        updates = jax.tree_util.tree_map(
            lambda update: -learning_rate * update, updates
        )
        return updates, state

    return optax.GradientTransformationExtraArgs(_init_empty_state, update_fn)


def _legacy_optimizer_update(optimizer: Any, grads: Any, lr: float) -> None:
    """Run an NNX optimizer update with extra args on Flax before 0.11."""
    from flax.nnx.training.optimizer import (
        _opt_state_variables_to_state,
        _update_opt_state,
    )

    params = nnx.state(optimizer.model, optimizer.wrt)
    opt_state = _opt_state_variables_to_state(optimizer.opt_state)
    updates, new_opt_state = optimizer.tx.update(
        grads,
        opt_state,
        params,
        learning_rate=lr,
    )
    new_params = optax.apply_updates(params, updates)
    optimizer.step.value += 1
    nnx.update(optimizer.model, new_params)
    _update_opt_state(optimizer.opt_state, new_opt_state)


def _copy_matching_state_tree(
    target: Any,
    source: Any,
    *,
    random_fitting: bool,
    path: tuple[Any, ...] = (),
) -> Any:
    if isinstance(target, dict):
        if not isinstance(source, dict):
            return target
        return {
            key: _copy_matching_state_tree(
                value,
                source.get(key),
                random_fitting=random_fitting,
                path=(*path, key),
            )
            for key, value in target.items()
        }
    if source is None:
        return target
    if random_fitting and not any("descriptor" in str(part) for part in path):
        return target
    if _same_state_leaf(target, source):
        return source
    return target


def _same_state_leaf(target: Any, source: Any) -> bool:
    target_shape = getattr(target, "shape", None)
    source_shape = getattr(source, "shape", None)
    target_dtype = getattr(target, "dtype", None)
    source_dtype = getattr(source, "dtype", None)
    return (
        target_shape is not None
        and source_shape is not None
        and target_shape == source_shape
        and target_dtype == source_dtype
    )


def _link_checkpoint(source: Path, target: Path) -> None:
    """Point the stable checkpoint path to the latest checkpoint directory."""
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if platform.system() != "Windows":
        os.symlink(os.path.relpath(source, target.parent), target)
    else:
        shutil.copytree(source, target)


def prepare_input(
    *,  # enforce keyword-only arguments
    rcut: float,
    sel: list[int],
    coord: np.ndarray,
    atype: np.ndarray,
    box: np.ndarray | None = None,
    fparam: np.ndarray | None = None,
    aparam: np.ndarray | None = None,
    pair_excl: "PairExcludeMask | None" = None,
    conservative_nlist: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """Build extended coordinates and neighbor lists for a training batch."""
    nframes, nloc = atype.shape[:2]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if bb is not None:
        coord_normalized = normalize_coord(
            cc.reshape(nframes, nloc, 3),
            bb.reshape(nframes, 3, 3),
        )
    else:
        coord_normalized = cc.reshape(nframes, nloc, 3).copy()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, bb, rcut
    )
    if conservative_nlist:
        # DPA4 treats ``sel`` as an initial capacity rather than a truncation
        # contract.  Use the full extended-atom capacity so every in-cutoff
        # edge survives the dense JAX input boundary; padded entries remain
        # masked by the lower descriptor path.
        sel = [extended_coord.shape[1]] * len(sel)
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        # types will be distinguished in the lower interface,
        # so it doesn't need to be distinguished here
        distinguish_types=False,
        # model-level pair exclusion is a nlist-BUILD transform (decision
        # #18/A4); the lower consumes a pre-excluded nlist.
        pair_excl=pair_excl,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    return extended_coord, extended_atype, nlist, mapping, fp, ap


def convert_numpy_data_to_jax_data(
    numpy_data: dict[str, np.ndarray | np.floating],
) -> dict[str, jnp.ndarray | bool]:
    """Convert NumPy data to JAX data.

    Parameters
    ----------
    numpy_data : dict[str, np.ndarray | np.floating]
        NumPy data

    Returns
    -------
    jax_data
        JAX data
    """
    # numpy to jax
    jax_data = {
        kk: jnp.asarray(vv) if not kk.startswith("find_") else bool(vv.item())
        for kk, vv in numpy_data.items()
    }
    return jax_data
