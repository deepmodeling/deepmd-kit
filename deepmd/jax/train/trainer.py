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
    Callable,
    Mapping,
    Sequence,
)
from copy import (
    deepcopy,
)
from dataclasses import (
    replace,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

import numpy as np
import orbax.checkpoint as ocp

from deepmd.dpmodel.loss.ener import (
    EnergyLoss,
)
from deepmd.dpmodel.loss.ener_spin import (
    EnergySpinLoss,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.dpmodel.loss.property import (
    PropertyLoss,
)
from deepmd.dpmodel.model.transform_output import (
    communicate_extended_output,
)
from deepmd.dpmodel.train import (
    DEFAULT_TASK_KEY,
    AbstractTrainer,
    RankContext,
    TrainerConfig,
    TrainingMetricAccumulator,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
    build_checkpoint_stores,
    change_model_out_bias_by_task,
    resolve_step_schedule,
)
from deepmd.dpmodel.train.data import (
    add_training_data_requirements,
)
from deepmd.dpmodel.train.data_progress import (
    DATA_PROGRESS_CHECKPOINT_KEY,
    collect_training_data_progress,
    preserve_training_data_progress,
    restore_training_data_progress,
)
from deepmd.dpmodel.train.ema import (
    get_ema_checkpoint_prefix,
    get_ema_validation_log_path,
)
from deepmd.dpmodel.train.inference import (
    infer_env_defaults,
    scoped_env_defaults,
)
from deepmd.dpmodel.train.validation import (
    resolve_best_checkpoint_dir,
)
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
)
from deepmd.dpmodel.utils.learning_rate import (
    BaseLR,
    make_learning_rate_schedule,
)
from deepmd.dpmodel.utils.multi_task import (
    get_case_embd_config,
    sanitize_shared_links,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.jax.common import (
    as_pure,
    from_flat_state,
    iter_modules,
    replace_by_pure_dict,
    set_variable_value,
    to_flat_state,
    to_pure_dict,
)
from deepmd.jax.env import (
    OptState,
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
from deepmd.jax.optimizer import (
    update_optimizer,
)
from deepmd.jax.train.distributed import (
    DistributedTraining,
)
from deepmd.jax.train.parameter_activity import (
    parameter_activity,
)
from deepmd.jax.utils.checkpoint_state import (
    pack_checkpoint_state,
    restore_checkpoint_state,
)
from deepmd.jax.utils.finetune import (
    merged_initialization_source,
)
from deepmd.jax.utils.graph_batch import (
    prepare_graph_batches,
)
from deepmd.jax.utils.multi_task import (
    preprocess_shared_params,
    share_jax_model_params,
)
from deepmd.jax.utils.precision import (
    MatmulPrecisionPolicy,
)
from deepmd.jax.utils.serialization import (
    resolve_checkpoint_path,
    serialize_from_file,
)
from deepmd.utils.argcheck import (
    resolve_full_validation_start_step,
)
from deepmd.utils.data import (
    DataRequirementItem,
    has_data_requirement,
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
from deepmd.utils.stat_file import (
    StatFileSpec,
    open_stat_file,
    run_stat_on_chief,
    stat_file_specs_by_task,
)

log = logging.getLogger(__name__)


class PreparedBatch(NamedTuple):
    """Labels and fixed-shape model inputs consumed by a compiled training step."""

    labels: dict[str, Any]
    inputs: dict[str, Any]


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
        self.optimizer_params = deepcopy(jdata.get("optimizer", {"type": "Adam"}))
        self._portable_training: dict[str, Any] | None = None
        self.validating_param = jdata.get("validating", {}) or {}
        self.num_steps = int(self.training_param.get("numb_steps") or 0)
        self.start_step = 0

        self.model_def_script = jdata["model"]
        self.enable_tf32 = bool(self.training_param.get("enable_tf32", False))
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
        stat_file_specs = {}
        for model_key in self.model_keys:
            task_training = (
                self.training_param["data_dict"][model_key]
                if self.multi_task
                else self.training_param
            )
            stat_file_specs[model_key] = StatFileSpec(
                task_training.get("stat_file"),
                task_training.get("stat_file_mode", "update"),
            )
        self.stat_file_specs = stat_file_specs_by_task(
            stat_file_specs,
            self.model_keys,
        )

        with scoped_env_defaults(infer_env_defaults(self.validating_param)):
            if init_model is not None or restart is not None:
                checkpoint_path = init_model if init_model is not None else restart
                assert checkpoint_path is not None
                checkpoint_data = serialize_from_file(
                    checkpoint_path, training=restart is not None
                )
                checkpoint_multi_task = (
                    "model_dict" in checkpoint_data["model_def_script"]
                )
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
                source_models = self._deserialize_models(checkpoint_data)
                if init_model is not None:
                    self.models = {
                        key: self._copy_initial_model_state(
                            get_model(deepcopy(self.model_params_by_task[key])),
                            source_models[key],
                        )
                        for key in self.model_keys
                    }
                else:
                    self.models = source_models
                    self._restore_model_state(checkpoint_path)
                if restart is not None:
                    self._portable_training = checkpoint_data.get("training")
                    if self._portable_training is not None:
                        self.shared_links = self._portable_training.get(
                            "shared_links", self.shared_links
                        )
                        self._adopt_checkpoint_optimizer(
                            self._portable_training.get("optimizer", {}).get(
                                "config", self.optimizer_params
                            )
                        )
                    self.model_def_script = checkpoint_data["model_def_script"]
                    self.model_params_by_task = self._model_params_by_task(
                        self.model_def_script
                    )
                    self.start_step = int(
                        checkpoint_data.get("model_def_script", {}).get(
                            "current_step",
                            checkpoint_data.get("@variables", {}).get(
                                "current_step", 0
                            ),
                        )
                    )
                    if self._portable_training is not None:
                        self.start_step = int(self._portable_training["step"])
            else:
                self.models = {
                    model_key: get_model(deepcopy(self.model_params_by_task[model_key]))
                    for model_key in self.model_keys
                }
                if self.multi_task:
                    do_case_embd, case_indices = get_case_embd_config(
                        self.model_def_script
                    )
                    if do_case_embd:
                        for key, model in self.models.items():
                            model.set_case_embd(case_indices[key])
            self.matmul_precision = MatmulPrecisionPolicy(enable_tf32=self.enable_tf32)
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]
        self.distributed = DistributedTraining(
            self.training_param, devices=self._training_devices()
        )

        learning_rate_param = jdata["learning_rate"]
        self.learning_rate_params = deepcopy(learning_rate_param)
        self.lr = self._get_lr_and_coef(learning_rate_param) if self.num_steps else None
        self.losses = self._build_losses(jdata, learning_rate_param)
        self.data_requirements_by_task = {
            model_key: list(self.losses[model_key].label_requirement)
            for model_key in self.model_keys
        }
        self.pair_filter_enabled_by_task: dict[str, bool] = {}
        self.training_data_by_task: dict[str, Any] = {}
        for model_key, requirements in self.data_requirements_by_task.items():
            # Hessians are expensive and are only exposed by models whose task
            # loss requests the corresponding label. Enable each multi-task
            # branch independently so unrelated branches keep normal outputs.
            if has_data_requirement(requirements, "hessian"):
                enable_hessian = getattr(self.models[model_key], "enable_hessian", None)
                if not callable(enable_hessian):
                    raise RuntimeError(
                        f"Model {type(self.models[model_key]).__name__} does not "
                        "support Hessian supervision."
                    )
                enable_hessian()
                self.model_params_by_task[model_key]["hessian_mode"] = True
        self.loss = self.losses if self.multi_task else self.losses[DEFAULT_TASK_KEY]

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
        self.enable_ema = bool(tr_data.get("enable_ema", False))
        self.ema_decay = float(tr_data.get("ema_decay", 0.999))
        self.ema_models: dict[str, BaseModel] = {}
        self.ema_validation_state: dict[str, Any] = {}
        self.ema_full_validator: Any | None = None
        self.numb_fparam = (
            {key: model.get_dim_fparam() for key, model in self.models.items()}
            if self.multi_task
            else self.models[DEFAULT_TASK_KEY].get_dim_fparam()
        )

        self.frz_model = None
        self.ckpt_meta = None
        self.model_type = None
        self.optimizer: nnx.Optimizer | None = None
        self._active_parameters: dict[str, Any] = {}
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
            metric_accumulator=(
                TrainingMetricAccumulator(
                    {
                        key: loss.training_metric_names
                        for key, loss in self.losses.items()
                    }
                )
                if tr_data.get("disp_avg", False)
                else None
            ),
        )

    def _training_devices(self) -> list[jax.Device]:
        """Return the local devices of data-parallel training.

        Every device reads its own graph batch, so a model on a dense neighbor
        list trains on one device.
        """
        devices = jax.local_devices()
        if len(devices) > 1 and not all(
            model.uses_full_neighbor_list() for model in self.models.values()
        ):
            log.info(
                "A model on a dense neighbor list trains on one device: %s.",
                devices[0],
            )
            return devices[:1]
        return devices

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
            models = {
                model_key: BaseModel.deserialize(
                    model_data["model"]["model_dict"][model_key]
                )
                for model_key in model_data["model_def_script"]["model_dict"]
            }
            params = model_data["model_def_script"]["model_dict"]
        else:
            models = {DEFAULT_TASK_KEY: BaseModel.deserialize(model_data["model"])}
            params = {DEFAULT_TASK_KEY: model_data["model_def_script"]}
        for key, model in models.items():
            minimum = params[key].get("_min_nbor_dist")
            if key == DEFAULT_TASK_KEY:
                minimum = model_data.get("min_nbor_dist", minimum)
            if minimum is not None:
                model.min_nbor_dist = float(minimum)
        return models

    def _restore_model_state(self, checkpoint_path: str) -> None:
        """Preserve native runtime variables, including random rotation state."""
        state_path = resolve_checkpoint_path(checkpoint_path) / "state"
        if not state_path.is_dir():
            return
        model_states = {key: nnx.state(model) for key, model in self.models.items()}
        target = (
            {
                "models": {
                    key: to_pure_dict(state) for key, state in model_states.items()
                }
            }
            if self.multi_task
            else to_pure_dict(model_states[DEFAULT_TASK_KEY])
        )
        with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
            restored = checkpointer.restore(
                state_path.absolute(),
                args=ocp.args.StandardRestore(pack_checkpoint_state(target)),
            )
        restored = restore_checkpoint_state(restored, target)
        restored_models = (
            restored["models"] if self.multi_task else {DEFAULT_TASK_KEY: restored}
        )
        for key, state in model_states.items():
            replace_by_pure_dict(state, restored_models[key])
            nnx.update(self.models[key], state)
        metadata_path = state_path.parent / "training_metadata"
        if self.restart is not None and metadata_path.is_dir():
            with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
                metadata = checkpointer.restore(
                    metadata_path.absolute(), args=ocp.args.JsonRestore()
                )
            self._adopt_checkpoint_optimizer(metadata["optimizer"])
            self.shared_links = metadata.get("shared_links", self.shared_links)

    def _adopt_checkpoint_optimizer(self, config: Mapping[str, Any]) -> None:
        """Continue a restart with the optimizer settings of its checkpoint.

        The saved moments belong to that optimizer, as a PyTorch restart
        restores its parameter groups with the state.
        """
        config = deepcopy(dict(config))
        if config != self.optimizer_params:
            log.warning(
                "The restart continues with the optimizer settings of the "
                "checkpoint, %s, instead of the input settings %s.",
                config,
                self.optimizer_params,
            )
        self.optimizer_params = config

    def _get_lr_and_coef(self, lr_param: dict[str, Any]) -> BaseLR:
        """Construct the schema-selected shared learning-rate schedule."""
        return make_learning_rate_schedule(lr_param, self.num_steps)

    def _build_losses(
        self,
        jdata: dict[str, Any],
        learning_rate_param: dict[str, Any],
    ) -> dict[str, Loss]:
        losses: dict[str, Loss] = {}
        for model_key in self.model_keys:
            loss_param = deepcopy(
                jdata["loss_dict"][model_key]
                if self.multi_task
                else jdata.get("loss", {})
            )
            loss_param["starter_learning_rate"] = learning_rate_param["start_lr"]
            loss_type = loss_param.get("type", "ener")
            loss_class = {
                "ener": EnergyLoss,
                "ener_spin": EnergySpinLoss,
                "property": PropertyLoss,
            }.get(loss_type)
            if loss_class is None:
                raise RuntimeError("unknown loss type " + loss_type)
            if loss_type == "property":
                model = self.models[model_key]
                loss_param.update(
                    task_dim=model.get_task_dim(),
                    var_name=model.get_var_name(),
                    intensive=model.get_intensive(),
                )
            losses[model_key] = loss_class.get_loss(loss_param)
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
            symm_virial=getattr(self.losses[DEFAULT_TASK_KEY], "symm_virial", False),
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
            precision_policy=self.matmul_precision,
        )

    def _create_ema_full_validator(self, validation_data: Any) -> Any | None:
        """Build an independent validator over the EMA parameter trajectory."""
        if not self.enable_ema or not self._is_validation_requested(
            self.validating_param, "full_validation_ema"
        ):
            return None
        self._raise_if_full_validation_unsupported(validation_data)
        from deepmd.jax.train.validation import (
            JAXFullValidator,
        )

        params = dict(self.validating_param, full_validation=True)
        return JAXFullValidator(
            symm_virial=getattr(self.losses[DEFAULT_TASK_KEY], "symm_virial", False),
            validating_params=params,
            validation_data=validation_data,
            model=self.ema_models[DEFAULT_TASK_KEY],
            state_store=self.ema_validation_state,
            num_steps=self.num_steps,
            rank=self.rank_context.rank,
            restart_training=self.restart_training,
            checkpoint_dir=resolve_best_checkpoint_dir(params, self.save_ckpt),
            full_val_file=get_ema_validation_log_path(
                params.get("full_val_file", "val.log")
            ),
            best_checkpoint_prefix="best_ema.ckpt",
            precision_policy=self.matmul_precision,
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

        if not isinstance(self.loss, (EnergyLoss, EnergySpinLoss)):
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
        self.training_data_by_task = dict(train_data_by_task)
        for model_key in self.model_keys:
            requirements = self.data_requirements_by_task[model_key]
            requirements.extend(
                requirement
                for requirement in get_additional_data_requirement(
                    self.models[model_key]
                )
                if not has_data_requirement(requirements, requirement.key)
            )
            task_params = (
                self.training_param["data_dict"][model_key]
                if self.multi_task
                else self.training_param
            )
            self.pair_filter_enabled_by_task[model_key] = (
                add_training_data_requirements(
                    train_data_by_task[model_key],
                    task_params.get("training_data", {}),
                    requirements,
                    lambda data, items: data.add_data_requirements(items),
                )
            )
            if valid_data_by_task[model_key] is not None:
                valid_data_by_task[model_key].add_data_requirements(
                    self.data_requirements_by_task[model_key]
                )

        schedule = resolve_step_schedule(
            self.training_param,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
            training_data=train_data_by_task,
            epoch_length=lambda key: self.distributed.epoch_length(
                train_data_by_task[key]
            ),
            rank=self.rank_context.rank,
        )
        self.num_steps = schedule.num_steps
        self.model_prob = schedule.model_prob
        self.lr = self._get_lr_and_coef(self.learning_rate_params)
        self.trainer_config = replace(self.trainer_config, num_steps=self.num_steps)
        self.checkpoint_store, self.ema_checkpoint_store = build_checkpoint_stores(
            self.training_param,
            num_steps=self.num_steps,
            ema_prefix=get_ema_checkpoint_prefix(self.save_ckpt),
            rank=self.rank_context.rank,
            suffix=".jax",
        )

        for model_key in self.model_keys:
            self._sample_funcs[model_key] = self._make_sample_func(
                train_data_by_task[model_key],
                self.model_params_by_task[model_key].get("data_stat_nbatch", 10),
                filter_enabled=self.pair_filter_enabled_by_task[model_key],
            )

        synchronize_model_state = False
        merge_shared_statistics = False
        if self.init_model is None and self.restart is None:
            for model_key in self.model_keys:
                finetune_has_new_type = (
                    self.finetune_model is not None
                    and self.finetune_links is not None
                    and model_key in self.finetune_links
                    and self.finetune_links[model_key].get_has_new_type()
                )
                if self.finetune_model is None or finetune_has_new_type:
                    self._run_on_chief(
                        lambda _model_key=model_key: self._initialize_stat(_model_key),
                        operation=f"statistics initialization for task {model_key!r}",
                    )
                    synchronize_model_state = True
                    merge_shared_statistics = True

        if self.finetune_model is not None:
            self._run_on_chief(
                self._apply_finetune,
                operation="fine-tuning initialization",
            )
            synchronize_model_state = True

        self._synchronize_initial_model_state(
            state_changed=synchronize_model_state,
            merge_shared_statistics=merge_shared_statistics,
        )

        from deepmd.jax.optimizer import (
            build_optimizer,
        )

        tx = build_optimizer(
            self.models,
            self.optimizer_params,
            matrix_sharding=(
                self.distributed.data_sharding()
                if self.distributed.is_distributed
                else None
            ),
        )
        self.optimizer = nnx.Optimizer(self.models, tx, wrt=nnx.Param)
        self.optimizer.nonfinite_gradient_seen = OptState(jnp.asarray(False))
        self._active_parameters = _task_parameter_masks(self.models)
        for model_key in self.model_keys:
            (
                self._train_step_impls[model_key],
                self._loss_fn_more_loss[model_key],
            ) = self._make_step_functions(self.losses[model_key], model_key)
        restored_optimizer = self._restore_optimizer_state()
        restored_ema = False
        if self.enable_ema:
            self.ema_models = nnx.clone(self.models)
            for model in self.ema_models.values():
                model.eval()
            restored_ema = self._restore_ema_state()
        if self._portable_training is not None:
            from deepmd.jax.utils.checkpoint import (
                import_training_state,
            )

            portable = dict(self._portable_training)
            if restored_optimizer:
                portable.pop("optimizer", None)
            if restored_ema:
                portable.pop("ema", None)
            ema_validation = import_training_state(
                self.models,
                self.optimizer,
                portable,
                ema_models=self.ema_models if self.enable_ema else None,
            )
            if not restored_ema:
                self.ema_validation_state = ema_validation
        self.distributed.place_state(
            self.models,
            self.optimizer,
            self.ema_models if self.enable_ema else None,
            active=self._active_parameters,
        )
        self._restore_data_progress()
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]
        self.full_validator = self._create_full_validator(
            validating_params=self.validating_param,
            validation_data=valid_data_by_task[DEFAULT_TASK_KEY]
            if not self.multi_task
            else None,
        )
        self.ema_full_validator = self._create_ema_full_validator(
            valid_data_by_task[DEFAULT_TASK_KEY] if not self.multi_task else None
        )

    def _restore_optimizer_state(self) -> bool:
        """Restore optimizer moments and counts for native JAX restarts."""
        if self.restart is None:
            return False
        training_path = resolve_checkpoint_path(self.restart) / "training_state"
        if not training_path.is_dir():
            if not self._portable_training:
                log.warning(
                    "Checkpoint %s has no optimizer state; optimizer moments start at zero.",
                    self.restart,
                )
            return False
        state = nnx.state(self.optimizer, OptState)
        target = to_pure_dict(state)
        with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
            restored = checkpointer.restore(
                training_path.absolute(),
                args=ocp.args.StandardRestore(pack_checkpoint_state(target)),
            )
        replace_by_pure_dict(state, restore_checkpoint_state(restored, target))
        nnx.update(self.optimizer, state)
        return True

    def _restore_data_progress(self) -> None:
        """Continue the training data streams and task selection of a restart.

        The checkpoint stores the next-batch cursor of every data source that
        supports one and the DeePMD random state that selects tasks and
        systems. ``training.reset_data_progress`` keeps the freshly initialized
        streams instead.
        """
        if self.restart is None:
            return
        metadata_path = resolve_checkpoint_path(self.restart) / "training_metadata"
        if not metadata_path.is_dir():
            return
        with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
            metadata = checkpointer.restore(metadata_path.absolute())
        progress = metadata.get(DATA_PROGRESS_CHECKPOINT_KEY)
        if progress is None:
            return
        if self.training_param.get("reset_data_progress", False):
            log.warning(
                "training.reset_data_progress is true. Model, optimizer and step "
                "are restored, but training data order and selection RNG restart "
                "from their newly initialized state; batches may repeat."
            )
            return
        restore_training_data_progress(self.training_data_by_task, progress)

    def _restore_ema_state(self) -> bool:
        """Restore the EMA trajectory and its independent validation records."""
        if self.restart is None:
            return False
        checkpoint_path = resolve_checkpoint_path(self.restart)
        ema_path = checkpoint_path / "ema_state"
        if not ema_path.is_dir():
            if not (self._portable_training or {}).get("ema"):
                log.warning(
                    "Checkpoint %s has no EMA state; EMA starts from the restored model.",
                    self.restart,
                )
            return False
        state = nnx.state(self.ema_models)
        target = to_pure_dict(state)
        with ocp.Checkpointer(ocp.StandardCheckpointHandler()) as checkpointer:
            restored = checkpointer.restore(
                ema_path.absolute(),
                args=ocp.args.StandardRestore(pack_checkpoint_state(target)),
            )
        replace_by_pure_dict(state, restore_checkpoint_state(restored, target))
        nnx.update(self.ema_models, state)
        with ocp.Checkpointer(ocp.JsonCheckpointHandler()) as checkpointer:
            metadata = checkpointer.restore(
                (checkpoint_path / "training_metadata").absolute(),
                args=ocp.args.JsonRestore(),
            )
        self.ema_validation_state = metadata["ema_validation_state"]
        return True

    @staticmethod
    def _make_sample_func(
        train_data: DeepmdDataSystem,
        data_stat_nbatch: int,
        filter_enabled: bool = False,
    ) -> Any:
        @functools.lru_cache
        def sample() -> list[dict[str, Any]]:
            # Statistics must not advance the training cursors or the
            # selection RNG.
            with preserve_training_data_progress({"statistics": train_data}):
                stat_data = make_stat_input(
                    train_data, data_stat_nbatch, filter_enabled
                )
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
        pretrained_data = serialize_from_file(self.finetune_model, training=False)
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

    def _initialize_stat(self, model_key: str) -> None:
        """Initialize one model from its scoped statistics cache."""
        with open_stat_file(
            self.stat_file_specs[model_key],
        ) as stat_file_path:
            self.models[model_key].atomic_model.compute_or_load_stat(
                self._sample_funcs[model_key],
                stat_file_path=stat_file_path,
            )

    def _run_on_chief(
        self,
        action: Callable[[], None],
        *,
        operation: str,
    ) -> None:
        """Run one statistics action on rank 0 and propagate failure status."""
        synchronize_failure: Callable[[bool], bool] | None = None
        if self.rank_context.world_size > 1:
            from jax.experimental import (
                multihost_utils,
            )

            def broadcast_failure(failed: bool) -> bool:
                return bool(
                    np.asarray(
                        multihost_utils.broadcast_one_to_all(
                            np.asarray(failed, dtype=np.bool_),
                            is_source=self.rank_context.is_chief,
                        )
                    ).item()
                )

            synchronize_failure = broadcast_failure

        run_stat_on_chief(
            action,
            is_chief=self.rank_context.is_chief,
            synchronize_failure=synchronize_failure,
            operation=operation,
        )

    def _broadcast_model_states(self) -> None:
        """Broadcast complete model states from rank 0 to every JAX process."""
        if self.rank_context.world_size <= 1:
            return
        from jax.experimental import (
            multihost_utils,
        )

        for model_key in self.model_keys:
            _, state = nnx.split(self.models[model_key])
            state = multihost_utils.broadcast_one_to_all(
                to_pure_dict(state),
                is_source=self.rank_context.is_chief,
            )
            nnx.update(self.models[model_key], state)

    def _synchronize_initial_model_state(
        self,
        *,
        state_changed: bool,
        merge_shared_statistics: bool,
    ) -> None:
        """Merge, broadcast, and bind initial multi-task model state."""
        has_shared_parameters = self.multi_task and bool(self.shared_links)
        if has_shared_parameters and merge_shared_statistics:
            self._run_on_chief(
                lambda: self._share_model_params(resume=False),
                operation="shared statistics merge",
            )
            state_changed = True
        if state_changed:
            self._broadcast_model_states()
        if has_shared_parameters:
            self._share_model_params(resume=True)

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
    def _copy_initial_model_state(
        target_model: BaseModel, source_model: BaseModel
    ) -> BaseModel:
        """Load matching model state while preserving target runtime configuration.

        Initialization resets the random stream and optimizer step. A plain
        checkpoint may initialize freshly constructed LoRA factors; every
        other mathematical state slot must exist with the expected shape.
        """
        source_model = merged_initialization_source(source_model)
        target = dict(to_flat_state(nnx.state(target_model)))
        source = dict(to_flat_state(nnx.state(source_model)))
        if target_model.get_type_map() != source_model.get_type_map():
            raise ValueError(
                "Init-model requires the same type_map; use fine-tuning to remap types."
            )
        adapter_fields = {
            "A_by_l",
            "B_by_l",
            "A_m0",
            "B_m0",
            "A_m",
            "B_m",
            "lora_scaling",
        }
        unexpected = source.keys() - target.keys()
        if unexpected:
            names = sorted(".".join(map(str, path)) for path in unexpected)
            raise ValueError(f"Init-model checkpoint has unexpected state: {names}.")
        for path, variable in target.items():
            if path[-1] == "rotation_key":
                continue
            name = ".".join(map(str, path))
            if path not in source:
                if adapter_fields.intersection(path):
                    continue
                raise ValueError(f"Init-model checkpoint is missing state {name!r}.")
            value = source[path][...]
            expected = variable[...]
            if value.shape != expected.shape:
                raise ValueError(
                    f"Init-model state {name!r} expects shape {expected.shape}, "
                    f"got {value.shape}; use fine-tuning for architecture changes."
                )
            target[path] = variable.replace(
                value=jnp.asarray(value, dtype=expected.dtype)
            )
        nnx.update(target_model, from_flat_state(target))
        target_model.min_nbor_dist = source_model.get_min_nbor_dist()
        return target_model

    @staticmethod
    def _copy_finetune_state(
        target_model: BaseModel,
        source_model: BaseModel,
        *,
        random_fitting: bool,
    ) -> BaseModel:
        source_model = merged_initialization_source(source_model)
        graphdef, target_state = nnx.split(target_model)
        _, source_state = nnx.split(source_model)
        copied = _copy_matching_state_tree(
            to_pure_dict(target_state),
            to_pure_dict(source_state),
            random_fitting=random_fitting,
        )
        replace_by_pure_dict(target_state, copied)
        return nnx.merge(graphdef, target_state)

    def _make_step_functions(self, loss_obj: Loss, task_key: str) -> tuple[Any, Any]:
        ema_decay = self.ema_decay
        precision = self.matmul_precision
        gradient_max_norm = self.training_param.get("gradient_max_norm", 0.0)
        distributed = self.distributed

        def loss_fn(
            models: dict[str, BaseModel],
            lr: float,
            batch: PreparedBatch,
        ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
            with precision.applied(training=True):
                model_dict = _evaluate_inputs(models[task_key], batch.inputs)
            return distributed.evaluate_loss(
                loss_obj,
                learning_rate=lr,
                natoms=_batch_atom_capacity(batch.inputs),
                model_dict=model_dict,
                label_dict=batch.labels,
            )

        @nnx.jit
        def loss_fn_more_loss(
            model: BaseModel,
            lr: float,
            batch: PreparedBatch,
        ) -> dict[str, jnp.ndarray]:
            with precision.applied(training=False):
                model_dict = _evaluate_inputs(model, batch.inputs)
            _, more_loss = distributed.evaluate_loss(
                loss_obj,
                learning_rate=lr,
                natoms=_batch_atom_capacity(batch.inputs),
                model_dict=model_dict,
                label_dict=batch.labels,
            )
            return more_loss

        @distributed.jit_step
        def train_step(
            models: dict[str, BaseModel],
            optimizer: nnx.Optimizer,
            lr: float,
            batch: PreparedBatch,
            active: Any,
            ema_models: dict[str, BaseModel] | None,
        ) -> dict[str, jnp.ndarray]:
            # One rotation realization is shared by every derivative and
            # reference-energy evaluation within the optimization step.
            for _, module in iter_modules(models[task_key]):
                advance_random_state = getattr(module, "advance_random_state", None)
                if advance_random_state is not None:
                    advance_random_state()
            (_, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
                models, lr, batch
            )
            structural = parameter_activity(
                lambda current: loss_fn(current, lr, batch)[0], models
            )
            active = jax.tree.map(jnp.logical_and, active, structural)
            grads = distributed.constrain_gradients(grads)
            if gradient_max_norm > 0:
                total_norm = _global_gradient_norm(grads)
                set_variable_value(
                    optimizer.nonfinite_gradient_seen,
                    optimizer.nonfinite_gradient_seen[...] | ~jnp.isfinite(total_norm),
                )
                grads = _clip_gradients(grads, gradient_max_norm, total_norm=total_norm)
            update_optimizer(optimizer, models, grads, learning_rate=lr, active=active)
            if ema_models is not None:
                parameters = nnx.state(models, nnx.Param)
                shadow = nnx.state(ema_models, nnx.Param)
                shadow = jax.tree_util.tree_map(
                    lambda average, value: (
                        average + (1.0 - ema_decay) * (value - average)
                    ),
                    shadow,
                    parameters,
                )
                nnx.update(ema_models, shadow)
            return metrics

        return train_step, loss_fn_more_loss

    def select_task(self, tasks: TrainingTaskCollection) -> TrainingTask:
        """Select a task using DeePMD's seeded random helper."""
        from deepmd.utils import random as dp_random

        return tasks.select(dp_random.choice)

    def get_training_batch(self, task: TrainingTask) -> tuple[dict[str, Any], ...]:
        """Fetch the next globally valid group of per-device batches."""
        return self.distributed.get_batch_group(
            task.training_data,
            filter_enabled=self.pair_filter_enabled_by_task[task.key],
            task_key=task.key,
        )

    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        """Run one JAX optimizer step."""
        task_key = task.key
        if self.optimizer is None or task_key not in self._train_step_impls:
            raise RuntimeError("JAX trainer has not been initialized.")
        prepared = self._prepare_batch(
            self.get_training_batch(task), task_key, observe_contacts=True
        )
        self.models[task_key].train()
        metrics = self._train_step_impls[task_key](
            self.models,
            self.optimizer,
            self.lr.value(step),
            prepared,
            self._active_parameters[task_key],
            self.ema_models if self.enable_ema else None,
        )
        return TrainStepResult(
            task_key=task_key,
            step=step,
            payload=prepared,
            train_results=metrics,
        )

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
            else self._prepare_batch(
                self.get_training_batch(task),
                task.key,
            )
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
        results = []
        weights = []
        for _ in range(task.valid_numb_batch):
            batches = self.distributed.get_batch_group(
                task.validation_data, task_key=task.key
            )
            weights.append(int(np.count_nonzero(batches[0]["atype"] >= 0)))
            results.append(
                self._evaluate_prepared_batch(
                    task.key,
                    step,
                    self._prepare_batch(batches, task.key, bucket_subdivisions=1),
                )
            )
        return {
            key: sum(
                result[key] * weight
                for result, weight in zip(results, weights, strict=True)
            )
            / sum(weights)
            for key in results[0]
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
        self._broadcast_model_states()

    def run_full_validation(
        self,
        *,
        step: int,
        display_step: int,
        learning_rate: float,
    ) -> None:
        """Run optional full validation for one step."""
        if self.ema_full_validator is not None and self.ema_full_validator.should_run(
            display_step
        ):
            self._synchronize_ema_buffers()
        for validator, save_checkpoint in (
            (self.full_validator, self._save_full_validation_checkpoint),
            (self.ema_full_validator, self._save_ema_validation_checkpoint),
        ):
            if validator is not None:
                validator.run(
                    step_id=display_step,
                    display_step=display_step,
                    lr=learning_rate,
                    save_checkpoint=save_checkpoint,
                )

    def _synchronize_ema_buffers(self) -> None:
        """Combine EMA parameters with the live model's current non-parameter state."""
        if self.ema_models:
            nnx.update(self.ema_models, nnx.state(self.models, nnx.Not(nnx.Param)))

    def _prepare_batch(
        self,
        batch_data: dict[str, Any] | Sequence[dict[str, Any]],
        task_key: str,
        *,
        bucket_subdivisions: int = 8,
        observe_contacts: bool = False,
    ) -> PreparedBatch:
        """Convert original device batches into labels and fixed-shape inputs.

        Parameters
        ----------
        batch_data : dict or sequence of dict
            One NumPy batch, or one batch per logical device.
        task_key : str
            Task whose model and data requirements apply.
        bucket_subdivisions : int
            Number of node/edge capacity buckets per power-of-two interval.
        observe_contacts : bool
            Record the contacts of the batch in the contact clamps of the
            model before it is evaluated, which a training batch does.

        Returns
        -------
        PreparedBatch
            Labels and model inputs placed on the training devices.
        """
        model = self.models[task_key]
        batches = (
            (batch_data,) if isinstance(batch_data, Mapping) else tuple(batch_data)
        )
        if model.uses_full_neighbor_list():
            labels, inputs = prepare_graph_batches(
                tuple(normalize_batch(batch) for batch in batches),
                rcut=model.get_rcut(),
                ntypes=model.get_ntypes(),
                data_requirements=self.data_requirements_by_task[task_key],
                pair_exclude_types=model.atomic_model.pair_exclude_types,
                bucket_subdivisions=bucket_subdivisions,
            )
            if observe_contacts:
                # The host graph holds every device batch, so the record
                # covers all of them before the step reads the edges.
                model.atomic_model.observe_contacts(
                    NeighborGraph(
                        n_node=inputs["n_node"],
                        edge_index=inputs["edge_index"],
                        edge_vec=inputs["edge_vec"],
                        edge_mask=inputs["edge_mask"],
                    ),
                    inputs["atype"],
                )
            return PreparedBatch(
                *self.distributed.prepare_batch(
                    labels,
                    inputs,
                    data_requirements=self.data_requirements_by_task[task_key],
                )
            )
        if len(batches) != 1:
            raise NotImplementedError(
                "Per-device JAX batches require a model with sparse graph training, such as DPA4."
            )
        jax_data = convert_numpy_data_to_jax_data(normalize_batch(batches[0]))
        extended_coord, extended_atype, nlist, mapping, fp, ap = prepare_input(
            rcut=model.get_rcut(),
            sel=model.get_sel(),
            coord=jax_data["coord"],
            atype=jax_data["atype"],
            box=jax_data.get("box") if jax_data.get("find_box", True) else None,
            fparam=jax_data.get("fparam"),
            aparam=jax_data.get("aparam"),
            pair_excl=model.atomic_model.pair_excl,
        )
        inputs = {
            "extended_coord": extended_coord,
            "extended_atype": extended_atype,
            "nlist": nlist,
            "mapping": mapping,
            "fp": fp,
            "ap": ap,
            "spin": jax_data.get("spin"),
            "charge_spin": jax_data.get("charge_spin"),
        }
        return PreparedBatch(*self.distributed.prepare_batch(jax_data, inputs))

    def _evaluate_prepared_batch(
        self,
        task_key: str,
        step: int,
        prepared: PreparedBatch,
    ) -> dict[str, float]:
        if task_key not in self._loss_fn_more_loss:
            raise RuntimeError("JAX trainer has not been initialized.")
        self.models[task_key].eval()
        more_loss = self._loss_fn_more_loss[task_key](
            self.models[task_key], self.lr.value(step), prepared
        )
        return {key: float(value) for key, value in more_loss.items()}

    def _save_checkpoint(self, step: int) -> None:
        """Save a JAX checkpoint and update the stable checkpoint pointer."""
        ckpt_path = self.checkpoint_store.path_for(step)
        self._write_checkpoint(ckpt_path, step=step)
        log.info(f"Trained model has been saved to: {ckpt_path!s}")
        _link_checkpoint(ckpt_path, Path(f"{self.save_ckpt}.jax"))
        self.checkpoint_store.prune(ckpt_path)
        # Write the pointer next to the checkpoint prefix, with a value relative
        # to that directory (basename only). The freeze entrypoint looks for the
        # pointer inside the folder it is given and resolves the value relative
        # to it, so a directory-valued save_ckpt would otherwise be unresolvable.
        ckpt_dir = Path(self.save_ckpt).parent
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "checkpoint", "w") as fp:
            fp.write(f"{Path(self.save_ckpt).name}.jax")
        if self.enable_ema:
            ema_path = self.ema_checkpoint_store.path_for(step)
            self._write_checkpoint(ema_path, step=step, ema=True)
            _link_checkpoint(ema_path, Path(f"{self.ema_checkpoint_store.prefix}.jax"))
            self.ema_checkpoint_store.prune(ema_path)

    def _save_full_validation_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        """Save a full-validation-selected JAX checkpoint."""
        del lr
        self._write_checkpoint(save_path, step=step, merge_adapters=True)

    def _save_ema_validation_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        """Save an EMA model selected by its independent full validator."""
        del lr
        self._write_checkpoint(save_path, step=step, ema=True, merge_adapters=True)

    def _write_checkpoint(
        self,
        ckpt_path: Path,
        *,
        step: int,
        ema: bool = False,
        merge_adapters: bool = False,
    ) -> None:
        """Write a JAX checkpoint directory to an explicit path."""
        if self.optimizer is not None and bool(
            self.optimizer.nonfinite_gradient_seen[...]
        ):
            nonfinite = [
                ".".join(map(str, path))
                for path, variable in to_flat_state(nnx.state(self.models, nnx.Param))
                if not np.isfinite(np.asarray(as_pure(variable))).all()
            ]
            detail = ", ".join(nonfinite) or "current parameters are finite"
            raise RuntimeError(
                "A non-finite gradient norm occurred during training; "
                f"checkpoint {ckpt_path} was not written. Parameter state: {detail}."
            )
        self._synchronize_ema_buffers()
        models = self.ema_models if ema else self.models
        merged = False
        if merge_adapters:
            from deepmd.jax.descriptor.lora import (
                has_lora,
                serialize_merged_model,
            )

            merged = has_lora(models)
            if merged:
                models = {
                    key: BaseModel.deserialize(serialize_merged_model(model))
                    for key, model in models.items()
                }
        if self.multi_task:
            state = {
                "models": {
                    model_key: to_pure_dict(nnx.split(model)[1])
                    for model_key, model in models.items()
                }
            }
        else:
            _, single_state = nnx.split(models[DEFAULT_TASK_KEY])
            state = to_pure_dict(single_state)
        if ckpt_path.is_dir():
            shutil.rmtree(ckpt_path)
        model_def_script_cpy = deepcopy(self.model_def_script)
        model_def_script_cpy["current_step"] = step
        for key, model in self.models.items():
            minimum = model.get_min_nbor_dist()
            if minimum is not None:
                params = (
                    model_def_script_cpy["model_dict"][key]
                    if self.multi_task
                    else model_def_script_cpy
                )
                params["_min_nbor_dist"] = float(minimum)
        if merged:
            model_def_script_cpy.pop("lora", None)
            for params in model_def_script_cpy.get("model_dict", {}).values():
                params.pop("lora", None)
        checkpoint_items = {
            "state": ocp.args.StandardSave(pack_checkpoint_state(state)),
            "model_def_script": ocp.args.JsonSave(model_def_script_cpy),
        }
        if not ema and not merged:
            checkpoint_items["training_state"] = ocp.args.StandardSave(
                pack_checkpoint_state(to_pure_dict(nnx.state(self.optimizer, OptState)))
            )
            checkpoint_items["training_metadata"] = ocp.args.JsonSave(
                {
                    "optimizer": self.optimizer_params,
                    "shared_links": sanitize_shared_links(self.shared_links),
                    "lr": self.lr.value(max(step - 1, 0)),
                    "gradient_max_norm": self.training_param.get(
                        "gradient_max_norm", 0.0
                    ),
                    "ema_decay": self.ema_decay,
                    "ema_validation_state": self.ema_validation_state,
                    DATA_PROGRESS_CHECKPOINT_KEY: collect_training_data_progress(
                        self.training_data_by_task
                    ),
                }
            )
            if self.enable_ema:
                checkpoint_items["ema_state"] = ocp.args.StandardSave(
                    pack_checkpoint_state(to_pure_dict(nnx.state(self.ema_models)))
                )
        with ocp.Checkpointer(
            ocp.CompositeCheckpointHandler(*checkpoint_items)
        ) as checkpointer:
            checkpointer.save(
                ckpt_path.absolute(),
                ocp.args.Composite(**checkpoint_items),
            )


def _task_parameter_masks(models: Mapping[str, BaseModel]) -> dict[str, Any]:
    """Select each task's unique variables within the shared parameter tree."""
    parameters = {
        path: variable
        for path, variable in nnx.iter_graph(models)
        if isinstance(variable, nnx.Param)
    }
    state = to_flat_state(nnx.state(models, nnx.Param))
    masks = {}
    for key, model in models.items():
        identities = {
            id(variable)
            for _, variable in nnx.iter_graph(model)
            if isinstance(variable, nnx.Param)
        }
        masks[key] = from_flat_state(
            [
                (
                    path,
                    variable.replace(
                        value=jnp.asarray(id(parameters[path]) in identities)
                    ),
                )
                for path, variable in state
            ]
        )
    return masks


def _global_gradient_norm(grads: Any) -> jnp.ndarray:
    """Compute the global L2 norm without overflowing finite gradient squares."""
    leaves = [value for value in jax.tree.leaves(grads) if value.size]
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float64)
    maximum = jnp.max(
        jnp.stack([jnp.max(jnp.abs(value)).astype(jnp.float64) for value in leaves])
    )
    scale = jnp.where(maximum > 0, maximum, jnp.ones_like(maximum))
    scaled = [value / scale.astype(value.dtype) for value in leaves]
    # Low-precision gradients accumulate their squared norm in float32, then
    # round the scalar back to their own dtype before the float64 global norm.
    normalized = []
    for value in scaled:
        norm = jnp.linalg.vector_norm(
            value.astype(jnp.promote_types(value.dtype, jnp.float32)).reshape(-1)
        ).astype(value.dtype)
        if value.dtype in (jnp.float16, jnp.bfloat16):
            # The intervening low-precision rounding is part of the reduction
            # contract; GPU cast folding must not replace it by a float32 norm.
            norm = jax.lax.optimization_barrier(norm)
        normalized.append(norm.astype(jnp.float64))
    normalized = jnp.stack(normalized)
    return scale * jnp.linalg.vector_norm(normalized)


def _clip_gradients(
    grads: Any, max_norm: float, *, total_norm: jnp.ndarray | None = None
) -> Any:
    """Apply PyTorch-compatible global-norm clipping to one gradient tree."""
    if total_norm is None:
        total_norm = _global_gradient_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (total_norm + 1e-6))
    return jax.tree.map(lambda gradient: gradient * scale.astype(gradient.dtype), grads)


def _batch_atom_capacity(inputs: dict[str, Any]) -> int:
    """Return the static atom axis used by loss terms without a physical mask."""
    if "edge_index" in inputs:
        return inputs["atype"].shape[0]
    return inputs["nlist"].shape[1]


def _evaluate_inputs(
    model: BaseModel, inputs: dict[str, Any]
) -> dict[str, jnp.ndarray]:
    """Evaluate the model's dense or graph lower interface for training losses."""
    if "edge_index" not in inputs:
        return _evaluate_model_dict(model, **inputs)
    model_dict = model.call_common_lower_graph(**inputs)
    model_dict["n_node"] = inputs["n_node"]
    return _adapt_model_outputs(model, model_dict)


def _adapt_model_outputs(
    model: BaseModel, model_dict: dict[str, jnp.ndarray]
) -> dict[str, jnp.ndarray]:
    """Expose frame properties or energies and derivatives for shared losses."""
    var_name = model.get_var_name()
    if var_name is not None:
        model_dict[f"atom_{var_name}"] = model_dict[var_name]
        model_dict[var_name] = model_dict[f"{var_name}_redu"]
        model_dict["_out_std"] = model.atomic_model.out_std[0, 0]
        model_dict["_out_bias"] = model.atomic_model.out_bias[0, 0]
        return model_dict
    model_dict["atom_energy"] = model_dict["energy"]
    model_dict["energy"] = model_dict["energy_redu"]
    model_dict["force"] = model_dict["energy_derv_r"].squeeze(-2)
    model_dict["virial"] = model_dict["energy_derv_c_redu"].squeeze(-2)
    if model_dict.get("energy_derv_r_mag") is not None:
        model_dict["force_mag"] = model_dict["energy_derv_r_mag"].squeeze(-2)
    if model_dict.get("energy_derv_r_derv_r") is not None:
        model_dict["hessian"] = model_dict["energy_derv_r_derv_r"].squeeze(-3)
    return model_dict


def _evaluate_model_dict(
    model: BaseModel,
    extended_coord: jnp.ndarray,
    extended_atype: jnp.ndarray,
    nlist: jnp.ndarray,
    mapping: jnp.ndarray | None,
    fp: jnp.ndarray | None,
    ap: jnp.ndarray | None,
    *,
    spin: jnp.ndarray | None = None,
    charge_spin: jnp.ndarray | None = None,
) -> dict[str, jnp.ndarray]:
    model_kwargs = {}
    if spin is not None:
        local_spin = spin.reshape(nlist.shape[0], nlist.shape[1], 3)
        model_kwargs["spin"] = (
            jnp.take_along_axis(local_spin, mapping[..., None], axis=1)
            if mapping is not None
            else local_spin
        )
    if charge_spin is not None:
        model_kwargs["charge_spin"] = charge_spin
    model_dict_lower = model.call_common_lower(
        extended_coord,
        extended_atype,
        nlist,
        mapping,
        fp,
        ap,
        **model_kwargs,
    )
    model_dict = communicate_extended_output(
        model_dict_lower,
        model.model_output_def(),
        mapping,
        do_atomic_virial=False,
    )
    # Loss normalization counts physical atoms, excluding padded type -1 rows.
    model_dict["mask"] = model_dict["mask"].reshape(nlist.shape[:2])
    return _adapt_model_outputs(model, model_dict)


def get_additional_data_requirement(model: BaseModel) -> list[DataRequirementItem]:
    """Return conditioning fields required by the model's input contract."""
    requirements = []
    if model.get_dim_fparam() > 0:
        has_default = model.has_default_fparam()
        requirements.append(
            DataRequirementItem(
                "fparam",
                model.get_dim_fparam(),
                atomic=False,
                must=not has_default,
                default=np.asarray(model.get_default_fparam()) if has_default else 0.0,
                source_policy="default" if has_default else "tracked",
            )
        )
    if model.get_dim_aparam() > 0:
        requirements.append(
            DataRequirementItem(
                "aparam", model.get_dim_aparam(), atomic=True, must=True
            )
        )
    if model.has_spin():
        allow_missing = model.spin.allow_missing_label
        requirements.append(
            DataRequirementItem(
                "spin",
                ndof=3,
                atomic=True,
                must=not allow_missing,
                default=0.0,
                source_policy="default" if allow_missing else "tracked",
            )
        )
    if model.has_chg_spin_ebd():
        default_charge_spin = model.get_default_chg_spin()
        has_default = default_charge_spin is not None
        requirements.append(
            DataRequirementItem(
                "charge_spin",
                ndof=2,
                atomic=False,
                must=not has_default,
                default=np.asarray(default_charge_spin) if has_default else 0.0,
                source_policy="default" if has_default else "tracked",
            )
        )
    return requirements


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
    if path and path[-1] == "rotation_key":
        return target
    if random_fitting and not any("descriptor" in str(part) for part in path):
        return target
    if _same_state_leaf(target, source):
        return source.astype(target.dtype) if source.dtype != target.dtype else source
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
        and (
            target_dtype == source_dtype
            or (
                jnp.issubdtype(target_dtype, jnp.floating)
                and jnp.issubdtype(source_dtype, jnp.floating)
            )
        )
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
        kk: (bool(vv) if kk.startswith("find_") else jnp.asarray(vv))
        for kk, vv in numpy_data.items()
        if vv is not None
    }
    return jax_data
