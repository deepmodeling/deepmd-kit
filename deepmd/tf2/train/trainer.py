# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow 2 eager training loop."""

from __future__ import (
    annotations,
)

import functools
import json
import logging
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

import numpy as np

from deepmd.dpmodel.loss import (
    DOSLoss,
    EnergyLoss,
    PropertyLoss,
    TensorLoss,
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
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.dpmodel.utils.learning_rate import (
    make_learning_rate_schedule,
)
from deepmd.dpmodel.utils.training_utils import (
    resolve_model_prob,
)
from deepmd.tf2.common import (
    to_tensorflow_array,
    to_tf_tensor,
    unwrap_value,
    wrap_value,
)
from deepmd.tf2.env import (
    tf,
)
from deepmd.tf2.model.make_model import (
    prepare_lower_inputs,
)
from deepmd.tf2.model.model import (
    get_model,
)
from deepmd.tf2.transform_output import (
    communicate_extended_output,
)
from deepmd.tf2.utils.multi_task import (
    apply_shared_links,
    sanitize_shared_links,
)
from deepmd.utils.argcheck import (
    resolve_full_validation_start_step,
)
from deepmd.utils.data import (
    DataRequirementItem,
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
    stat_file_specs_by_task,
)

if TYPE_CHECKING:
    from deepmd.utils.data_system import (
        DeepmdDataSystem,
    )

log = logging.getLogger(__name__)

TF2_TRAINING_STATE_FILE = "training_state.json"


def get_loss(
    loss_params: dict[str, Any],
    start_lr: float,
    _ntypes: int,
    _model: Any,
) -> EnergyLoss | DOSLoss | TensorLoss | PropertyLoss:
    """Build a dpmodel-compatible loss object for TF2 training."""
    loss_type = loss_params.get("type", "ener")
    loss_params = dict(loss_params)
    if loss_type == "ener":
        loss_params["starter_learning_rate"] = start_lr
        return EnergyLoss(**loss_params)
    if loss_type == "dos":
        loss_params["starter_learning_rate"] = start_lr
        loss_params["numb_dos"] = _model.model_output_def()["dos"].output_size
        return DOSLoss(**loss_params)
    if loss_type == "tensor":
        model_output_type = list(_model.model_output_type())
        if "mask" in model_output_type:
            model_output_type.remove("mask")
        tensor_name = model_output_type[0]
        loss_params["tensor_size"] = _model.model_output_def()[tensor_name].output_size
        loss_params["label_name"] = tensor_name
        if tensor_name == "polarizability":
            tensor_name = "polar"
        loss_params["tensor_name"] = tensor_name
        return TensorLoss(**loss_params)
    if loss_type == "property":
        loss_params["task_dim"] = _model.get_task_dim()
        loss_params["var_name"] = _model.get_var_name()
        loss_params["intensive"] = _model.get_intensive()
        return PropertyLoss(**loss_params)
    raise ValueError(f"Unsupported loss type for tf2: {loss_type}")


def get_additional_data_requirement(_model: Any) -> list[DataRequirementItem]:
    """Return model-input data requirements not declared by the loss."""
    additional_data_requirement: list[DataRequirementItem] = []
    if _model.get_dim_fparam() > 0:
        has_default_fparam = _model.has_default_fparam()
        fparam_default = (
            np.asarray(_model.get_default_fparam()) if has_default_fparam else 0.0
        )
        additional_data_requirement.append(
            DataRequirementItem(
                "fparam",
                _model.get_dim_fparam(),
                atomic=False,
                must=not has_default_fparam,
                default=fparam_default,
            )
        )
    if _model.get_dim_aparam() > 0:
        additional_data_requirement.append(
            DataRequirementItem(
                "aparam",
                _model.get_dim_aparam(),
                atomic=True,
                must=True,
            )
        )
    if _model.has_chg_spin_ebd():
        has_default_cs = _model.has_default_chg_spin()
        default_cs = (
            np.asarray(to_tf_tensor(_model.get_default_chg_spin()).numpy())
            if has_default_cs
            else 0.0
        )
        additional_data_requirement.append(
            DataRequirementItem(
                "charge_spin",
                ndof=2,
                atomic=False,
                must=not has_default_cs,
                default=default_cs,
            )
        )
    return additional_data_requirement


def _as_task_map(
    value: Any,
    *,
    multi_task: bool,
    model_keys: list[str],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        if all(model_key in value for model_key in model_keys):
            return {model_key: value[model_key] for model_key in model_keys}
    if multi_task:
        return {model_key: value[model_key] for model_key in model_keys}
    return {DEFAULT_TASK_KEY: value}


class _TaskModelContainer(tf.Module):
    """Track task-keyed TF modules with stable attribute names."""

    def __init__(self, models: Mapping[str, tf.Module]) -> None:
        super().__init__(name="models")
        self.task_keys = tuple(models)
        for index, key in enumerate(self.task_keys):
            setattr(self, f"task_{index}", models[key])


class Trainer(AbstractTrainer):
    """Training driver for TensorFlow 2 eager models."""

    def __init__(
        self,
        config: dict[str, Any],
        training_data: DeepmdDataSystem | Mapping[str, DeepmdDataSystem],
        stat_file_spec: StatFileSpec | Mapping[str, StatFileSpec] | None = None,
        validation_data: DeepmdDataSystem
        | Mapping[str, DeepmdDataSystem | None]
        | None = None,
        init_model: str | None = None,
        restart_model: str | None = None,
        finetune_model: str | None = None,
        finetune_links: dict[str, Any] | None = None,
        shared_links: dict[str, Any] | None = None,
        min_nbor_dist: float | Mapping[str, float | None] | None = None,
    ) -> None:
        if finetune_model is not None and (
            init_model is not None or restart_model is not None
        ):
            raise ValueError(
                "finetune_model cannot be combined with init_model or restart_model."
            )
        if init_model is not None and restart_model is not None:
            raise ValueError("init_model cannot be combined with restart_model.")

        self.config = config
        self.init_model = init_model
        self.restart_model = restart_model
        self.finetune_model = finetune_model
        self.finetune_links = finetune_links
        self.shared_links = shared_links
        self.restart_training = restart_model is not None
        model_params = config["model"]
        training_params = config["training"]
        self.validating_params = config.get("validating", {}) or {}
        self._validate_unsupported_config(config)

        self.multi_task = "model_dict" in model_params
        self.model_def_script = deepcopy(model_params)
        self.full_validation_state: dict[str, Any] = {}
        self.model_keys = (
            list(model_params["model_dict"]) if self.multi_task else [DEFAULT_TASK_KEY]
        )
        self.model_params_by_task = (
            {
                model_key: model_params["model_dict"][model_key]
                for model_key in self.model_keys
            }
            if self.multi_task
            else {DEFAULT_TASK_KEY: model_params}
        )
        self.training_data_by_task = _as_task_map(
            training_data,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
        )
        self.validation_data_by_task = _as_task_map(
            validation_data,
            multi_task=self.multi_task,
            model_keys=self.model_keys,
        )
        self.stat_file_specs = stat_file_specs_by_task(
            stat_file_spec,
            self.model_keys,
        )

        self.num_steps = int(training_params["numb_steps"])
        self.save_ckpt = str(training_params.get("save_ckpt", "model.ckpt"))
        self.max_ckpt_keep = int(training_params.get("max_ckpt_keep", 5))
        self.gradient_max_norm = float(training_params.get("gradient_max_norm", 0.0))
        self.tensorboard = bool(training_params.get("tensorboard", False))
        self.tensorboard_log_dir = str(
            training_params.get("tensorboard_log_dir", "log")
        )
        self.tensorboard_freq = int(training_params.get("tensorboard_freq", 1))
        self.enable_compile = bool(training_params.get("enable_compile", False))
        self.change_bias_after_training = bool(
            training_params.get("change_bias_after_training", False)
        )
        self.start_step = 0

        self.models = {
            model_key: get_model(deepcopy(self.model_params_by_task[model_key]))
            for model_key in self.model_keys
        }
        self._configure_model_compile()
        self.set_min_nbor_dist(min_nbor_dist)
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]

        self.losses = {}
        for model_key in self.model_keys:
            loss_param = (
                config["loss_dict"][model_key]
                if self.multi_task
                else config.get("loss", {})
            )
            self.losses[model_key] = get_loss(
                deepcopy(loss_param),
                config["learning_rate"]["start_lr"],
                len(self.model_params_by_task[model_key]["type_map"]),
                self.models[model_key],
            )
        self.loss = self.losses if self.multi_task else self.losses[DEFAULT_TASK_KEY]

        self.valid_numb_batch_by_task = {}
        for model_key in self.model_keys:
            data_requirement = list(self.losses[model_key].label_requirement)
            data_requirement += get_additional_data_requirement(self.models[model_key])
            self.training_data_by_task[model_key].add_data_requirements(
                data_requirement
            )
            valid_data = self.validation_data_by_task[model_key]
            if valid_data is not None:
                valid_data.add_data_requirements(data_requirement)
            valid_params = (
                training_params["data_dict"][model_key].get("validation_data", {})
                if self.multi_task
                else training_params.get("validation_data", {})
            ) or {}
            self.valid_numb_batch_by_task[model_key] = max(
                int(valid_params.get("numb_btch", 1)),
                1,
            )

        self._sample_funcs = {}
        for model_key in self.model_keys:
            nbatch = int(
                self.model_params_by_task[model_key].get("data_stat_nbatch", 10)
            )
            train_data = self.training_data_by_task[model_key]

            @functools.lru_cache
            def sample(
                _data: DeepmdDataSystem = train_data,
                _nbatch: int = nbatch,
            ) -> list[dict[str, np.ndarray]]:
                return make_stat_input(_data, _nbatch)

            self._sample_funcs[model_key] = sample

        if init_model is None and restart_model is None:
            for model_key in self.model_keys:
                finetune_has_new_type = (
                    self.finetune_model is not None
                    and self.finetune_links is not None
                    and model_key in self.finetune_links
                    and self.finetune_links[model_key].get_has_new_type()
                )
                if self.finetune_model is not None and not finetune_has_new_type:
                    continue
                log.info(
                    "data stating for task %s... (this step may take long time)",
                    model_key,
                )
                with open_stat_file(self.stat_file_specs[model_key]) as stat_file_path:
                    self.models[model_key].compute_or_load_stat(
                        self._sample_funcs[model_key],
                        stat_file_path=stat_file_path,
                    )

        if self.finetune_model is not None:
            self._apply_finetune()
            self.model = (
                self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]
            )

        self.model_prob = (
            resolve_model_prob(
                self.model_keys,
                training_params.get("model_prob"),
                self.training_data_by_task,
            )
            if self.multi_task
            else None
        )
        self._apply_shared_links(
            resume=init_model is not None or restart_model is not None
        )

        self.lr_schedule = make_learning_rate_schedule(
            config["learning_rate"], self.num_steps
        )
        self.optimizer = self._build_optimizer(config.get("optimizer", {}))
        self.model_container = _TaskModelContainer(self.models)
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False, name="step")
        self.checkpoint = tf.train.Checkpoint(
            step=self.step,
            optimizer=self.optimizer,
            model=self.model_container,
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=self._checkpoint_directory(),
            max_to_keep=self.max_ckpt_keep if self.max_ckpt_keep > 0 else None,
            checkpoint_name=Path(self.save_ckpt).name,
        )

        restart_restore: tuple[str, Any] | None = None
        if init_model is not None:
            self._restore_model(init_model)
            self.step.assign(0)
        elif restart_model is not None:
            restart_restore = self._restore_checkpoint(restart_model)

        self._build_optimizer_slots()
        if restart_restore is not None:
            resolved, restore_status = restart_restore
            restore_status.assert_existing_objects_matched()
            self.start_step = int(self.step.numpy())
            log.info(
                "Restarted TF2 training from %s at step %d",
                resolved,
                self.start_step,
            )
        self._compiled_train_steps: dict[str, Any] = {}
        self._compiled_prepare_steps: dict[str, Any] = {}
        self._compiled_prepared_train_steps: dict[str, Any] = {}
        self._compiled_eval_steps: dict[str, Any] = {}
        self.training_tasks = self._make_training_tasks()
        self.summary_writer: Any | None = None
        self.full_validator: Any | None = None
        super().__init__(
            TrainerConfig.from_training_params(
                training_params,
                num_steps=self.num_steps,
                start_step=self.start_step,
                restart_training=self.restart_training,
            ),
            rank_context=RankContext(rank=0, world_size=1),
        )
        self.full_validator = self._create_full_validator()

    def _validate_unsupported_config(self, config: Mapping[str, Any]) -> None:
        training_params = config["training"]
        unsupported_true_flags = {
            "profiling": "profiling",
            "enable_profiler": "TensorFlow profiler",
        }
        for flag_name, feature_name in unsupported_true_flags.items():
            if training_params.get(flag_name, False):
                raise NotImplementedError(
                    f"TF2 training does not support {feature_name} yet."
                )
        if training_params.get("mixed_precision") is not None:
            raise NotImplementedError(
                "TF2 training does not support mixed_precision yet."
            )
        if (config.get("nvnmd", {}) or {}).get("enable", False):
            raise NotImplementedError("TF2 training does not support NVNMD yet.")
        if config["model"].get("modifier") is not None:
            raise NotImplementedError(
                "TF2 training does not support model.modifier yet."
            )

    def _configure_model_compile(self) -> None:
        """Apply training.enable_compile to TF2 models that support lower XLA."""
        if not self.enable_compile:
            return
        log.info("Enabling TF2 lower-forward XLA compilation.")
        for model_key, model in self.models.items():
            set_enable_compile = getattr(model, "set_enable_compile", None)
            if not callable(set_enable_compile):
                log.warning(
                    "Model %s does not support training.enable_compile; ignoring.",
                    model_key,
                )
                continue
            set_enable_compile(True)

    def _create_full_validator(self) -> Any | None:
        if not self._is_validation_requested("full_validation"):
            return None
        self._raise_if_full_validation_unsupported()
        from deepmd.dpmodel.train.validation import (
            resolve_best_checkpoint_dir,
        )
        from deepmd.tf2.train.validation import (
            TF2FullValidator,
        )

        return TF2FullValidator(
            validating_params=self.validating_params,
            validation_data=self.validation_data_by_task[DEFAULT_TASK_KEY],
            model=self.models[DEFAULT_TASK_KEY],
            state_store=self.full_validation_state,
            num_steps=self.num_steps,
            rank=0,
            restart_training=self.restart_training,
            checkpoint_dir=resolve_best_checkpoint_dir(
                self.validating_params, self.save_ckpt
            ),
        )

    def _is_validation_requested(self, flag_name: str) -> bool:
        if not self.validating_params.get(flag_name, False):
            return False
        start_step = resolve_full_validation_start_step(
            self.validating_params.get("full_val_start", 0.5),
            self.num_steps,
        )
        return start_step is not None and start_step <= self.num_steps

    def _raise_if_full_validation_unsupported(self) -> None:
        if self.multi_task:
            raise ValueError(
                "validating.full_validation only supports single-task energy "
                "training; multi-task training is not supported."
            )
        if not isinstance(self.loss, EnergyLoss):
            raise ValueError(
                "validating.full_validation only supports single-task energy training."
            )
        if self.validation_data_by_task[DEFAULT_TASK_KEY] is None:
            raise ValueError(
                "validating.full_validation requires `training.validation_data` "
                "to be configured."
            )

    def _build_optimizer(self, optimizer_params: Mapping[str, Any]) -> Any:
        optimizer_type = optimizer_params.get("type", "Adam")
        beta1 = float(optimizer_params.get("adam_beta1", 0.9))
        beta2 = float(optimizer_params.get("adam_beta2", 0.999))
        weight_decay = float(optimizer_params.get("weight_decay", 0.0))
        learning_rate = float(self.lr_schedule.value(self.start_step))
        if optimizer_type == "Adam":
            if weight_decay != 0.0:
                raise RuntimeError(
                    "TF2 Adam optimizer does not support weight_decay. "
                    "Set optimizer/weight_decay to 0 or use AdamW."
                )
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                beta_1=beta1,
                beta_2=beta2,
            )
        if optimizer_type == "AdamW":
            if not hasattr(tf.keras.optimizers, "AdamW"):
                raise RuntimeError("This TensorFlow version does not provide AdamW.")
            return tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                beta_1=beta1,
                beta_2=beta2,
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer type for tf2: {optimizer_type}")

    def _build_optimizer_slots(self) -> None:
        variables = _unique_variables(self.model_container.trainable_variables)
        build = getattr(self.optimizer, "build", None)
        if callable(build):
            build(variables)

    def _checkpoint_directory(self) -> str:
        return str(Path(f"{self.save_ckpt}.tf2"))

    def _resolve_checkpoint_path(self, checkpoint_path: str) -> str:
        path = Path(checkpoint_path)
        candidates = [path]
        if not str(path).endswith(".tf2"):
            candidates.append(Path(f"{checkpoint_path}.tf2"))
        for candidate in candidates:
            if candidate.is_dir():
                latest = tf.train.latest_checkpoint(str(candidate))
                if latest is not None:
                    return latest
        if path.is_file() or Path(f"{path}.index").is_file():
            return str(path)
        raise FileNotFoundError(
            f"Cannot find TF2 checkpoint {checkpoint_path!r}. Expected a "
            "CheckpointManager directory or a checkpoint prefix."
        )

    def _restore_model(self, checkpoint_path: str) -> None:
        resolved = self._resolve_checkpoint_path(checkpoint_path)
        model_checkpoint = tf.train.Checkpoint(model=self.model_container)
        restore_status = model_checkpoint.restore(resolved).expect_partial()
        restore_status.assert_existing_objects_matched()
        log.info("Initialized TF2 model variables from %s", resolved)

    def _restore_checkpoint(self, checkpoint_path: str) -> tuple[str, Any]:
        resolved = self._resolve_checkpoint_path(checkpoint_path)
        restore_status = self.checkpoint.restore(resolved).expect_partial()
        return resolved, restore_status

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
    def _deserialize_models(model_data: dict[str, Any]) -> dict[str, Any]:
        from deepmd.tf2.model.base_model import (
            BaseModel,
        )

        if "model_dict" in model_data["model_def_script"]:
            return {
                model_key: BaseModel.deserialize(
                    model_data["model"]["model_dict"][model_key]
                )
                for model_key in model_data["model_def_script"]["model_dict"]
            }
        return {DEFAULT_TASK_KEY: BaseModel.deserialize(model_data["model"])}

    def set_min_nbor_dist(
        self,
        min_nbor_dist: float | Mapping[str, float | None] | None,
    ) -> None:
        if min_nbor_dist is None:
            return
        if isinstance(min_nbor_dist, Mapping):
            for model_key, value in min_nbor_dist.items():
                if value is not None and model_key in self.models:
                    self.models[model_key].min_nbor_dist = float(value)
            return
        self.models[DEFAULT_TASK_KEY].min_nbor_dist = float(min_nbor_dist)

    def _apply_finetune(self) -> None:
        if self.finetune_model is None or self.finetune_links is None:
            return
        from deepmd.tf2.utils.serialization import (
            serialize_from_file,
        )

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

    def _apply_shared_links(self, *, resume: bool) -> None:
        if self.shared_links is None:
            return
        model_key_prob_map = (
            {
                model_key: float(prob)
                for model_key, prob in zip(
                    self.model_keys,
                    self.model_prob,
                    strict=True,
                )
            }
            if self.model_prob is not None
            else dict.fromkeys(self.model_keys, 1.0)
        )
        apply_shared_links(
            self.models,
            self.shared_links,
            model_key_prob_map=model_key_prob_map,
            resume=resume,
        )
        self.model = self.models if self.multi_task else self.models[DEFAULT_TASK_KEY]

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
        target_model: Any,
        source_model: Any,
        *,
        random_fitting: bool,
    ) -> Any:
        from deepmd.tf2.model.base_model import (
            BaseModel,
        )

        copied = _copy_matching_state_tree(
            target_model.serialize(),
            source_model.serialize(),
            random_fitting=random_fitting,
        )
        return BaseModel.deserialize(copied)

    def _make_training_tasks(self) -> TrainingTaskCollection:
        return TrainingTaskCollection(
            [
                TrainingTask(
                    key=model_key,
                    training_data=self.training_data_by_task[model_key],
                    validation_data=self.validation_data_by_task[model_key],
                    valid_numb_batch=self.valid_numb_batch_by_task[model_key],
                )
                for model_key in self.model_keys
            ],
            probabilities=self.model_prob,
        )

    def run(self, tasks: TrainingTaskCollection | None = None) -> None:
        """Run TF2 training through the backend-independent trainer loop."""
        if tasks is None:
            tasks = self.training_tasks
        log.info("Start to train %d steps.", self.num_steps)
        wall_start = time.time()
        super().run(tasks)
        if self.change_bias_after_training:
            self._change_bias_after_training()
            if self.rank_context.is_chief:
                self.save_checkpoint(self.num_steps)
        log.info("Training finished. Total wall time: %.2fs", time.time() - wall_start)

    def on_train_begin(self, tasks: TrainingTaskCollection) -> None:
        del tasks
        if self.tensorboard and self.rank_context.is_chief:
            self.summary_writer = tf.summary.create_file_writer(
                self.tensorboard_log_dir
            )

    def on_train_end(self, tasks: TrainingTaskCollection) -> None:
        del tasks
        if self.summary_writer is not None:
            self.summary_writer.close()
            self.summary_writer = None

    def select_task(self, tasks: TrainingTaskCollection) -> TrainingTask:
        if not tasks.is_multitask:
            return tasks[tasks.keys[0]]
        from deepmd.utils import random as dp_random

        model_index = dp_random.choice(
            np.arange(len(tasks), dtype=np.int_),
            p=tasks.probabilities,
        )
        return tasks[tasks.keys[int(model_index)]]

    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        """Run one TensorFlow optimizer step."""
        task_key = task.key
        cur_lr = float(self.lr_schedule.value(step))
        input_dict, label_dict, natoms = self.get_data(is_train=True, task_key=task_key)
        do_virial = bool(label_dict.pop("_do_virial", True))
        if self._use_prepared_step(task_key):
            prepared = self._prepare_lower_batch(task_key, input_dict)
            more_loss = self._compiled_prepared_train_step(
                task_key,
                label_dict,
                prepared,
                tf.constant(float(natoms), dtype=tf.float64),
                tf.constant(cur_lr, dtype=tf.float64),
                tf.constant(step + 1, dtype=tf.int64),
                do_virial,
            )
        else:
            more_loss = self._compiled_train_step(
                task_key,
                input_dict,
                label_dict,
                tf.constant(float(natoms), dtype=tf.float64),
                tf.constant(cur_lr, dtype=tf.float64),
                tf.constant(step + 1, dtype=tf.int64),
                do_virial,
            )
        self._write_tensorboard_step(
            task_key,
            display_step=step + 1,
            learning_rate=cur_lr,
            more_loss=more_loss,
        )
        return TrainStepResult(
            task_key=task_key,
            step=step,
            payload={
                "more_loss": more_loss,
                "cur_lr": cur_lr,
            },
        )

    def _use_prepared_step(self, task_key: str) -> bool:
        if not bool(getattr(self, "enable_compile", False)):
            return False
        model = self.models[task_key]
        return callable(getattr(model, "call_common_lower", None)) or callable(
            getattr(model, "_call_common_lower_formatted", None)
        )

    def _prepare_lower_batch(
        self,
        task_key: str,
        input_dict: dict[str, Any],
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        if task_key not in self._compiled_prepare_steps:
            self._compiled_prepare_steps[task_key] = (
                self._make_compiled_prepare_lower_batch(task_key)
            )
        prepared = self._compiled_prepare_steps[task_key](
            input_dict["coord"],
            input_dict["atype"],
            input_dict.get("box"),
            input_dict.get("fparam"),
            input_dict.get("aparam"),
            input_dict.get("charge_spin"),
        )
        return prepared[:-1]

    def _make_compiled_prepare_lower_batch(self, task_key: str) -> Any:
        model = self.models[task_key]

        @tf.function(reduce_retracing=True)
        def compiled_prepare_lower_batch(
            coord: Any,
            atype: Any,
            box: Any,
            fparam: Any,
            aparam: Any,
            charge_spin: Any,
        ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, bool]:
            cc, bb, fp, ap, cs, _input_prec = model._input_type_cast(
                to_tensorflow_array(coord),
                box=to_tensorflow_array(box),
                fparam=to_tensorflow_array(fparam),
                aparam=to_tensorflow_array(aparam),
                charge_spin=to_tensorflow_array(charge_spin),
            )
            return prepare_lower_inputs(
                rcut=model.get_rcut(),
                sel=model.get_sel(),
                mixed_types=model.mixed_types(),
                coord=cc,
                atype=to_tensorflow_array(atype),
                box=bb,
                fparam=fp,
                aparam=ap,
                charge_spin=cs,
                # Model-level pair exclusion is a nlist-BUILD transform
                # (decision #18/A4): the compiled lower consumes a pre-excluded
                # nlist, so fold exclusion in here at the compiled-training
                # prepare seam. Guard atomic_model for test doubles.
                pair_excl=getattr(
                    getattr(model, "atomic_model", None), "pair_excl", None
                ),
            )

        return compiled_prepare_lower_batch

    def _compiled_prepared_train_step(
        self,
        task_key: str,
        label_dict: dict[str, Any],
        prepared: tuple[Any, Any, Any, Any, Any, Any, Any, Any],
        natoms: Any,
        cur_lr: Any,
        next_step: Any,
        do_virial: bool,
    ) -> dict[str, Any]:
        if task_key not in self._compiled_prepared_train_steps:
            self._compiled_prepared_train_steps[task_key] = (
                self._make_compiled_prepared_train_step(task_key)
            )
        return self._compiled_prepared_train_steps[task_key](
            label_dict,
            *prepared,
            natoms,
            cur_lr,
            next_step,
            do_virial,
        )

    def _make_compiled_prepared_train_step(self, task_key: str) -> Any:
        variables = _unique_variables(self.models[task_key].trainable_variables)

        @tf.function(reduce_retracing=True, jit_compile=True)
        def compiled_prepared_train_step(
            label_dict: dict[str, Any],
            extended_coord: Any,
            extended_atype: Any,
            nlist: Any,
            mapping: Any,
            fparam: Any,
            aparam: Any,
            charge_spin: Any,
            extended_coord_corr: Any,
            natoms: Any,
            cur_lr: Any,
            next_step: Any,
            do_virial: bool,
        ) -> dict[str, Any]:
            self._assign_learning_rate(cur_lr)
            with tf.GradientTape() as tape:
                model_pred = self._call_prepared_model(
                    task_key,
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping,
                    fparam,
                    aparam,
                    charge_spin,
                    extended_coord_corr,
                    label_dict=label_dict,
                    do_virial=do_virial,
                )
                loss, more_loss = self.losses[task_key](
                    learning_rate=cur_lr,
                    natoms=natoms,
                    model_dict=model_pred,
                    label_dict=label_dict,
                )
                loss_tensor = to_tf_tensor(loss)
            gradients = tape.gradient(loss_tensor, variables)
            gradients_and_variables = [
                (grad, var)
                for grad, var in zip(gradients, variables, strict=True)
                if grad is not None
            ]
            if self.gradient_max_norm > 0.0 and gradients_and_variables:
                grads, vars_ = zip(*gradients_and_variables, strict=True)
                grads, _ = tf.clip_by_global_norm(grads, self.gradient_max_norm)
                gradients_and_variables = list(zip(grads, vars_, strict=True))
            self.optimizer.apply_gradients(gradients_and_variables)
            self.step.assign(next_step)
            return unwrap_value(more_loss)

        return compiled_prepared_train_step

    def _compiled_train_step(
        self,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        natoms: Any,
        cur_lr: Any,
        next_step: Any,
        do_virial: bool,
    ) -> dict[str, Any]:
        if task_key not in self._compiled_train_steps:
            self._compiled_train_steps[task_key] = self._make_compiled_train_step(
                task_key
            )
        return self._compiled_train_steps[task_key](
            input_dict,
            label_dict,
            natoms,
            cur_lr,
            next_step,
            do_virial,
        )

    def _make_compiled_train_step(self, task_key: str) -> Any:
        variables = _unique_variables(self.models[task_key].trainable_variables)

        @tf.function(reduce_retracing=True)
        def compiled_train_step(
            input_dict: dict[str, Any],
            label_dict: dict[str, Any],
            natoms: Any,
            cur_lr: Any,
            next_step: Any,
            do_virial: bool,
        ) -> dict[str, Any]:
            self._assign_learning_rate(cur_lr)
            with tf.GradientTape() as tape:
                model_pred = self._call_model(
                    task_key,
                    input_dict,
                    label_dict=label_dict,
                    do_virial=do_virial,
                )
                loss, more_loss = self.losses[task_key](
                    learning_rate=cur_lr,
                    natoms=natoms,
                    model_dict=model_pred,
                    label_dict=label_dict,
                )
                loss_tensor = to_tf_tensor(loss)
            gradients = tape.gradient(loss_tensor, variables)
            gradients_and_variables = [
                (grad, var)
                for grad, var in zip(gradients, variables, strict=True)
                if grad is not None
            ]
            if self.gradient_max_norm > 0.0 and gradients_and_variables:
                grads, vars_ = zip(*gradients_and_variables, strict=True)
                grads, _ = tf.clip_by_global_norm(grads, self.gradient_max_norm)
                gradients_and_variables = list(zip(grads, vars_, strict=True))
            self.optimizer.apply_gradients(gradients_and_variables)
            self.step.assign(next_step)
            return unwrap_value(more_loss)

        return compiled_train_step

    def evaluate_training(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float]:
        if step_result is not None and step_result.task_key == task.key:
            return self._more_loss_to_float(step_result.payload["more_loss"])
        input_dict, label_dict, natoms = self.get_data(is_train=True, task_key=task.key)
        do_virial = bool(label_dict.pop("_do_virial", True))
        return self._evaluate_batch(
            task.key,
            step,
            input_dict,
            label_dict,
            natoms,
            do_virial=do_virial,
        )

    def evaluate_validation(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> dict[str, float] | None:
        if task.validation_data is None:
            return None
        valid_results: dict[str, float] = {}
        sum_natoms = 0
        for _ii in range(task.valid_numb_batch):
            input_dict, label_dict, natoms = self.get_data(
                is_train=False,
                task_key=task.key,
            )
            do_virial = bool(label_dict.pop("_do_virial", True))
            results = self._evaluate_batch(
                task.key,
                step,
                input_dict,
                label_dict,
                natoms,
                do_virial=do_virial,
            )
            sum_natoms += natoms
            for key, value in results.items():
                valid_results[key] = valid_results.get(key, 0.0) + value * natoms
        if sum_natoms == 0:
            return valid_results
        return {key: value / sum_natoms for key, value in valid_results.items()}

    def _evaluate_batch(
        self,
        task_key: str,
        step: int,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        natoms: int,
        do_virial: bool = True,
    ) -> dict[str, float]:
        cur_lr = float(self.lr_schedule.value(step))
        return self._compiled_eval_step(
            task_key,
            input_dict,
            label_dict,
            tf.constant(float(natoms), dtype=tf.float64),
            tf.constant(cur_lr, dtype=tf.float64),
            do_virial,
        )

    def _compiled_eval_step(
        self,
        task_key: str,
        input_dict: dict[str, Any],
        label_dict: dict[str, Any],
        natoms: Any,
        cur_lr: Any,
        do_virial: bool,
    ) -> dict[str, float]:
        if task_key not in self._compiled_eval_steps:
            self._compiled_eval_steps[task_key] = self._make_compiled_eval_step(
                task_key
            )
        more_loss = self._compiled_eval_steps[task_key](
            input_dict,
            label_dict,
            natoms,
            cur_lr,
            do_virial,
        )
        return self._more_loss_to_float(more_loss)

    def _make_compiled_eval_step(self, task_key: str) -> Any:
        @tf.function(reduce_retracing=True)
        def compiled_eval_step(
            input_dict: dict[str, Any],
            label_dict: dict[str, Any],
            natoms: Any,
            cur_lr: Any,
            do_virial: bool,
        ) -> dict[str, Any]:
            model_pred = self._call_model(
                task_key,
                input_dict,
                label_dict=label_dict,
                do_virial=do_virial,
            )
            _, more_loss = self.losses[task_key](
                learning_rate=cur_lr,
                natoms=natoms,
                model_dict=model_pred,
                label_dict=label_dict,
            )
            return unwrap_value(more_loss)

        return compiled_eval_step

    def learning_rate(self, step: int) -> float:
        return float(self.lr_schedule.value(step))

    def save_checkpoint(self, step: int) -> None:
        self.step.assign(step)
        save_path = self.checkpoint_manager.save(checkpoint_number=step)
        self._write_training_state(Path(self._checkpoint_directory()), step=step)
        log.info("Saved TF2 checkpoint to %s", save_path)

    def run_full_validation(
        self,
        *,
        step: int,
        display_step: int,
        learning_rate: float,
    ) -> None:
        if self.full_validator is None:
            return None
        self.full_validator.model = self.models[DEFAULT_TASK_KEY]
        self.full_validator.run(
            step_id=display_step,
            display_step=display_step,
            lr=learning_rate,
            save_checkpoint=self._save_full_validation_checkpoint,
        )
        return None

    def _save_full_validation_checkpoint(
        self,
        save_path: Path,
        lr: float = 0.0,
        step: int = 0,
    ) -> None:
        del lr
        self._write_checkpoint_directory(save_path, step=step)

    def _write_checkpoint_directory(self, directory: Path, *, step: int) -> None:
        if directory.exists():
            shutil.rmtree(directory)
        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(step, dtype=tf.int64, trainable=False, name="step"),
            optimizer=self.optimizer,
            model=self.model_container,
        )
        manager = tf.train.CheckpointManager(
            checkpoint,
            directory=str(directory),
            max_to_keep=1,
            checkpoint_name=directory.stem,
        )
        manager.save(checkpoint_number=step)
        self._write_training_state(directory, step=step)

    def _write_training_state(self, directory: Path, *, step: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "backend": "TensorFlow2",
            "format_version": 1,
            "current_step": int(step),
            "model_def_script": deepcopy(self.model_def_script),
            "shared_links": sanitize_shared_links(self.shared_links),
            "multi_task": self.multi_task,
            "model_keys": list(self.model_keys),
            "min_nbor_dist": self._current_min_nbor_dist(),
            "full_validation": deepcopy(self.full_validation_state),
        }
        with (directory / TF2_TRAINING_STATE_FILE).open("w") as fp:
            json.dump(state, fp, indent=2)

    def _current_min_nbor_dist(self) -> Any:
        values = {}
        for model_key in self.model_keys:
            value = self.models[model_key].get_min_nbor_dist()
            values[model_key] = None if value is None else float(value)
        if self.multi_task:
            return values
        return values[DEFAULT_TASK_KEY]

    def _write_tensorboard_step(
        self,
        task_key: str,
        *,
        display_step: int,
        learning_rate: float,
        more_loss: dict[str, Any],
    ) -> None:
        if (
            self.summary_writer is None
            or self.tensorboard_freq <= 0
            or display_step % self.tensorboard_freq != 0
        ):
            return
        prefix = f"train/{task_key}" if self.multi_task else "train"
        with self.summary_writer.as_default():
            tf.summary.scalar(
                "learning_rate",
                tf.convert_to_tensor(learning_rate, dtype=tf.float64),
                step=display_step,
            )
            for key, value in more_loss.items():
                if "l2_" in key:
                    continue
                tf.summary.scalar(
                    f"{prefix}/{key}", to_tf_tensor(value), step=display_step
                )
            self.summary_writer.flush()

    def _change_bias_after_training(self) -> None:
        change_model_out_bias_by_task(
            self.models,
            self._sample_funcs,
            self.model_keys,
            bias_adjust_mode="change-by-statistic",
        )

    def get_data(
        self,
        *,
        is_train: bool,
        task_key: str,
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        task_key = task_key if self.multi_task else DEFAULT_TASK_KEY
        data_sys = (
            self.training_data_by_task[task_key]
            if is_train
            else self.validation_data_by_task[task_key]
        )
        if data_sys is None:
            return {}, {}, 0
        batch = normalize_batch(data_sys.get_batch())
        input_dict, label_dict = split_batch(batch)
        for opt_key in ("fparam", "charge_spin"):
            find_key = f"find_{opt_key}"
            if (
                opt_key in input_dict
                and find_key in label_dict
                and not bool(label_dict[find_key])
            ):
                input_dict.pop(opt_key)
        natoms = int(input_dict["atype"].shape[1])
        label_dict["type"] = input_dict["atype"]
        do_virial = self._batch_needs_virial(task_key, label_dict)
        input_tf = {
            key: self._to_input_tensor(key, value) for key, value in input_dict.items()
        }
        label_tf = {
            key: self._to_label_array(key, value) for key, value in label_dict.items()
        }
        label_tf["_do_virial"] = do_virial
        return input_tf, label_tf, natoms

    def _call_model(
        self,
        task_key: str,
        input_dict: dict[str, Any],
        *,
        label_dict: dict[str, Any] | None = None,
        do_virial: bool = True,
    ) -> dict[str, Any]:
        model = self.models[task_key]
        call_common = getattr(model, "call_common", None)
        if callable(call_common):
            model_ret = call_common(
                input_dict["coord"],
                input_dict["atype"],
                box=input_dict.get("box"),
                fparam=input_dict.get("fparam"),
                aparam=input_dict.get("aparam"),
                charge_spin=input_dict.get("charge_spin"),
                do_atomic_virial=False,
                do_deriv_c=do_virial,
            )
            return self._translate_model_ret_to_loss_dict(
                task_key,
                model_ret,
                label_dict=label_dict,
                do_virial=do_virial,
            )
        return model.call(
            input_dict["coord"],
            input_dict["atype"],
            box=input_dict.get("box"),
            fparam=input_dict.get("fparam"),
            aparam=input_dict.get("aparam"),
            charge_spin=input_dict.get("charge_spin"),
        )

    def _call_prepared_model(
        self,
        task_key: str,
        extended_coord: Any,
        extended_atype: Any,
        nlist: Any,
        mapping: Any,
        fparam: Any,
        aparam: Any,
        charge_spin: Any,
        extended_coord_corr: Any,
        *,
        label_dict: dict[str, Any] | None = None,
        do_virial: bool = True,
    ) -> dict[str, Any]:
        model = self.models[task_key]
        call_lower_formatted = getattr(model, "_call_common_lower_formatted", None)
        if callable(call_lower_formatted):
            model_ret_lower = wrap_value(
                call_lower_formatted(
                    extended_coord,
                    extended_atype,
                    nlist,
                    mapping=mapping,
                    fparam=fparam,
                    aparam=aparam,
                    do_atomic_virial=False,
                    do_deriv_c=do_virial,
                    extended_coord_corr=extended_coord_corr,
                    charge_spin=charge_spin,
                )
            )
        else:
            model_ret_lower = model.call_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=False,
                do_deriv_c=do_virial,
                extended_coord_corr=extended_coord_corr,
                charge_spin=charge_spin,
                nlist_is_formatted=True,
            )
        model_ret = wrap_value(
            communicate_extended_output(
                unwrap_value(model_ret_lower),
                model.model_output_def(),
                to_tf_tensor(mapping),
                do_atomic_virial=False,
            )
        )
        return self._translate_model_ret_to_loss_dict(
            task_key,
            model_ret,
            label_dict=label_dict,
            do_virial=do_virial,
        )

    def _translate_model_ret_to_loss_dict(
        self,
        task_key: str,
        model_ret: dict[str, Any],
        *,
        label_dict: dict[str, Any] | None = None,
        do_virial: bool = True,
    ) -> dict[str, Any]:
        translated_output_def = getattr(
            self.models[task_key],
            "translated_output_def",
            None,
        )
        if not callable(translated_output_def):
            return model_ret
        output_defs = translated_output_def()
        model_pred = {}
        for output_key, output_def in output_defs.items():
            source_key = output_def.name
            if source_key not in model_ret or model_ret[source_key] is None:
                continue
            model_pred[output_key] = self._match_output_rank(
                model_ret[source_key],
                output_def,
            )
        if (
            not do_virial
            and label_dict is not None
            and "virial" in label_dict
            and "virial" in output_defs
            and "virial" not in model_pred
        ):
            model_pred["virial"] = label_dict["virial"]
        return model_pred

    @classmethod
    def _match_output_rank(cls, value: Any, output_def: Any) -> Any:
        expected_rank = len(output_def.shape) + (2 if output_def.atomic else 1)
        axis = -(len(output_def.shape) + 1)
        while True:
            rank = cls._shape_rank(value)
            if rank is None or rank <= expected_rank:
                return value
            if cls._shape_dim(value, axis) != 1:
                return value
            squeeze = getattr(value, "squeeze", None)
            if not callable(squeeze):
                value = tf.squeeze(value, axis=axis)
            else:
                value = squeeze(axis)

    @staticmethod
    def _shape_rank(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        rank = getattr(shape, "rank", None)
        if rank is not None:
            return int(rank)
        try:
            return len(shape)
        except TypeError:
            return None

    @staticmethod
    def _shape_dim(value: Any, axis: int) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            dim = shape[axis]
        except (IndexError, TypeError):
            return None
        return getattr(dim, "value", dim)

    def _batch_needs_virial(
        self,
        task_key: str,
        label_dict: dict[str, Any],
    ) -> bool:
        loss = self.losses[task_key]
        if not isinstance(loss, EnergyLoss) or not loss.has_v:
            return False
        return bool(np.asarray(label_dict.get("find_virial", False)).any())

    @staticmethod
    def _to_input_tensor(key: str, value: Any) -> Any:
        if value is None:
            return None
        if key == "atype":
            return tf.convert_to_tensor(value, dtype=tf.int32)
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
            return tf.convert_to_tensor(value, dtype=tf.int64)
        return tf.convert_to_tensor(value, dtype=tf.float64)

    @staticmethod
    def _to_label_array(key: str, value: Any) -> Any:
        if value is None:
            return None
        if key in {"type", "natoms"}:
            return to_tensorflow_array(tf.convert_to_tensor(value, dtype=tf.int32))
        if key.startswith("find_"):
            return to_tensorflow_array(
                tf.constant(1.0 if bool(value) else 0.0, dtype=tf.float64)
            )
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer):
            return to_tensorflow_array(tf.convert_to_tensor(value, dtype=tf.int32))
        return to_tensorflow_array(tf.convert_to_tensor(value, dtype=tf.float64))

    def _assign_learning_rate(self, learning_rate: float) -> None:
        lr_attr = self.optimizer.learning_rate
        if hasattr(lr_attr, "assign"):
            lr_attr.assign(learning_rate)
        else:
            self.optimizer.learning_rate = learning_rate

    @staticmethod
    def _to_float(value: Any) -> float:
        tensor = to_tf_tensor(value)
        if tensor is not None:
            return float(tensor.numpy())
        return float(value)

    @classmethod
    def _more_loss_to_float(cls, more_loss: dict[str, Any]) -> dict[str, float]:
        return {
            key: cls._to_float(value)
            for key, value in more_loss.items()
            if "l2_" not in key
        }


DPTrainer = Trainer


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
    target_array = np.asarray(target) if _is_array_like(target) else None
    source_array = np.asarray(source) if _is_array_like(source) else None
    if target_array is None or source_array is None:
        return False
    return (
        target_array.shape == source_array.shape
        and target_array.dtype == source_array.dtype
    )


def _is_array_like(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _unique_variables(variables: list[Any] | tuple[Any, ...]) -> list[Any]:
    unique = []
    seen: set[int] = set()
    for variable in variables:
        variable_id = id(variable)
        if variable_id in seen:
            continue
        seen.add(variable_id)
        unique.append(variable)
    return unique
