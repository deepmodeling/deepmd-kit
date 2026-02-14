# SPDX-License-Identifier: LGPL-3.0-or-later
"""Refactored PyTorch trainer with modular components.

This module provides a clean, extensible trainer implementation that
uses composition over monolithic design. It supports:

- Single-task and multi-task training
- Multiple optimizer types via strategy pattern
- Hook system for extensibility
- Clean separation of concerns
- Fine-tuning support
- Multi-task parameter sharing

Future extension points for multi-backend support:
- AbstractTrainingLoop can be extended for JAX/NumPy backends
- OptimizerFactory can support backend-specific optimizers
- DataManager can use backend-specific data loading
"""

from __future__ import (
    annotations,
)

import functools
import logging
import time
from copy import (
    deepcopy,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from deepmd.pt.loss import (
    DenoiseLoss,
    DOSLoss,
    EnergyHessianStdLoss,
    EnergySpinLoss,
    EnergyStdLoss,
    PropertyLoss,
    TaskLoss,
    TensorLoss,
)
from deepmd.pt.model.model import (
    get_model,
    get_zbl_model,
)
from deepmd.pt.train.checkpoint_manager import (
    CheckpointManager,
)
from deepmd.pt.train.config import (
    TrainingConfig,
)
from deepmd.pt.train.data_manager import (
    DataManager,
)
from deepmd.pt.train.hooks import (
    HookManager,
    TensorBoardHook,
    TimingHook,
)
from deepmd.pt.train.logger import (
    LossAccumulator,
    TrainingLogger,
)
from deepmd.pt.train.optimizer_factory import (
    OptimizerFactory,
)
from deepmd.pt.train.training_loop import (
    TrainingLoopFactory,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    dp_random,
)
from deepmd.pt.utils.env import (
    DEVICE,
    LOCAL_RANK,
)
from deepmd.pt.utils.learning_rate import (
    BaseLR,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.path import (
    DPH5Path,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from deepmd.pt.utils.dataloader import (
        DpLoaderSet,
    )

log = logging.getLogger(__name__)


def model_change_out_bias(
    _model: Any,
    _sample_func: Callable[[], Any],
    _bias_adjust_mode: str = "change-by-statistic",
) -> Any:
    """Change model output bias during fine-tuning.

    Parameters
    ----------
    _model : Any
        Model to modify.
    _sample_func : Callable[[], Any]
        Function to get sample data for statistics.
    _bias_adjust_mode : str
        Bias adjustment mode.

    Returns
    -------
    Any
        Modified model.
    """
    old_bias = deepcopy(_model.get_out_bias())
    _model.change_out_bias(
        _sample_func,
        bias_adjust_mode=_bias_adjust_mode,
    )
    new_bias = deepcopy(_model.get_out_bias())

    model_type_map = _model.get_type_map()
    log.info(
        f"Change output bias of {model_type_map!s} from {to_numpy_array(old_bias).reshape(-1)!s} to {to_numpy_array(new_bias).reshape(-1)!s}."
    )
    return _model


class Trainer:
    """Main trainer class orchestrating the training process.

    This is a refactored, modular trainer that delegates specific
    responsibilities to focused components:

    - TrainingConfig: Configuration management
    - DataManager: Data loading and iteration
    - OptimizerFactory: Optimizer creation
    - CheckpointManager: Model persistence
    - TrainingLoop: Core training step logic
    - HookManager: Extensibility hooks
    - TrainingLogger: Output formatting

    Parameters
    ----------
    config : dict[str, Any]
        Training configuration dictionary.
    training_data : DpLoaderSet | dict[str, DpLoaderSet]
        Training dataset(s).
    validation_data : DpLoaderSet | dict[str, DpLoaderSet] | None
        Validation dataset(s).
    stat_file_path : str | None
        Path to statistics file.
    init_model : str | None
        Path to initialization model.
    restart_model : str | None
        Path to checkpoint for restart.
    finetune_model : str | None
        Path to model for fine-tuning.
    init_frz_model : str | None
        Path to frozen model for initialization.
    force_load : bool
        Whether to force load mismatched checkpoints.
    shared_links : dict[str, Any] | None
        Parameter sharing configuration for multi-task.
    finetune_links : dict[str, Any] | None
        Fine-tuning mapping configuration.
    rank : int
        Distributed training rank.

    Attributes
    ----------
    config : TrainingConfig
        Parsed training configuration.
    data_manager : DataManager
        Data loading manager.
    checkpoint_manager : CheckpointManager
        Checkpoint persistence manager.
    hook_manager : HookManager
        Training hooks manager.
    """

    def __init__(
        self,
        config: dict[str, Any],
        training_data: DpLoaderSet | dict[str, DpLoaderSet],
        validation_data: DpLoaderSet | dict[str, DpLoaderSet] | None = None,
        stat_file_path: str | None = None,
        init_model: str | None = None,
        restart_model: str | None = None,
        finetune_model: str | None = None,
        init_frz_model: str | None = None,
        force_load: bool = False,
        shared_links: dict[str, Any] | None = None,
        finetune_links: dict[str, Any] | None = None,
        rank: int = 0,
    ) -> None:
        """Initialize the trainer with all components."""
        self.rank = rank
        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )

        # Determine resume/finetune state
        self.resume_model = restart_model or init_model or finetune_model
        self.is_restart = restart_model is not None
        self.is_finetune = finetune_model is not None
        self.finetune_update_stat = False

        # Parse configuration
        model_params = config.get("model", {})
        self.model_keys = (
            list(model_params.get("model_dict", {}).keys())
            if "model_dict" in model_params
            else ["Default"]
        )
        self.is_multitask = len(self.model_keys) > 1 and "model_dict" in model_params

        self.config = TrainingConfig.from_dict(config, self.model_keys)
        self.shared_links = shared_links
        self.finetune_links = finetune_links

        # Store for later use
        self._config_dict = config
        self._model_params = model_params
        self._stat_file_path = stat_file_path

        # Initialize components
        self._init_model(model_params, config)
        self._init_loss(config, model_params)

        # Compute statistics before data manager (need get_sample_func)
        self._compute_statistics_before_data(training_data, stat_file_path)

        self._init_data(training_data, validation_data, config, stat_file_path)
        self._init_optimizer_and_scheduler()
        self._init_distributed()
        self._setup_finetune(finetune_model)
        self._setup_multitask_shared_params(shared_links)
        self._init_checkpoint_manager()
        self._init_hooks()
        self._init_logger()

        # Load checkpoint if resuming
        self.start_step = 0
        if self.resume_model:
            self._load_resume_checkpoint(finetune_model)

        # Load frozen model if specified
        if init_frz_model:
            self._load_frozen_model(init_frz_model)

        # Initialize training loop
        self._init_training_loop()

        # Log model info
        if self.rank == 0:
            self._log_model_info()

    def _init_model(self, model_params: dict[str, Any], config: dict[str, Any]) -> None:
        """Initialize model(s)."""
        loss_params = self._get_loss_params(config)

        if self.is_multitask:
            self.model: dict[str, torch.nn.Module] | torch.nn.Module = {}
            for key in self.model_keys:
                model_dict = model_params["model_dict"][key]
                if loss_params and loss_params.get(key):
                    if self._is_hessian_loss(loss_params[key]):
                        model_dict = deepcopy(model_dict)
                        model_dict["hessian_mode"] = True
                self.model[key] = self._create_single_model(model_dict)
        else:
            if loss_params and self._is_hessian_loss(loss_params):
                model_params = deepcopy(model_params)
                model_params["hessian_mode"] = True
            self.model = self._create_single_model(model_params)

    def _create_single_model(self, model_params: dict[str, Any]) -> torch.nn.Module:
        """Create a single model instance."""
        if "use_srtab" in model_params:
            return get_zbl_model(deepcopy(model_params)).to(DEVICE)
        return get_model(deepcopy(model_params)).to(DEVICE)

    def _init_loss(self, config: dict[str, Any], model_params: dict[str, Any]) -> None:
        """Initialize loss function(s)."""
        if self.is_multitask:
            self.loss: dict[str, TaskLoss] | TaskLoss = {}
            for key in self.model_keys:
                loss_param = config["loss_dict"][key]
                lr_param = self._get_lr_for_task(config, key)
                ntypes = len(model_params["model_dict"][key]["type_map"])
                self.loss[key] = self._create_loss(
                    loss_param, lr_param, ntypes, self.model[key]
                )
        else:
            self.loss = self._create_loss(
                config["loss"],
                config["learning_rate"]["start_lr"],
                len(model_params["type_map"]),
                self.model,
            )

    def _create_loss(
        self,
        loss_params: dict[str, Any],
        start_lr: float,
        ntypes: int,
        model: torch.nn.Module,
    ) -> TaskLoss:
        """Create loss function instance."""
        loss_type = loss_params.get("type", "ener")

        if loss_type == "ener":
            if loss_params.get("start_pref_h", 0.0) > 0.0:
                loss_params["starter_learning_rate"] = start_lr
                return EnergyHessianStdLoss(**loss_params)
            else:
                loss_params["starter_learning_rate"] = start_lr
                return EnergyStdLoss(**loss_params)
        elif loss_type == "ener_spin":
            loss_params["starter_learning_rate"] = start_lr
            return EnergySpinLoss(**loss_params)
        elif loss_type == "denoise":
            loss_params["ntypes"] = ntypes
            return DenoiseLoss(**loss_params)
        elif loss_type == "dos":
            loss_params["starter_learning_rate"] = start_lr
            loss_params["numb_dos"] = model.model_output_def()["dos"].output_size
            return DOSLoss(**loss_params)
        elif loss_type == "tensor":
            model_output_type = model.model_output_type()
            if "mask" in model_output_type:
                model_output_type.pop(model_output_type.index("mask"))
            tensor_name = model_output_type[0]
            loss_params["tensor_size"] = model.model_output_def()[
                tensor_name
            ].output_size
            loss_params["starter_learning_rate"] = start_lr
            return TensorLoss(**loss_params)
        elif loss_type == "property":
            loss_params["task_dim"] = model.get_task_dim()
            loss_params["var_name"] = model.get_var_name()
            loss_params["intensive"] = model.get_intensive()
            loss_params["starter_learning_rate"] = start_lr
            return PropertyLoss(**loss_params)
        else:
            # Use TaskLoss.get_class_by_type for other types
            loss_params["starter_learning_rate"] = start_lr
            return TaskLoss.get_class_by_type(loss_type).get_loss(loss_params)

    def _compute_statistics_before_data(
        self,
        training_data: DpLoaderSet | dict[str, DpLoaderSet],
        stat_file_path: str | None,
    ) -> None:
        """Compute model statistics before creating data manager."""
        # Determine finetune_has_new_type
        finetune_has_new_type = False
        if self.is_finetune and self.finetune_links is not None:
            if self.is_multitask:
                for key in self.model_keys:
                    if self.finetune_links[key].get_has_new_type():
                        finetune_has_new_type = True
                        break
            else:
                finetune_has_new_type = self.finetune_links[
                    "Default"
                ].get_has_new_type()

        # Only compute stats on rank 0 and when not resuming (or finetune with new type)
        # For finetune, we need sample_func for model_change_out_bias
        should_compute = (
            not self.resume_model or finetune_has_new_type or self.is_finetune
        ) and self.rank == 0

        if not should_compute:
            self.get_sample_func = None
            return

        # Create get_sample_func for each model
        if self.is_multitask:
            self.get_sample_func = {}
            for key in self.model_keys:
                self.get_sample_func[key] = self._create_sample_func(
                    training_data[key],
                    self._config_dict["training"]["data_dict"][key]["training_data"],
                )

                # Compute statistics
                finetune_has_new_type_key = (
                    self.finetune_links[key].get_has_new_type()
                    if self.is_finetune and self.finetune_links
                    else False
                )

                # Get stat file path for this key
                stat_path_key = None
                if stat_file_path and isinstance(stat_file_path, dict):
                    stat_path_key = stat_file_path.get(key)
                elif stat_file_path:
                    stat_path_key = stat_file_path

                self.model[key].compute_or_load_stat(
                    sampled_func=self.get_sample_func[key],
                    stat_file_path=stat_path_key,
                )

                if isinstance(stat_path_key, DPH5Path):
                    stat_path_key.root.close()
        else:
            self.get_sample_func = self._create_sample_func(
                training_data,
                self._config_dict["training"]["training_data"],
            )

            self.model.compute_or_load_stat(
                sampled_func=self.get_sample_func,
                stat_file_path=stat_file_path,
            )

            if isinstance(stat_file_path, DPH5Path):
                stat_file_path.root.close()

    def _create_sample_func(
        self,
        training_data: DpLoaderSet,
        training_params: dict[str, Any],
    ) -> Callable[[], Any]:
        """Create sample function for statistics computation."""
        data_stat_nbatch = training_params.get("data_stat_nbatch", 10)

        @functools.cache
        def get_sample() -> Any:
            sampled = make_stat_input(
                training_data.systems,
                training_data.dataloaders,
                data_stat_nbatch,
            )
            return sampled

        return get_sample

    def _init_data(
        self,
        training_data: DpLoaderSet | dict[str, DpLoaderSet],
        validation_data: DpLoaderSet | dict[str, DpLoaderSet] | None,
        config: dict[str, Any],
        stat_file_path: str | None,
    ) -> None:
        """Initialize data manager and compute statistics."""
        # Add data requirements
        self._setup_data_requirements(training_data, validation_data)

        # Create data manager
        self.data_manager = DataManager(
            training_data,
            validation_data,
            config.get("training", {}),
            DEVICE,
        )

        # Print data summary
        self.data_manager.print_summary(self.rank)

    def _setup_data_requirements(
        self,
        training_data: DpLoaderSet | dict[str, DpLoaderSet],
        validation_data: DpLoaderSet | dict[str, DpLoaderSet] | None,
    ) -> None:
        """Setup data requirements for training and validation."""
        if self.is_multitask:
            for key in self.model_keys:
                data_req = self.loss[key].label_requirement
                data_req += self._get_additional_data_requirement(self.model[key])
                training_data[key].add_data_requirement(data_req)
                if validation_data and validation_data[key] is not None:
                    validation_data[key].add_data_requirement(data_req)

                training_data[key].preload_and_modify_all_data_torch()
                if validation_data and validation_data[key] is not None:
                    validation_data[key].preload_and_modify_all_data_torch()
        else:
            data_req = self.loss.label_requirement
            data_req += self._get_additional_data_requirement(self.model)
            training_data.add_data_requirement(data_req)
            if validation_data is not None:
                validation_data.add_data_requirement(data_req)

            training_data.preload_and_modify_all_data_torch()
            if validation_data is not None:
                validation_data.preload_and_modify_all_data_torch()

    def _get_additional_data_requirement(
        self, model: torch.nn.Module
    ) -> list[DataRequirementItem]:
        """Get additional data requirements from model."""
        requirements = []

        if model.get_dim_fparam() > 0:
            fparam_default = (
                model.get_default_fparam().cpu().numpy()
                if model.has_default_fparam()
                else 0.0
            )
            requirements.append(
                DataRequirementItem(
                    "fparam",
                    model.get_dim_fparam(),
                    atomic=False,
                    must=not model.has_default_fparam(),
                    default=fparam_default,
                )
            )

        if model.get_dim_aparam() > 0:
            requirements.append(
                DataRequirementItem(
                    "aparam", model.get_dim_aparam(), atomic=True, must=True
                )
            )

        has_spin = getattr(model, "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if has_spin:
            requirements.append(
                DataRequirementItem("spin", ndof=3, atomic=True, must=True)
            )

        return requirements

    def _init_optimizer_and_scheduler(self) -> None:
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer_factory = OptimizerFactory()

        # Create wrapper
        self.wrapper = ModelWrapper(
            self.model, self.loss, model_params=getattr(self, "_model_params", {})
        )

        # Create optimizer
        opt_config = self.config.get_optimizer_config()
        lr_config = self.config.get_lr_config()

        self.optimizer = self.optimizer_factory.create_optimizer(
            self.wrapper.parameters(),
            opt_config,
            lr_config,
        )

        # Create LR schedule
        self.lr_schedule = BaseLR(
            type="exp",
            start_lr=lr_config.start_lr,
            stop_lr=lr_config.stop_lr,
            decay_steps=lr_config.decay_steps,
            decay_rate=lr_config.decay_rate,
            stop_steps=lr_config.stop_steps,
        )

        # Create scheduler if supported
        if self.optimizer_factory.supports_scheduler(opt_config.opt_type):
            self.scheduler = self.optimizer_factory.create_scheduler(
                opt_config.opt_type,
                self.optimizer,
                self.config.warmup_steps,
                self.config.warmup_start_factor,
                self.lr_schedule,
                0,  # start_step, will be updated after loading checkpoint
            )
        else:
            self.scheduler = None

    def _setup_finetune(self, finetune_model: str | None) -> None:
        """Setup fine-tuning if applicable."""
        if finetune_model is None or self.finetune_links is None:
            return

        if self.is_multitask:
            for key in self.model_keys:
                finetune_rule_single = self.finetune_links[key]

                if finetune_rule_single.get_has_new_type():
                    self.finetune_update_stat = True

                if not finetune_rule_single.get_resuming():
                    self.model[key] = self._apply_finetune_to_model(
                        self.model[key],
                        finetune_rule_single,
                        self.get_sample_func[key]
                        if isinstance(self.get_sample_func, dict)
                        else self.get_sample_func,
                    )
        else:
            finetune_rule_single = self.finetune_links["Default"]
            self.model = self._apply_finetune_to_model(
                self.model,
                finetune_rule_single,
                self.get_sample_func,
            )

    def _apply_finetune_to_model(
        self,
        model: torch.nn.Module,
        finetune_rule: Any,
        sample_func: Callable[[], Any],
    ) -> torch.nn.Module:
        """Apply fine-tuning modifications to a model."""
        # Handle change_out_bias
        if not finetune_rule.get_random_fitting():
            model = model_change_out_bias(
                model,
                sample_func,
                _bias_adjust_mode="change-by-statistic",
            )
        return model

    def _setup_multitask_shared_params(
        self, shared_links: dict[str, Any] | None
    ) -> None:
        """Setup multi-task parameter sharing."""
        if shared_links is None or not self.is_multitask:
            return

        # Get data_stat_protect values
        data_stat_protect_values = [
            self._model_params["model_dict"][key].get("data_stat_protect", 1e-2)
            for key in self.model_keys
        ]

        # Check all values are the same
        assert all(
            abs(v - data_stat_protect_values[0]) < 1e-10
            for v in data_stat_protect_values
        ), (
            "Model key 'data_stat_protect' must be the same in each branch when multitask!"
        )

        # Compute model probabilities
        model_prob = np.zeros(len(self.model_keys), dtype=np.float32)
        for ii, model_key in enumerate(self.model_keys):
            # Get training data size for this model
            if hasattr(self, "data_manager") and self.data_manager:
                # Try to get from data_manager
                pass
            # Use uniform probability for now
            model_prob[ii] = 1.0

        model_prob = model_prob / np.sum(model_prob)
        model_key_prob_map = dict(zip(self.model_keys, model_prob))

        # Call share_params
        self.wrapper.share_params(
            shared_links,
            resume=(self.is_restart and not self.finetune_update_stat)
            or self.rank != 0,
            model_key_prob_map=model_key_prob_map,
            data_stat_protect=data_stat_protect_values[0],
        )

    def _init_distributed(self) -> None:
        """Initialize distributed training."""
        if dist.is_available() and dist.is_initialized():
            torch.cuda.set_device(LOCAL_RANK)
            self.wrapper = DDP(
                self.wrapper,
                device_ids=[LOCAL_RANK],
                find_unused_parameters=True,
                output_device=LOCAL_RANK,
            )

    def _init_checkpoint_manager(self) -> None:
        """Initialize checkpoint manager."""
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint,
            self.rank,
        )

    def _init_hooks(self) -> None:
        """Initialize hook manager and default hooks."""
        self.hook_manager = HookManager()

        # Register timing hook
        if self.config.display.time_training:
            self.hook_manager.register(TimingHook())

        # Register TensorBoard hook if enabled
        if self.config.display.tensorboard:
            self.hook_manager.register(
                TensorBoardHook(
                    log_dir=self.config.display.tensorboard_log_dir,
                    log_freq=self.config.display.tensorboard_freq,
                )
            )

    def _init_logger(self) -> None:
        """Initialize training logger."""
        self.logger = TrainingLogger(
            log_file=self.config.display.disp_file,
            is_multitask=self.is_multitask,
            model_keys=self.model_keys if self.is_multitask else None,
            rank=self.rank,
            restart=self.is_restart,
        )

        # Initialize loss accumulator if averaging enabled
        if self.config.display.disp_avg:
            self.loss_accumulator = LossAccumulator(
                self.is_multitask,
                self.model_keys if self.is_multitask else None,
            )
        else:
            self.loss_accumulator = None

    def _init_training_loop(self) -> None:
        """Initialize training loop based on optimizer type."""
        opt_config = self.config.get_optimizer_config()
        loss = self.loss["Default"] if self.is_multitask else self.loss

        loop_factory = TrainingLoopFactory(
            opt_config.opt_type,
            {
                "kf_start_pref_e": opt_config.kf_start_pref_e,
                "kf_limit_pref_e": opt_config.kf_limit_pref_e,
                "kf_start_pref_f": opt_config.kf_start_pref_f,
                "kf_limit_pref_f": opt_config.kf_limit_pref_f,
            },
            self.config.num_steps,
        )

        self.training_loop = loop_factory.create(
            self.wrapper,
            self.optimizer,
            loss,
            self.config.gradient_max_norm,
        )

    def _load_resume_checkpoint(self, finetune_model: str | None) -> None:
        """Load checkpoint for resume or finetune."""
        checkpoint = self.checkpoint_manager.load(
            self.resume_model,
            self.wrapper,
            self.optimizer if self.is_restart else None,
            strict=not self.is_finetune,
        )

        if self.is_restart:
            self.start_step = checkpoint.get("step", 0)
            log.info(f"Resuming training from step {self.start_step}")

            # Update scheduler start step
            if self.scheduler is not None:
                # Recreate scheduler with correct start step
                opt_config = self.config.get_optimizer_config()
                self.scheduler = self.optimizer_factory.create_scheduler(
                    opt_config.opt_type,
                    self.optimizer,
                    self.config.warmup_steps,
                    self.config.warmup_start_factor,
                    self.lr_schedule,
                    self.start_step,
                )
        else:
            log.info(f"Initialized model from {self.resume_model}")

    def _load_frozen_model(self, frozen_model_path: str) -> None:
        """Load frozen model for initialization."""
        log.info(f"Loading frozen model from {frozen_model_path}")
        frz_model = torch.jit.load(frozen_model_path, map_location=DEVICE)
        state = frz_model.state_dict()
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning(
                f"Non-strict load. Missing: {missing}, Unexpected: {unexpected}"
            )

    def _log_model_info(self) -> None:
        """Log model parameter count."""
        if self.is_multitask:
            log.warning("In multitask mode, parameters may be shared across tasks.")
            for key in self.model_keys:
                trainable, total = self._count_parameters(self.model[key])
                log.info(
                    f"Model Params [{key}]: {total / 1e6:.3f} M "
                    f"(Trainable: {trainable / 1e6:.3f} M)"
                )
        else:
            trainable, total = self._count_parameters(self.model)
            log.info(
                f"Model Params: {total / 1e6:.3f} M "
                f"(Trainable: {trainable / 1e6:.3f} M)"
            )

    @staticmethod
    def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        """Count model parameters."""
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    def _get_loss_params(self, config: dict[str, Any]) -> dict[str, Any] | None:
        """Extract loss parameters from config."""
        if self.is_multitask:
            return config.get("loss_dict")
        return config.get("loss")

    def _get_lr_for_task(self, config: dict[str, Any], task_key: str) -> float:
        """Get learning rate for a specific task."""
        if (
            config.get("learning_rate_dict")
            and task_key in config["learning_rate_dict"]
        ):
            return config["learning_rate_dict"][task_key]["start_lr"]
        return config["learning_rate"]["start_lr"]

    def _is_hessian_loss(self, loss_params: dict[str, Any]) -> bool:
        """Check if loss uses hessian."""
        return (
            loss_params.get("type", "ener") == "ener"
            and loss_params.get("start_pref_h", 0.0) > 0.0
        )

    def run(self) -> None:
        """Execute the training loop."""
        log.info(f"Starting training for {self.config.num_steps} steps")
        if self.world_size > 1:
            log.info(f"Rank: {self.rank}/{self.world_size}")

        # Training state
        start_time = time.time()
        total_train_time = 0.0
        timed_steps = 0
        last_display_step = self.start_step

        self.hook_manager.on_train_begin(
            {"start_step": self.start_step, "num_steps": self.config.num_steps}
        )

        try:
            for step in range(self.start_step, self.config.num_steps):
                # Select task for multi-task
                if self.is_multitask:
                    task_probs = self._compute_task_probs()
                    task_idx = dp_random.choice(
                        np.arange(len(self.model_keys), dtype=np.int_),
                        p=task_probs,
                    )
                    task_key = self.model_keys[task_idx]
                else:
                    task_key = "Default"

                # Execute training step
                step_result = self._training_step(step, task_key)

                # Update loss accumulator
                if self.loss_accumulator is not None:
                    self.loss_accumulator.update(step_result.more_loss, task_key)

                # Log and validate at display frequency
                display_step = step + 1
                if (
                    display_step % self.config.display.disp_freq == 0
                    or display_step == 1
                ):
                    self._log_and_validate(
                        step,
                        display_step,
                        task_key,
                        step_result,
                        start_time,
                        total_train_time,
                        timed_steps,
                        last_display_step,
                    )

                    # Update timing stats
                    current_time = time.time()
                    train_time = current_time - start_time
                    start_time = current_time

                    if display_step > self.start_step + 1:
                        total_train_time += train_time
                        timed_steps += min(
                            self.config.display.disp_freq,
                            display_step - last_display_step,
                        )
                        last_display_step = display_step

                # Save checkpoint
                if (
                    display_step % self.config.checkpoint.save_freq == 0
                    or display_step == self.config.num_steps
                ):
                    self._save_checkpoint(step, step_result.lr)

        except KeyboardInterrupt:
            log.info("Training interrupted by user")
        finally:
            self._finalize_training(total_train_time, timed_steps)

    def _training_step(self, step: int, task_key: str) -> Any:
        """Execute a single training step."""
        self.hook_manager.on_step_begin(step, {"task_key": task_key})

        # Get learning rates
        lr_config = self.config.get_lr_config(task_key)
        cur_lr = self.lr_schedule.value(step)

        if self.scheduler is not None:
            cur_lr = self.scheduler.get_last_lr()[0]
            if step < self.config.warmup_steps:
                pref_lr = lr_config.start_lr
            else:
                pref_lr = cur_lr
        else:
            pref_lr = cur_lr

        # Get batch
        input_dict, label_dict, log_dict = self.data_manager.get_train_batch(
            task_key if self.is_multitask else None
        )

        # Execute training step via training loop
        result = self.training_loop.step(
            input_dict,
            label_dict,
            cur_lr,
            pref_lr,
            task_key,
        )

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.hook_manager.on_step_end(
            step,
            {
                "loss": result.loss.item(),
                "lr": result.lr,
                "task_key": task_key,
                **result.more_loss,
            },
        )

        return result

    def _compute_task_probs(self) -> np.ndarray:
        """Compute sampling probabilities for multi-task."""
        # Check if model_prob is provided in config
        if hasattr(self, "_config_dict"):
            model_prob_dict = self._config_dict.get("training", {}).get(
                "model_prob", {}
            )
            if model_prob_dict:
                probs = np.array(
                    [model_prob_dict.get(key, 1.0) for key in self.model_keys]
                )
                return probs / probs.sum()

        # Default: uniform
        probs = np.ones(len(self.model_keys), dtype=np.float32)
        return probs / probs.sum()

    def _log_and_validate(
        self,
        step: int,
        display_step: int,
        task_key: str,
        step_result: Any,
        start_time: float,
        total_train_time: float,
        timed_steps: int,
        last_display_step: int,
    ) -> None:
        """Log training progress and run validation."""
        # Set eval mode for validation
        self.wrapper.eval()

        # Get training results
        if self.loss_accumulator is not None:
            train_results = self.loss_accumulator.get_all_averaged()
            self.loss_accumulator.reset()
        else:
            if self.is_multitask:
                train_results = {key: {} for key in self.model_keys}
                train_results[task_key] = {
                    k: v for k, v in step_result.more_loss.items() if "l2_" not in k
                }
            else:
                train_results = {
                    k: v for k, v in step_result.more_loss.items() if "l2_" not in k
                }

        # Run validation
        valid_results = self._run_validation(task_key)

        # Compute timing
        current_time = time.time()
        train_time = current_time - start_time
        if timed_steps > 0:
            eta = int(
                (self.config.num_steps - display_step) * total_train_time / timed_steps
            )
        else:
            eta = 0

        # Log
        self.logger.log_step(
            display_step,
            train_results,
            valid_results,
            step_result.lr,
            train_time if self.config.display.time_training else None,
            eta if self.config.display.time_training else None,
            task_key,
        )

        self.hook_manager.on_validation_end(
            step, {"train": train_results, "valid": valid_results}
        )

        # Restore train mode
        self.wrapper.train()

    def _run_validation(
        self, current_task_key: str
    ) -> dict[str, Any] | dict[str, dict[str, Any]] | None:
        """Run validation on all tasks."""
        self.hook_manager.on_validation_begin(0, {})

        if self.is_multitask:
            results: dict[str, dict[str, Any]] = {}
            for key in self.model_keys:
                results[key] = self._validate_task(key)
            return results
        else:
            return self._validate_task("Default")

    def _validate_task(self, task_key: str) -> dict[str, Any]:
        """Validate a single task."""
        num_batches = self.data_manager.get_valid_numb_batch(
            task_key if self.is_multitask else None
        )

        if num_batches == 0:
            return {}

        results: dict[str, float] = {}
        total_natoms = 0

        for _ in range(num_batches):
            input_dict, label_dict, _ = self.data_manager.get_valid_batch(
                task_key if self.is_multitask else None
            )

            if not input_dict:
                break

            # Note: Don't use torch.no_grad() here because the model
            # needs to compute gradients for force calculations via autograd
            _, loss, more_loss = self.wrapper(
                **input_dict,
                cur_lr=0.0,
                label=label_dict,
                task_key=task_key,
            )

            natoms = int(input_dict["atype"].shape[-1])
            total_natoms += natoms

            for k, v in more_loss.items():
                if "l2_" not in k and isinstance(v, (int, float)):
                    results[k] = results.get(k, 0.0) + v * natoms

        # Average by atom count
        if total_natoms > 0:
            results = {k: v / total_natoms for k, v in results.items()}

        return results

    def _save_checkpoint(self, step: int, lr: float) -> None:
        """Save training checkpoint."""
        path = self.checkpoint_manager.save(
            step + 1,
            self.wrapper,
            self.optimizer,
            lr,
        )

        if path:
            self.hook_manager.on_save_checkpoint(step, str(path), {"lr": lr})

    def _finalize_training(self, total_time: float, timed_steps: int) -> None:
        """Finalize training and cleanup."""
        # Save final checkpoint
        self._save_checkpoint(self.config.num_steps - 1, 0.0)

        # Log summary
        if timed_steps > 0:
            excluded = self.config.num_steps - self.start_step - timed_steps
            self.logger.log_summary(total_time, timed_steps, excluded)

        self.hook_manager.on_train_end(
            {"total_time": total_time, "timed_steps": timed_steps}
        )

        self.logger.close()

        log.info(
            f"Training completed. Model saved to {self.config.checkpoint.save_ckpt}"
        )

    def get_data(
        self, is_train: bool = True, task_key: str = "Default"
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Get a batch of data.

        This method is provided for backward compatibility and testing.

        Parameters
        ----------
        is_train : bool
            Whether to get training data (True) or validation data (False).
        task_key : str
            Task key for multi-task training.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
            (input_dict, label_dict, log_dict)
        """
        if is_train:
            return self.data_manager.get_train_batch(
                task_key if self.is_multitask else None
            )
        else:
            return self.data_manager.get_valid_batch(
                task_key if self.is_multitask else None
            )
