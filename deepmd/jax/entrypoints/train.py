# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD training entrypoint script.

Can handle local training.
"""

import logging
import time
from typing import (
    Any,
)

from deepmd.dpmodel.train import (
    AbstractTrainEntrypoint,
    TrainEntrypointOptions,
    TrainingTaskConfig,
    iter_training_task_configs,
    make_task_maps,
    print_data_summaries,
)
from deepmd.jax.env import (
    jax,
    jax_export,
)
from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
)
from deepmd.jax.utils.serialization import (
    serialize_from_file,
)
from deepmd.jax.utils.update_sel import (
    use_jax_update_sel,
)
from deepmd.utils import random as dp_random
from deepmd.utils.data_system import (
    get_data,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

__all__ = ["train"]

log = logging.getLogger(__name__)


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for JAX."""

    def is_built_with_cuda(self) -> bool:
        """Check if the backend is built with CUDA."""
        return jax_export.default_export_platform() == "cuda"

    def is_built_with_rocm(self) -> bool:
        """Check if the backend is built with ROCm."""
        return jax_export.default_export_platform() == "rocm"

    def get_compute_device(self) -> str:
        """Get Compute device."""
        return jax.default_backend()

    def get_ngpus(self) -> int:
        """Get the number of GPUs."""
        return jax.device_count()

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "Backend": "JAX",
            "JAX ver": jax.__version__,
        }

    def get_device_name(self) -> str:
        """Get the name of the device."""
        devices = jax.devices()
        if devices:
            return devices[0].device_kind
        else:
            return "Unknown"


class JAXTrainEntrypoint(AbstractTrainEntrypoint):
    """JAX implementation of the common training entrypoint pipeline."""

    def __init__(self) -> None:
        self.finetune_links: dict[str, Any] | None = None

    def validate_options(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> None:
        """Validate currently unsupported JAX train features."""
        if options.init_frz_model:
            raise NotImplementedError(
                "JAX training does not support init_frz_model yet"
            )
        if self.is_multi_task(config) and config["model"].get("shared_dict"):
            raise NotImplementedError(
                "JAX multi-task training does not support shared_dict yet"
            )

    def preprocess_config(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> dict[str, Any]:
        """Apply JAX fine-tuning and pretrained-script preprocessing."""
        self.finetune_links = None
        if options.finetune is not None:
            from deepmd.jax.utils.finetune import (
                get_finetune_rules,
            )

            config["model"], self.finetune_links = get_finetune_rules(
                options.finetune,
                config["model"],
                model_branch=options.model_branch,
                change_model_params=options.use_pretrain_script,
            )
        elif options.init_model is not None and options.use_pretrain_script:
            model_data = serialize_from_file(options.init_model)
            config["model"] = model_data["model_def_script"]
        return config

    def update_neighbor_stat(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        *,
        multi_task: bool,
    ) -> tuple[dict[str, Any], float | dict[str, float | None] | None]:
        """Update JAX descriptor selections from neighbor statistics."""
        return update_sel(config, multi_task=multi_task)

    def print_summary(self) -> None:
        """Print JAX backend summary."""
        SummaryPrinter()()

    def run_training(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        neighbor_stat: float | dict[str, float | None] | None,
    ) -> None:
        """Build JAX data/trainer objects and run training."""
        # make necessary checks
        assert "training" in config

        model = DPTrainer(
            config,
            init_model=options.init_model,
            restart=options.restart,
            finetune_model=options.finetune,
            finetune_links=self.finetune_links,
        )
        if neighbor_stat is not None:
            model.set_min_nbor_dist(neighbor_stat)

        # init random seed of data systems
        seed = config["training"].get("seed", None)
        if seed is not None:
            seed += jax.process_index()
            seed = seed % (2**32)
        dp_random.seed(seed)

        def factory(
            task_config: TrainingTaskConfig,
        ) -> tuple[Any, Any | None, None]:
            task_model = model.models[task_config.key]
            type_map = task_model.get_type_map()
            ipt_type_map = type_map if len(type_map) > 0 else None
            train_data = get_data(
                dict(task_config.training_data_params),
                task_model.get_rcut(),
                ipt_type_map,
                None,
            )
            valid_data = None
            if task_config.validation_data_params is not None:
                valid_data = get_data(
                    dict(task_config.validation_data_params),
                    task_model.get_rcut(),
                    train_data.type_map,
                    None,
                )
            return train_data, valid_data, None

        train_data_map, valid_data_map, _ = make_task_maps(config, factory)
        print_data_summaries(train_data_map, valid_data_map)

        start_time = time.time()
        model.train(train_data_map, valid_data_map)
        end_time = time.time()
        log.info("finished training")
        log.info(f"wall time: {(end_time - start_time):.3f} s")


def train(
    *,
    INPUT: str,
    init_model: str | None,
    restart: str | None,
    output: str,
    init_frz_model: str | None,
    mpi_log: str,
    log_level: int,
    log_path: str | None,
    skip_neighbor_stat: bool = False,
    finetune: str | None = None,
    model_branch: str = "",
    use_pretrain_script: bool = False,
    **kwargs: Any,
) -> None:
    """Run DeePMD model training.

    Parameters
    ----------
    INPUT : str
        json/yaml control file
    init_model : Optional[str]
        path prefix of checkpoint files or None
    restart : Optional[str]
        path prefix of checkpoint files or None
    output : str
        path for dump file with arguments
    init_frz_model : str | None
        path to frozen model, or None if no frozen model is used
    mpi_log : str
        mpi logging mode
    log_level : int
        logging level defined by int 0-3
    log_path : Optional[str]
        logging file path or None if logs are to be output only to stdout
    skip_neighbor_stat : bool, default=False
        skip checking neighbor statistics
    finetune : Optional[str]
        path to pretrained model or None
    use_pretrain_script : bool
        Whether to use model script in pretrained model when doing init-model or init-frz-model.
        Note that this option is true and unchangeable for fine-tuning.
    **kwargs
        additional arguments

    Raises
    ------
    RuntimeError
        if the training command fails.
    """
    JAXTrainEntrypoint().run(
        TrainEntrypointOptions(
            input_file=INPUT,
            output=output,
            init_model=init_model,
            restart=restart,
            init_frz_model=init_frz_model,
            finetune=finetune,
            model_branch=model_branch,
            use_pretrain_script=use_pretrain_script,
            skip_neighbor_stat=skip_neighbor_stat,
        )
    )


def update_sel(
    jdata: dict,
    *,
    multi_task: bool | None = None,
) -> tuple[dict, float | dict[str, float | None] | None]:
    """Update descriptor selections from neighbor statistics when available."""
    log.info(
        "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
    )
    jdata_cpy = jdata.copy()
    if multi_task is None:
        multi_task = "model_dict" in jdata["model"]
    min_nbor_dist: dict[str, float | None] = {}
    with use_jax_update_sel():
        for task_config in iter_training_task_configs(jdata):
            type_map = task_config.model_params.get("type_map")
            train_data = get_data(
                dict(task_config.training_data_params),
                0,  # not used
                type_map,
                None,  # not used
            )
            updated_model, task_min_nbor_dist = BaseModel.update_sel(
                train_data, type_map, dict(task_config.model_params)
            )
            if multi_task:
                jdata_cpy["model"]["model_dict"][task_config.key] = updated_model
                min_nbor_dist[task_config.key] = task_min_nbor_dist
            else:
                jdata_cpy["model"] = updated_model
                return jdata_cpy, task_min_nbor_dist
    return jdata_cpy, min_nbor_dist
