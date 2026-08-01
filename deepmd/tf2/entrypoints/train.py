# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training entrypoint for the TensorFlow 2 eager backend."""

from __future__ import (
    annotations,
)

import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.train import (
    AbstractTrainEntrypoint,
    TrainEntrypointOptions,
    TrainingTaskConfig,
    iter_training_task_configs,
    make_task_maps,
    print_data_summaries,
)
from deepmd.tf2.env import (
    tf,
)
from deepmd.tf2.train.trainer import (
    DPTrainer,
)
from deepmd.tf2.utils.serialization import (
    serialize_from_file,
)
from deepmd.utils import random as dp_random
from deepmd.utils.data_system import (
    get_data,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

if TYPE_CHECKING:
    from deepmd.utils.stat_file import (
        StatFileSpec,
    )

__all__ = ["train", "update_sel"]

log = logging.getLogger(__name__)


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for TensorFlow 2."""

    def is_built_with_cuda(self) -> bool:
        return bool(tf.test.is_built_with_cuda())

    def is_built_with_rocm(self) -> bool:
        return False

    def get_compute_device(self) -> str:
        return "gpu" if tf.config.list_physical_devices("GPU") else "cpu"

    def get_ngpus(self) -> int:
        return len(tf.config.list_physical_devices("GPU"))

    def get_backend_info(self) -> dict:
        return {
            "Backend": "TensorFlow2",
            "TensorFlow ver": tf.__version__,
            "Eager mode": str(tf.executing_eagerly()),
        }

    def get_device_name(self) -> str | None:
        devices = tf.config.list_physical_devices("GPU")
        return devices[0].name if devices else None


class TF2TrainEntrypoint(AbstractTrainEntrypoint):
    """TensorFlow 2 implementation of the common training entrypoint pipeline."""

    def __init__(self) -> None:
        self.finetune_links: dict[str, Any] | None = None
        self.shared_links: dict[str, Any] | None = None

    def validate_options(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> None:
        if options.init_frz_model:
            raise NotImplementedError(
                "TF2 training does not support init_frz_model yet."
            )

    def preprocess_config(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
    ) -> dict[str, Any]:
        """Apply TF2 fine-tuning and pretrained-script preprocessing."""
        self.finetune_links = None
        self.shared_links = None
        if self.is_multi_task(config):
            if "RANDOM" in config["model"]["model_dict"]:
                raise ValueError("Model name can not be 'RANDOM' in multi-task mode!")
            if config["model"].get("shared_dict"):
                from deepmd.tf2.utils.multi_task import (
                    preprocess_shared_params,
                )

                config["model"], self.shared_links = preprocess_shared_params(
                    config["model"]
                )
        if options.finetune is not None:
            from deepmd.tf2.utils.finetune import (
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
            self.shared_links = model_data.get("shared_links")
        return config

    def update_neighbor_stat(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        *,
        multi_task: bool,
    ) -> tuple[dict[str, Any], float | dict[str, float | None] | None]:
        log.info(
            "Calculate neighbor statistics... "
            "(add --skip-neighbor-stat to skip this step)"
        )
        return update_sel(config, multi_task=multi_task)

    def print_summary(self) -> None:
        SummaryPrinter()()

    def run_training(
        self,
        config: dict[str, Any],
        options: TrainEntrypointOptions,
        neighbor_stat: Any,
    ) -> None:
        seed = config["training"].get("seed")
        dp_random.seed(None if seed is None else int(seed) % (2**32))

        def factory(
            task_config: TrainingTaskConfig,
        ) -> tuple[Any, Any | None, StatFileSpec]:
            type_map = list(task_config.model_params.get("type_map", []))
            ipt_type_map = type_map if type_map else None
            train_data = get_data(
                dict(task_config.training_data_params),
                None,
                ipt_type_map,
                None,
            )
            valid_data = None
            if task_config.validation_data_params is not None:
                valid_data = get_data(
                    dict(task_config.validation_data_params),
                    None,
                    train_data.type_map,
                    None,
                )
            return train_data, valid_data, task_config.stat_file_spec

        train_data_map, valid_data_map, stat_file_spec_map = make_task_maps(
            config,
            factory,
        )
        print_data_summaries(train_data_map, valid_data_map)

        trainer = DPTrainer(
            config,
            train_data_map,
            stat_file_spec=stat_file_spec_map,
            validation_data=valid_data_map,
            init_model=options.init_model,
            restart_model=options.restart,
            finetune_model=options.finetune,
            finetune_links=self.finetune_links,
            shared_links=self.shared_links,
            min_nbor_dist=neighbor_stat,
        )
        start_time = time.time()
        trainer.run()
        end_time = time.time()
        log.info("finished training")
        log.info("wall time: %.3f s", end_time - start_time)


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
    """Run DeePMD model training with TensorFlow 2 eager execution."""
    TF2TrainEntrypoint().run(
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
    jdata: dict[str, Any],
    *,
    multi_task: bool | None = None,
) -> tuple[dict[str, Any], float | dict[str, float | None] | None]:
    """Update descriptor selections from neighbor statistics when available."""
    jdata_cpy = jdata.copy()
    if multi_task is None:
        multi_task = "model_dict" in jdata["model"]
    min_nbor_dist: dict[str, float | None] = {}
    for task_config in iter_training_task_configs(jdata):
        type_map = task_config.model_params.get("type_map")
        train_data = get_data(
            dict(task_config.training_data_params),
            0,
            type_map,
            None,
        )
        updated_model, task_min_nbor_dist = BaseModel.update_sel(
            train_data,
            type_map,
            dict(task_config.model_params),
        )
        min_nbor_dist[task_config.key] = task_min_nbor_dist
        if multi_task:
            jdata_cpy["model"]["model_dict"][task_config.key] = updated_model
        else:
            jdata_cpy["model"] = updated_model
            return jdata_cpy, task_min_nbor_dist
    return jdata_cpy, min_nbor_dist
