# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import json
import logging
import os
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Union,
)

import h5py
import torch
import torch.distributed as dist
import torch.version
from torch.distributed.elastic.multiprocessing.errors import (
    record,
)

from deepmd import (
    __version__,
)
from deepmd.loggers.loggers import (
    set_log_handles,
)
from deepmd.main import (
    parse_args,
)
from deepmd.pt.infer import (
    inference,
)
from deepmd.pt.train import (
    training,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEVICE,
)
from deepmd.pt.utils.finetune import (
    change_finetune_model_params,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.summary import SummaryPrinter as BaseSummaryPrinter

log = logging.getLogger(__name__)


def get_trainer(
    config,
    init_model=None,
    restart_model=None,
    finetune_model=None,
    model_branch="",
    force_load=False,
    init_frz_model=None,
):
    # Initialize DDP
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
        assert dist.is_nccl_available()
        dist.init_process_group(backend="nccl")

    multi_task = "model_dict" in config["model"]
    ckpt = init_model if init_model is not None else restart_model
    config["model"] = change_finetune_model_params(
        ckpt,
        finetune_model,
        config["model"],
        multi_task=multi_task,
        model_branch=model_branch,
    )
    shared_links = None
    if multi_task:
        config["model"], shared_links = preprocess_shared_params(config["model"])

    def prepare_trainer_input_single(
        model_params_single, data_dict_single, loss_dict_single, suffix=""
    ):
        training_dataset_params = data_dict_single["training_data"]
        type_split = False
        if model_params_single["descriptor"]["type"] in ["se_e2_a"]:
            type_split = True
        validation_dataset_params = data_dict_single["validation_data"]
        training_systems = training_dataset_params["systems"]
        validation_systems = validation_dataset_params["systems"]
        # stat files
        stat_file_path_single = data_dict_single.get("stat_file", None)
        if stat_file_path_single is not None:
            if Path(stat_file_path_single).is_dir():
                raise ValueError(
                    f"stat_file should be a file, not a directory: {stat_file_path_single}"
                )
            if not Path(stat_file_path_single).is_file():
                with h5py.File(stat_file_path_single, "w") as f:
                    pass
            stat_file_path_single = DPPath(stat_file_path_single, "a")

        # validation and training data
        validation_data_single = DpLoaderSet(
            validation_systems,
            validation_dataset_params["batch_size"],
            model_params_single,
        )
        if ckpt or finetune_model:
            train_data_single = DpLoaderSet(
                training_systems,
                training_dataset_params["batch_size"],
                model_params_single,
            )
        else:
            train_data_single = DpLoaderSet(
                training_systems,
                training_dataset_params["batch_size"],
                model_params_single,
            )
        return (
            train_data_single,
            validation_data_single,
            stat_file_path_single,
        )

    if not multi_task:
        (
            train_data,
            validation_data,
            stat_file_path,
        ) = prepare_trainer_input_single(
            config["model"], config["training"], config["loss"]
        )
    else:
        train_data, validation_data, stat_file_path = {}, {}, {}
        for model_key in config["model"]["model_dict"]:
            (
                train_data[model_key],
                validation_data[model_key],
                stat_file_path[model_key],
            ) = prepare_trainer_input_single(
                config["model"]["model_dict"][model_key],
                config["training"]["data_dict"][model_key],
                config["loss_dict"][model_key],
                suffix=f"_{model_key}",
            )

    trainer = training.Trainer(
        config,
        train_data,
        stat_file_path=stat_file_path,
        validation_data=validation_data,
        init_model=init_model,
        restart_model=restart_model,
        finetune_model=finetune_model,
        force_load=force_load,
        shared_links=shared_links,
        init_frz_model=init_frz_model,
    )
    return trainer


class SummaryPrinter(BaseSummaryPrinter):
    """Summary printer for PyTorch."""

    def is_built_with_cuda(self) -> bool:
        """Check if the backend is built with CUDA."""
        return torch.version.cuda is not None

    def is_built_with_rocm(self) -> bool:
        """Check if the backend is built with ROCm."""
        return torch.version.hip is not None

    def get_compute_device(self) -> str:
        """Get Compute device."""
        return str(DEVICE)

    def get_ngpus(self) -> int:
        """Get the number of GPUs."""
        return torch.cuda.device_count()

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "Backend": "PyTorch",
            "PT ver": f"v{torch.__version__}-g{torch.version.git_version[:11]}",
        }


def train(FLAGS):
    log.info("Configuration path: %s", FLAGS.INPUT)
    SummaryPrinter()()
    with open(FLAGS.INPUT) as fin:
        config = json.load(fin)
    trainer = get_trainer(
        config,
        FLAGS.init_model,
        FLAGS.restart,
        FLAGS.finetune,
        FLAGS.model_branch,
        FLAGS.force_load,
        FLAGS.init_frz_model,
    )
    trainer.run()


def freeze(FLAGS):
    model = torch.jit.script(
        inference.Tester(FLAGS.model, numb_test=1, head=FLAGS.head).model
    )
    torch.jit.save(
        model,
        FLAGS.output,
        {
            # TODO: _extra_files
        },
    )


@record
def main(args: Optional[Union[List[str], argparse.Namespace]] = None):
    if not isinstance(args, argparse.Namespace):
        FLAGS = parse_args(args=args)
    else:
        FLAGS = args

    set_log_handles(FLAGS.log_level, FLAGS.log_path, mpi_log=None)
    log.debug("Log handles were successfully set")
    log.info("DeepMD version: %s", __version__)

    if FLAGS.command == "train":
        train(FLAGS)
    elif FLAGS.command == "freeze":
        if Path(FLAGS.checkpoint_folder).is_dir():
            checkpoint_path = Path(FLAGS.checkpoint_folder)
            latest_ckpt_file = (checkpoint_path / "checkpoint").read_text()
            FLAGS.model = str(checkpoint_path.joinpath(latest_ckpt_file))
        else:
            FLAGS.model = FLAGS.checkpoint_folder
        FLAGS.output = str(Path(FLAGS.output).with_suffix(".pth"))
        freeze(FLAGS)
    else:
        raise RuntimeError(f"Invalid command {FLAGS.command}!")


if __name__ == "__main__":
    main()
