# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import json
import logging
import os
from copy import (
    deepcopy,
)
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
from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.pt.infer import (
    inference,
)
from deepmd.pt.model.model import (
    BaseModel,
)
from deepmd.pt.train import (
    training,
)
from deepmd.pt.utils import (
    env,
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
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    process_systems,
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
    shared_links=None,
):
    multi_task = "model_dict" in config.get("model", {})

    # Initialize DDP
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
        assert dist.is_nccl_available()
        dist.init_process_group(backend="nccl")

    ckpt = init_model if init_model is not None else restart_model
    finetune_links = None
    if finetune_model is not None:
        config["model"], finetune_links = change_finetune_model_params(
            finetune_model,
            config["model"],
            model_branch=model_branch,
        )
    config["model"]["resuming"] = (finetune_model is not None) or (ckpt is not None)

    def prepare_trainer_input_single(
        model_params_single, data_dict_single, loss_dict_single, suffix="", rank=0
    ):
        training_dataset_params = data_dict_single["training_data"]
        type_split = False
        if model_params_single["descriptor"]["type"] in ["se_e2_a"]:
            type_split = True
        validation_dataset_params = data_dict_single.get("validation_data", None)
        validation_systems = (
            validation_dataset_params["systems"] if validation_dataset_params else None
        )
        training_systems = training_dataset_params["systems"]
        training_systems = process_systems(training_systems)
        if validation_systems is not None:
            validation_systems = process_systems(validation_systems)

        # stat files
        stat_file_path_single = data_dict_single.get("stat_file", None)
        if rank != 0:
            stat_file_path_single = None
        elif stat_file_path_single is not None:
            if not Path(stat_file_path_single).exists():
                if stat_file_path_single.endswith((".h5", ".hdf5")):
                    with h5py.File(stat_file_path_single, "w") as f:
                        pass
                else:
                    Path(stat_file_path_single).mkdir()
            stat_file_path_single = DPPath(stat_file_path_single, "a")

        # validation and training data
        validation_data_single = (
            DpLoaderSet(
                validation_systems,
                validation_dataset_params["batch_size"],
                model_params_single["type_map"],
            )
            if validation_systems
            else None
        )
        if ckpt or finetune_model:
            train_data_single = DpLoaderSet(
                training_systems,
                training_dataset_params["batch_size"],
                model_params_single["type_map"],
            )
        else:
            train_data_single = DpLoaderSet(
                training_systems,
                training_dataset_params["batch_size"],
                model_params_single["type_map"],
            )
        return (
            train_data_single,
            validation_data_single,
            stat_file_path_single,
        )

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if not multi_task:
        (
            train_data,
            validation_data,
            stat_file_path,
        ) = prepare_trainer_input_single(
            config["model"],
            config["training"],
            config["loss"],
            rank=rank,
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
                rank=rank,
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
        finetune_links=finetune_links,
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
            "Enable custom OP": ENABLE_CUSTOMIZED_OP,
        }


def train(FLAGS):
    log.info("Configuration path: %s", FLAGS.INPUT)
    SummaryPrinter()()
    with open(FLAGS.INPUT) as fin:
        config = json.load(fin)

    # update multitask config
    multi_task = "model_dict" in config["model"]
    shared_links = None
    if multi_task:
        config["model"], shared_links = preprocess_shared_params(config["model"])

    # argcheck
    if not multi_task:
        config = update_deepmd_input(config, warning=True, dump="input_v2_compat.json")
        config = normalize(config)

    # do neighbor stat
    if not FLAGS.skip_neighbor_stat:
        log.info(
            "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
        )
        if not multi_task:
            config["model"] = BaseModel.update_sel(config, config["model"])
        else:
            training_jdata = deepcopy(config["training"])
            training_jdata.pop("data_dict", {})
            training_jdata.pop("model_prob", {})
            for model_item in config["model"]["model_dict"]:
                fake_global_jdata = {
                    "model": deepcopy(config["model"]["model_dict"][model_item]),
                    "training": deepcopy(config["training"]["data_dict"][model_item]),
                }
                fake_global_jdata["training"].update(training_jdata)
                config["model"]["model_dict"][model_item] = BaseModel.update_sel(
                    fake_global_jdata, config["model"]["model_dict"][model_item]
                )

    with open(FLAGS.output, "w") as fp:
        json.dump(config, fp, indent=4)

    trainer = get_trainer(
        config,
        FLAGS.init_model,
        FLAGS.restart,
        FLAGS.finetune,
        FLAGS.model_branch,
        FLAGS.force_load,
        FLAGS.init_frz_model,
        shared_links=shared_links,
    )
    trainer.run()


def freeze(FLAGS):
    model = torch.jit.script(inference.Tester(FLAGS.model, head=FLAGS.head).model)
    if '"type": "dpa2"' in model.model_def_script:
        extra_files = {"type": "dpa2"}
    else:
        extra_files = {"type": "else"}
    torch.jit.save(
        model,
        FLAGS.output,
        extra_files,
    )


def show(FLAGS):
    if FLAGS.INPUT.split(".")[-1] == "pt":
        state_dict = torch.load(FLAGS.INPUT, map_location=env.DEVICE)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model_params = state_dict["_extra_state"]["model_params"]
    elif FLAGS.INPUT.split(".")[-1] == "pth":
        model_params_string = torch.jit.load(
            FLAGS.INPUT, map_location=env.DEVICE
        ).model_def_script
        model_params = json.loads(model_params_string)
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint file with a .pt extension "
            "or a frozen model with a .pth extension"
        )
    model_is_multi_task = "model_dict" in model_params
    log.info("This is a multitask model") if model_is_multi_task else log.info(
        "This is a singletask model"
    )

    if "model-branch" in FLAGS.ATTRIBUTES:
        #  The model must be multitask mode
        if not model_is_multi_task:
            raise RuntimeError(
                "The 'model-branch' option requires a multitask model."
                " The provided model does not meet this criterion."
            )
        model_branches = list(model_params["model_dict"].keys())
        log.info(f"Available model branches are {model_branches}")
    if "type-map" in FLAGS.ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                type_map = model_params["model_dict"][branch]["type_map"]
                log.info(f"The type_map of branch {branch} is {type_map}")
        else:
            type_map = model_params["type_map"]
            log.info(f"The type_map is {type_map}")
    if "descriptor" in FLAGS.ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                descriptor = model_params["model_dict"][branch]["descriptor"]
                log.info(f"The descriptor parameter of branch {branch} is {descriptor}")
        else:
            descriptor = model_params["descriptor"]
            log.info(f"The descriptor parameter is {descriptor}")
    if "fitting-net" in FLAGS.ATTRIBUTES:
        if model_is_multi_task:
            model_branches = list(model_params["model_dict"].keys())
            for branch in model_branches:
                fitting_net = model_params["model_dict"][branch]["fitting_net"]
                log.info(
                    f"The fitting_net parameter of branch {branch} is {fitting_net}"
                )
        else:
            fitting_net = model_params["fitting_net"]
            log.info(f"The fitting_net parameter is {fitting_net}")


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
    elif FLAGS.command == "show":
        show(FLAGS)
    else:
        raise RuntimeError(f"Invalid command {FLAGS.command}!")


if __name__ == "__main__":
    main()
