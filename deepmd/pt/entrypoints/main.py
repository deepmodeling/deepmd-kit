# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import copy
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
from deepmd.common import (
    expand_sys_str,
)
from deepmd.env import (
    GLOBAL_CONFIG,
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
from deepmd.pt.train.wrapper import (
    ModelWrapper,
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
    get_finetune_rules,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    get_data,
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
    force_load=False,
    init_frz_model=None,
    shared_links=None,
    finetune_links=None,
):
    multi_task = "model_dict" in config.get("model", {})

    # Initialize DDP
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
        assert dist.is_nccl_available()
        dist.init_process_group(backend="nccl")

    def prepare_trainer_input_single(model_params_single, data_dict_single, rank=0):
        training_dataset_params = data_dict_single["training_data"]
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
        if ENABLE_CUSTOMIZED_OP:
            op_info = {
                "build with PT ver": GLOBAL_CONFIG["pt_version"],
                "build with PT inc": GLOBAL_CONFIG["pt_include_dir"].replace(";", "\n"),
                "build with PT lib": GLOBAL_CONFIG["pt_libs"].replace(";", "\n"),
            }
        else:
            op_info = {}
        return {
            "Backend": "PyTorch",
            "PT ver": f"v{torch.__version__}-g{torch.version.git_version[:11]}",
            "Enable custom OP": ENABLE_CUSTOMIZED_OP,
            **op_info,
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
        # handle the special key
        assert (
            "RANDOM" not in config["model"]["model_dict"]
        ), "Model name can not be 'RANDOM' in multi-task mode!"

    # update fine-tuning config
    finetune_links = None
    if FLAGS.finetune is not None:
        config["model"], finetune_links = get_finetune_rules(
            FLAGS.finetune,
            config["model"],
            model_branch=FLAGS.model_branch,
            change_model_params=FLAGS.use_pretrain_script,
        )
    # update init_model or init_frz_model config if necessary
    if (
        FLAGS.init_model is not None or FLAGS.init_frz_model is not None
    ) and FLAGS.use_pretrain_script:
        if FLAGS.init_model is not None:
            init_state_dict = torch.load(FLAGS.init_model, map_location=DEVICE)
            if "model" in init_state_dict:
                init_state_dict = init_state_dict["model"]
            config["model"] = init_state_dict["_extra_state"]["model_params"]
        else:
            config["model"] = json.loads(
                torch.jit.load(
                    FLAGS.init_frz_model, map_location=DEVICE
                ).get_model_def_script()
            )

    # argcheck
    config = update_deepmd_input(config, warning=True, dump="input_v2_compat.json")
    config = normalize(config, multi_task=multi_task)

    # do neighbor stat
    min_nbor_dist = None
    if not FLAGS.skip_neighbor_stat:
        log.info(
            "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
        )

        type_map = config["model"].get("type_map")
        if not multi_task:
            train_data = get_data(
                config["training"]["training_data"], 0, type_map, None
            )
            config["model"], min_nbor_dist = BaseModel.update_sel(
                train_data, type_map, config["model"]
            )
        else:
            min_nbor_dist = {}
            for model_item in config["model"]["model_dict"]:
                train_data = get_data(
                    config["training"]["data_dict"][model_item]["training_data"],
                    0,
                    type_map,
                    None,
                )
                config["model"]["model_dict"][model_item], min_nbor_dist[model_item] = (
                    BaseModel.update_sel(
                        train_data, type_map, config["model"]["model_dict"][model_item]
                    )
                )

    with open(FLAGS.output, "w") as fp:
        json.dump(config, fp, indent=4)

    trainer = get_trainer(
        config,
        FLAGS.init_model,
        FLAGS.restart,
        FLAGS.finetune,
        FLAGS.force_load,
        FLAGS.init_frz_model,
        shared_links=shared_links,
        finetune_links=finetune_links,
    )
    # save min_nbor_dist
    if min_nbor_dist is not None:
        if not multi_task:
            trainer.model.min_nbor_dist = min_nbor_dist
        else:
            for model_item in min_nbor_dist:
                trainer.model[model_item].min_nbor_dist = min_nbor_dist[model_item]
    trainer.run()


def freeze(FLAGS):
    model = torch.jit.script(inference.Tester(FLAGS.model, head=FLAGS.head).model)
    extra_files = {}
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
        model_branches += ["RANDOM"]
        log.info(
            f"Available model branches are {model_branches}, "
            f"where 'RANDOM' means using a randomly initialized fitting net."
        )
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


def change_bias(FLAGS):
    if FLAGS.INPUT.endswith(".pt"):
        old_state_dict = torch.load(FLAGS.INPUT, map_location=env.DEVICE)
        model_state_dict = copy.deepcopy(old_state_dict.get("model", old_state_dict))
        model_params = model_state_dict["_extra_state"]["model_params"]
    elif FLAGS.INPUT.endswith(".pth"):
        old_model = torch.jit.load(FLAGS.INPUT, map_location=env.DEVICE)
        model_params_string = old_model.get_model_def_script()
        model_params = json.loads(model_params_string)
        old_state_dict = old_model.state_dict()
        model_state_dict = old_state_dict
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint file with a .pt extension "
            "or a frozen model with a .pth extension"
        )
    multi_task = "model_dict" in model_params
    model_branch = FLAGS.model_branch
    bias_adjust_mode = (
        "change-by-statistic" if FLAGS.mode == "change" else "set-by-statistic"
    )
    if multi_task:
        assert (
            model_branch is not None
        ), "For multitask model, the model branch must be set!"
        assert model_branch in model_params["model_dict"], (
            f"For multitask model, the model branch must be in the 'model_dict'! "
            f"Available options are : {list(model_params['model_dict'].keys())}."
        )
        log.info(f"Changing out bias for model {model_branch}.")
    model = training.get_model_for_wrapper(model_params)
    type_map = (
        model_params["type_map"]
        if not multi_task
        else model_params["model_dict"][model_branch]["type_map"]
    )
    model_to_change = model if not multi_task else model[model_branch]
    if FLAGS.INPUT.endswith(".pt"):
        wrapper = ModelWrapper(model)
        wrapper.load_state_dict(old_state_dict["model"])
    else:
        # for .pth
        model.load_state_dict(old_state_dict)

    if FLAGS.bias_value is not None:
        # use user-defined bias
        assert model_to_change.model_type in [
            "ener"
        ], "User-defined bias is only available for energy model!"
        assert (
            len(FLAGS.bias_value) == len(type_map)
        ), f"The number of elements in the bias should be the same as that in the type_map: {type_map}."
        old_bias = model_to_change.get_out_bias()
        bias_to_set = torch.tensor(
            FLAGS.bias_value, dtype=old_bias.dtype, device=old_bias.device
        ).view(old_bias.shape)
        model_to_change.set_out_bias(bias_to_set)
        log.info(
            f"Change output bias of {type_map!s} "
            f"from {to_numpy_array(old_bias).reshape(-1)!s} "
            f"to {to_numpy_array(bias_to_set).reshape(-1)!s}."
        )
        updated_model = model_to_change
    else:
        # calculate bias on given systems
        if FLAGS.datafile is not None:
            with open(FLAGS.datafile) as datalist:
                all_sys = datalist.read().splitlines()
        else:
            all_sys = expand_sys_str(FLAGS.system)
        data_systems = process_systems(all_sys)
        data_single = DpLoaderSet(
            data_systems,
            1,
            type_map,
        )
        mock_loss = training.get_loss(
            {"inference": True}, 1.0, len(type_map), model_to_change
        )
        data_requirement = mock_loss.label_requirement
        data_requirement += training.get_additional_data_requirement(model_to_change)
        data_single.add_data_requirement(data_requirement)
        nbatches = FLAGS.numb_batch if FLAGS.numb_batch != 0 else float("inf")
        sampled_data = make_stat_input(
            data_single.systems,
            data_single.dataloaders,
            nbatches,
        )
        updated_model = training.model_change_out_bias(
            model_to_change, sampled_data, _bias_adjust_mode=bias_adjust_mode
        )

    if not multi_task:
        model = updated_model
    else:
        model[model_branch] = updated_model

    if FLAGS.INPUT.endswith(".pt"):
        output_path = (
            FLAGS.output
            if FLAGS.output is not None
            else FLAGS.INPUT.replace(".pt", "_updated.pt")
        )
        wrapper = ModelWrapper(model)
        if "model" in old_state_dict:
            old_state_dict["model"] = wrapper.state_dict()
            old_state_dict["model"]["_extra_state"] = model_state_dict["_extra_state"]
        else:
            old_state_dict = wrapper.state_dict()
            old_state_dict["_extra_state"] = model_state_dict["_extra_state"]
        torch.save(old_state_dict, output_path)
    else:
        # for .pth
        output_path = (
            FLAGS.output
            if FLAGS.output is not None
            else FLAGS.INPUT.replace(".pth", "_updated.pth")
        )
        model = torch.jit.script(model)
        torch.jit.save(
            model,
            output_path,
            {},
        )
    log.info(f"Saved model to {output_path}")


@record
def main(args: Optional[Union[List[str], argparse.Namespace]] = None):
    if not isinstance(args, argparse.Namespace):
        FLAGS = parse_args(args=args)
    else:
        FLAGS = args

    set_log_handles(FLAGS.log_level, FLAGS.log_path, mpi_log=None)
    log.debug("Log handles were successfully set")
    log.info("DeePMD version: %s", __version__)

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
    elif FLAGS.command == "change-bias":
        change_bias(FLAGS)
    else:
        raise RuntimeError(f"Invalid command {FLAGS.command}!")


if __name__ == "__main__":
    main()
