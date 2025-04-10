# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import copy
import json
import logging
from pathlib import (
    Path,
)
from typing import (
    Optional,
    Union,
)

import h5py
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import paddle.version

from deepmd import (
    __version__,
)
from deepmd.common import (
    expand_sys_str,
)
from deepmd.loggers.loggers import (
    set_log_handles,
)
from deepmd.main import (
    parse_args,
)
from deepmd.pd.infer import (
    inference,
)
from deepmd.pd.model.model import (
    BaseModel,
)
from deepmd.pd.train import (
    training,
)
from deepmd.pd.train.wrapper import (
    ModelWrapper,
)
from deepmd.pd.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pd.utils.env import (
    DEVICE,
    LOCAL_RANK,
)
from deepmd.pd.utils.finetune import (
    get_finetune_rules,
)
from deepmd.pd.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pd.utils.stat import (
    make_stat_input,
)
from deepmd.pd.utils.utils import (
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
    world_size = dist.get_world_size()
    if world_size > 1:
        assert paddle.version.nccl() != "0"
        fleet.init(is_collective=True)

    def prepare_trainer_input_single(
        model_params_single, data_dict_single, rank=0, seed=None
    ):
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
        # avoid the same batch sequence among devices
        rank_seed = [rank, seed % (2**32)] if seed is not None else None
        validation_data_single = (
            DpLoaderSet(
                validation_systems,
                validation_dataset_params["batch_size"],
                model_params_single["type_map"],
                seed=rank_seed,
            )
            if validation_systems
            else None
        )
        train_data_single = DpLoaderSet(
            training_systems,
            training_dataset_params["batch_size"],
            model_params_single["type_map"],
            seed=rank_seed,
        )
        return (
            train_data_single,
            validation_data_single,
            stat_file_path_single,
        )

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    data_seed = config["training"].get("seed", None)
    if not multi_task:
        (
            train_data,
            validation_data,
            stat_file_path,
        ) = prepare_trainer_input_single(
            config["model"],
            config["training"],
            rank=rank,
            seed=data_seed,
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
                seed=data_seed,
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
    """Summary printer for Paddle."""

    def is_built_with_cuda(self) -> bool:
        """Check if the backend is built with CUDA."""
        return paddle.device.is_compiled_with_cuda()

    def is_built_with_rocm(self) -> bool:
        """Check if the backend is built with ROCm."""
        return paddle.device.is_compiled_with_rocm()

    def get_compute_device(self) -> str:
        """Get Compute device."""
        return str(DEVICE)

    def get_ngpus(self) -> int:
        """Get the number of GPUs."""
        return paddle.device.cuda.device_count()

    def get_backend_info(self) -> dict:
        """Get backend information."""
        op_info = {}
        return {
            "Backend": "Paddle",
            "PD ver": f"v{paddle.__version__}-g{paddle.version.commit[:11]}",
            "Enable custom OP": False,
            **op_info,
        }


def train(
    input_file: str,
    init_model: Optional[str],
    restart: Optional[str],
    finetune: Optional[str],
    init_frz_model: Optional[str],
    model_branch: str,
    skip_neighbor_stat: bool = False,
    use_pretrain_script: bool = False,
    force_load: bool = False,
    output: str = "out.json",
) -> None:
    log.info("Configuration path: %s", input_file)
    if LOCAL_RANK == 0:
        SummaryPrinter()()
    with open(input_file) as fin:
        config = json.load(fin)
    # ensure suffix, as in the command line help, we say "path prefix of checkpoint files"
    if init_model is not None and not init_model.endswith(".pd"):
        init_model += ".pd"
    if restart is not None and not restart.endswith(".pd"):
        restart += ".pd"

    # update multitask config
    multi_task = "model_dict" in config["model"]
    shared_links = None
    if multi_task:
        config["model"], shared_links = preprocess_shared_params(config["model"])
        # handle the special key
        assert "RANDOM" not in config["model"]["model_dict"], (
            "Model name can not be 'RANDOM' in multi-task mode!"
        )

    # update fine-tuning config
    finetune_links = None
    if finetune is not None:
        config["model"], finetune_links = get_finetune_rules(
            finetune,
            config["model"],
            model_branch=model_branch,
            change_model_params=use_pretrain_script,
        )
    # update init_model or init_frz_model config if necessary
    if (init_model is not None or init_frz_model is not None) and use_pretrain_script:
        if init_model is not None:
            init_state_dict = paddle.load(init_model)
            if "model" in init_state_dict:
                init_state_dict = init_state_dict["model"]
            config["model"] = init_state_dict["_extra_state"]["model_params"]
        else:
            raise NotImplementedError("init_frz_model is not supported yet")

    # argcheck
    config = update_deepmd_input(config, warning=True, dump="input_v2_compat.json")
    config = normalize(config, multi_task=multi_task)

    # do neighbor stat
    min_nbor_dist = None
    if not skip_neighbor_stat:
        log.info(
            "Calculate neighbor statistics... (add --skip-neighbor-stat to skip this step)"
        )

        if not multi_task:
            type_map = config["model"].get("type_map")
            train_data = get_data(
                config["training"]["training_data"], 0, type_map, None
            )
            config["model"], min_nbor_dist = BaseModel.update_sel(
                train_data, type_map, config["model"]
            )
        else:
            min_nbor_dist = {}
            for model_item in config["model"]["model_dict"]:
                type_map = config["model"]["model_dict"][model_item].get("type_map")
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

    with open(output, "w") as fp:
        json.dump(config, fp, indent=4)

    trainer = get_trainer(
        config,
        init_model,
        restart,
        finetune,
        force_load,
        init_frz_model,
        shared_links=shared_links,
        finetune_links=finetune_links,
    )
    # save min_nbor_dist
    if min_nbor_dist is not None:
        if not multi_task:
            trainer.model.min_nbor_dist = paddle.to_tensor(
                min_nbor_dist,
                dtype=paddle.float64,
                place=DEVICE,
            )
        else:
            for model_item in min_nbor_dist:
                trainer.model[model_item].min_nbor_dist = paddle.to_tensor(
                    min_nbor_dist[model_item],
                    dtype=paddle.float64,
                    place=DEVICE,
                )
    trainer.run()


def freeze(
    model: str,
    output: str = "frozen_model.json",
    head: Optional[str] = None,
) -> None:
    paddle.set_flags(
        {
            "FLAGS_save_cf_stack_op": 1,
            "FLAGS_prim_enable_dynamic": 1,
            "FLAGS_enable_pir_api": 1,
        }
    )
    model = inference.Tester(model, head=head).model
    model.eval()
    from paddle.static import (
        InputSpec,
    )

    """ Example output shape and dtype of `model.forward`
    atom_energy: fetch_name_0 (1, 6, 1) float64
    atom_virial: fetch_name_1 (1, 6, 1, 9) float64
    energy: fetch_name_2 (1, 1) float64
    force: fetch_name_3 (1, 6, 3) float64
    mask: fetch_name_4 (1, 6) int32
    virial: fetch_name_5 (1, 9) float64
    """
    if hasattr(model, "forward"):
        model.forward = paddle.jit.to_static(
            model.forward,
            input_spec=[
                InputSpec([1, -1, 3], dtype="float64", name="coord"),  # coord
                InputSpec([1, -1], dtype="int64", name="atype"),  # atype
                InputSpec([1, 9], dtype="float64", name="box"),  # box
                None,  # fparam
                None,  # aparam
                True,  # do_atomic_virial
            ],
            full_graph=True,
        )
    """ Example output shape and dtype of `model.forward_lower`
    fetch_name_0: atom_energy [1, 192, 1] paddle.float64
    fetch_name_1: energy [1, 1] paddle.float64
    fetch_name_2: extended_force [1, 5184, 3] paddle.float64
    fetch_name_3: extended_virial [1, 5184, 1, 9] paddle.float64
    fetch_name_4: virial [1, 9] paddle.float64
    """
    if hasattr(model, "forward_lower"):
        model.forward_lower = paddle.jit.to_static(
            model.forward_lower,
            input_spec=[
                InputSpec([1, -1, 3], dtype="float64", name="coord"),  # extended_coord
                InputSpec([1, -1], dtype="int32", name="atype"),  # extended_atype
                InputSpec([1, -1, -1], dtype="int32", name="nlist"),  # nlist
                InputSpec([1, -1], dtype="int64", name="mapping"),  # mapping
                None,  # fparam
                None,  # aparam
                True,  # do_atomic_virial
                None,  # comm_dict
            ],
            full_graph=True,
        )
    if output.endswith(".json"):
        output = output[:-5]
    paddle.jit.save(
        model,
        path=output,
        skip_prune_program=True,
    )
    log.info(
        f"Paddle inference model has been exported to: {output}.json and {output}.pdiparams"
    )


def change_bias(
    input_file: str,
    mode: str = "change",
    bias_value: Optional[list] = None,
    datafile: Optional[str] = None,
    system: str = ".",
    numb_batch: int = 0,
    model_branch: Optional[str] = None,
    output: Optional[str] = None,
) -> None:
    if input_file.endswith(".pd"):
        old_state_dict = paddle.load(input_file)
        model_state_dict = copy.deepcopy(old_state_dict.get("model", old_state_dict))
        model_params = model_state_dict["_extra_state"]["model_params"]
    else:
        raise RuntimeError(
            "Paddle now do not support change bias directly from a freezed model file"
            "Please provided a checkpoint file with a .pd extension"
        )
    multi_task = "model_dict" in model_params
    bias_adjust_mode = "change-by-statistic" if mode == "change" else "set-by-statistic"
    if multi_task:
        assert model_branch is not None, (
            "For multitask model, the model branch must be set!"
        )
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
    if input_file.endswith(".pd"):
        wrapper = ModelWrapper(model)
        wrapper.set_state_dict(old_state_dict["model"])
    else:
        raise NotImplementedError("Only support .pd file")

    if bias_value is not None:
        # use user-defined bias
        assert model_to_change.model_type in ["ener"], (
            "User-defined bias is only available for energy model!"
        )
        assert len(bias_value) == len(type_map), (
            f"The number of elements in the bias should be the same as that in the type_map: {type_map}."
        )
        old_bias = model_to_change.get_out_bias()
        bias_to_set = paddle.to_tensor(
            bias_value, dtype=old_bias.dtype, place=old_bias.place
        ).reshape(old_bias.shape)
        model_to_change.set_out_bias(bias_to_set)
        log.info(
            f"Change output bias of {type_map!s} "
            f"from {to_numpy_array(old_bias).reshape(-1)!s} "
            f"to {to_numpy_array(bias_to_set).reshape(-1)!s}."
        )
        updated_model = model_to_change
    else:
        # calculate bias on given systems
        if datafile is not None:
            with open(datafile) as datalist:
                all_sys = datalist.read().splitlines()
        else:
            all_sys = expand_sys_str(system)
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
        nbatches = numb_batch if numb_batch != 0 else float("inf")
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

    if input_file.endswith(".pd"):
        output_path = (
            output if output is not None else input_file.replace(".pd", "_updated.pd")
        )
        wrapper = ModelWrapper(model)
        if "model" in old_state_dict:
            old_state_dict["model"] = wrapper.state_dict()
            old_state_dict["model"]["_extra_state"] = model_state_dict["_extra_state"]
        else:
            old_state_dict = wrapper.state_dict()
            old_state_dict["_extra_state"] = model_state_dict["_extra_state"]
        paddle.save(old_state_dict, output_path)
    else:
        raise NotImplementedError("Only support .pd file now")

    log.info(f"Saved model to {output_path}")


def main(args: Optional[Union[list[str], argparse.Namespace]] = None):
    if not isinstance(args, argparse.Namespace):
        FLAGS = parse_args(args=args)
    else:
        FLAGS = args

    set_log_handles(
        FLAGS.log_level,
        Path(FLAGS.log_path) if FLAGS.log_path else None,
        mpi_log=None,
    )
    log.debug("Log handles were successfully set")
    log.info("DeePMD version: %s", __version__)

    if FLAGS.command == "train":
        train(
            input_file=FLAGS.INPUT,
            init_model=FLAGS.init_model,
            restart=FLAGS.restart,
            finetune=FLAGS.finetune,
            init_frz_model=FLAGS.init_frz_model,
            model_branch=FLAGS.model_branch,
            skip_neighbor_stat=FLAGS.skip_neighbor_stat,
            use_pretrain_script=FLAGS.use_pretrain_script,
            force_load=FLAGS.force_load,
            output=FLAGS.output,
        )
    elif FLAGS.command == "freeze":
        if Path(FLAGS.checkpoint_folder).is_dir():
            checkpoint_path = Path(FLAGS.checkpoint_folder)
            latest_ckpt_file = (checkpoint_path / "checkpoint").read_text()
            FLAGS.model = str(checkpoint_path.joinpath(latest_ckpt_file))
        else:
            FLAGS.model = FLAGS.checkpoint_folder
        FLAGS.output = str(Path(FLAGS.output).with_suffix(".json"))
        freeze(model=FLAGS.model, output=FLAGS.output, head=FLAGS.head)
    elif FLAGS.command == "change-bias":
        change_bias(
            input_file=FLAGS.INPUT,
            mode=FLAGS.mode,
            bias_value=FLAGS.bias_value,
            datafile=FLAGS.datafile,
            system=FLAGS.system,
            numb_batch=FLAGS.numb_batch,
            model_branch=FLAGS.model_branch,
            output=FLAGS.output,
        )
    else:
        raise RuntimeError(f"Invalid command {FLAGS.command}!")


if __name__ == "__main__":
    main()
