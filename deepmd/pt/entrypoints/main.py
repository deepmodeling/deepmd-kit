# SPDX-License-Identifier: LGPL-3.0-or-later
import argparse
import json
import logging
import os

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import (
    record,
)

from deepmd import (
    __version__,
)
from deepmd.pt.infer import (
    inference,
)
from deepmd.pt.model.descriptor import (
    Descriptor,
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
from deepmd.pt.utils.finetune import (
    change_finetune_model_params,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)


def get_trainer(
    config,
    init_model=None,
    restart_model=None,
    finetune_model=None,
    model_branch="",
    force_load=False,
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
    config["model"]["resuming"] = (finetune_model is not None) or (ckpt is not None)
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

        # noise params
        noise_settings = None
        if loss_dict_single.get("type", "ener") == "denoise":
            noise_settings = {
                "noise_type": loss_dict_single.pop("noise_type", "uniform"),
                "noise": loss_dict_single.pop("noise", 1.0),
                "noise_mode": loss_dict_single.pop("noise_mode", "fix_num"),
                "mask_num": loss_dict_single.pop("mask_num", 8),
                "mask_prob": loss_dict_single.pop("mask_prob", 0.15),
                "same_mask": loss_dict_single.pop("same_mask", False),
                "mask_coord": loss_dict_single.pop("mask_coord", False),
                "mask_type": loss_dict_single.pop("mask_type", False),
                "max_fail_num": loss_dict_single.pop("max_fail_num", 10),
                "mask_type_idx": len(model_params_single["type_map"]) - 1,
            }
        # noise_settings = None

        # stat files
        hybrid_descrpt = model_params_single["descriptor"]["type"] == "hybrid"
        has_stat_file_path = True
        if not hybrid_descrpt:
            ### this design requires "rcut", "rcut_smth" and "sel" in the descriptor
            ### VERY BAD DESIGN!!!!
            ### not all descriptors provides these parameter in their constructor
            default_stat_file_name = Descriptor.get_stat_name(
                model_params_single["descriptor"]
            )
            model_params_single["stat_file_dir"] = data_dict_single.get(
                "stat_file_dir", f"stat_files{suffix}"
            )
            model_params_single["stat_file"] = data_dict_single.get(
                "stat_file", default_stat_file_name
            )
            model_params_single["stat_file_path"] = os.path.join(
                model_params_single["stat_file_dir"], model_params_single["stat_file"]
            )
            if not os.path.exists(model_params_single["stat_file_path"]):
                has_stat_file_path = False
        else:  ### need to remove this
            default_stat_file_name = []
            for descrpt in model_params_single["descriptor"]["list"]:
                default_stat_file_name.append(
                    f'stat_file_rcut{descrpt["rcut"]:.2f}_'
                    f'smth{descrpt["rcut_smth"]:.2f}_'
                    f'sel{descrpt["sel"]}_{descrpt["type"]}.npz'
                )
            model_params_single["stat_file_dir"] = data_dict_single.get(
                "stat_file_dir", f"stat_files{suffix}"
            )
            model_params_single["stat_file"] = data_dict_single.get(
                "stat_file", default_stat_file_name
            )
            assert isinstance(
                model_params_single["stat_file"], list
            ), "Stat file of hybrid descriptor must be a list!"
            stat_file_path = []
            for stat_file_path_item in model_params_single["stat_file"]:
                single_file_path = os.path.join(
                    model_params_single["stat_file_dir"], stat_file_path_item
                )
                stat_file_path.append(single_file_path)
                if not os.path.exists(single_file_path):
                    has_stat_file_path = False
            model_params_single["stat_file_path"] = stat_file_path

        # validation and training data
        validation_data_single = DpLoaderSet(
            validation_systems,
            validation_dataset_params["batch_size"],
            model_params_single,
            type_split=type_split,
            noise_settings=noise_settings,
        )
        if ckpt or finetune_model or has_stat_file_path:
            train_data_single = DpLoaderSet(
                training_systems,
                training_dataset_params["batch_size"],
                model_params_single,
                type_split=type_split,
                noise_settings=noise_settings,
            )
            sampled_single = None
        else:
            train_data_single = DpLoaderSet(
                training_systems,
                training_dataset_params["batch_size"],
                model_params_single,
                type_split=type_split,
            )
            data_stat_nbatch = model_params_single.get("data_stat_nbatch", 10)
            sampled_single = make_stat_input(
                train_data_single.systems,
                train_data_single.dataloaders,
                data_stat_nbatch,
            )
            if noise_settings is not None:
                train_data_single = DpLoaderSet(
                    training_systems,
                    training_dataset_params["batch_size"],
                    model_params_single,
                    type_split=type_split,
                    noise_settings=noise_settings,
                )
        return train_data_single, validation_data_single, sampled_single

    if not multi_task:
        train_data, validation_data, sampled = prepare_trainer_input_single(
            config["model"], config["training"], config["loss"]
        )
    else:
        train_data, validation_data, sampled = {}, {}, {}
        for model_key in config["model"]["model_dict"]:
            (
                train_data[model_key],
                validation_data[model_key],
                sampled[model_key],
            ) = prepare_trainer_input_single(
                config["model"]["model_dict"][model_key],
                config["training"]["data_dict"][model_key],
                config["loss_dict"][model_key],
                suffix=f"_{model_key}",
            )

    trainer = training.Trainer(
        config,
        train_data,
        sampled,
        validation_data=validation_data,
        init_model=init_model,
        restart_model=restart_model,
        finetune_model=finetune_model,
        force_load=force_load,
        shared_links=shared_links,
    )
    return trainer


def train(FLAGS):
    logging.info("Configuration path: %s", FLAGS.INPUT)
    with open(FLAGS.INPUT) as fin:
        config = json.load(fin)
    trainer = get_trainer(
        config,
        FLAGS.init_model,
        FLAGS.restart,
        FLAGS.finetune,
        FLAGS.model_branch,
        FLAGS.force_load,
    )
    trainer.run()


def test(FLAGS):
    trainer = inference.Tester(
        FLAGS.model,
        input_script=FLAGS.input_script,
        system=FLAGS.system,
        datafile=FLAGS.datafile,
        numb_test=FLAGS.numb_test,
        detail_file=FLAGS.detail_file,
        shuffle_test=FLAGS.shuffle_test,
        head=FLAGS.head,
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


# avoid logger conflicts of tf version
def clean_loggers():
    logger = logging.getLogger()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])


@record
def main(args=None):
    clean_loggers()
    logging.basicConfig(
        level=logging.WARNING if env.LOCAL_RANK else logging.INFO,
        format=f"%(asctime)-15s {os.environ.get('RANK') or ''} [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
    )
    logging.info("DeepMD version: %s", __version__)
    parser = argparse.ArgumentParser(
        description="A tool to manager deep models of potential energy surface."
    )
    subparsers = parser.add_subparsers(dest="command")
    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("INPUT", help="A Json-format configuration file.")
    parser_train_subgroup = train_parser.add_mutually_exclusive_group()
    parser_train_subgroup.add_argument(
        "-i",
        "--init-model",
        type=str,
        default=None,
        help="Initialize the model by the provided checkpoint.",
    )
    parser_train_subgroup.add_argument(
        "-r",
        "--restart",
        type=str,
        default=None,
        help="Restart the training from the provided checkpoint.",
    )
    parser_train_subgroup.add_argument(
        "-t",
        "--finetune",
        type=str,
        default=None,
        help="Finetune the frozen pretrained model.",
    )
    train_parser.add_argument(
        "-m",
        "--model-branch",
        type=str,
        default="",
        help="Model branch chosen for fine-tuning if multi-task. If not specified, it will re-init the fitting net.",
    )
    train_parser.add_argument(
        "--force-load",
        action="store_true",
        help="Force load from ckpt, other missing tensors will init from scratch",
    )

    test_parser = subparsers.add_parser("test", help="Test a model.")
    test_parser_subgroup = test_parser.add_mutually_exclusive_group()
    test_parser_subgroup.add_argument(
        "-s",
        "--system",
        default=None,
        type=str,
        help="The system dir. Recursively detect systems in this directory",
    )
    test_parser_subgroup.add_argument(
        "-f",
        "--datafile",
        default=None,
        type=str,
        help="The path to file of test list.",
    )
    test_parser_subgroup.add_argument(
        "-i",
        "--input-script",
        default=None,
        type=str,
        help="The path to the input script, the validation systems will be tested.",
    )
    test_parser.add_argument(
        "-m",
        "--model",
        default="model.pt",
        type=str,
        help="Model checkpoint to import",
    )
    test_parser.add_argument(
        "--head",
        default=None,
        type=str,
        help="Task head to test if in multi-task mode.",
    )
    test_parser.add_argument(
        "-n", "--numb-test", default=100, type=int, help="The number of data for test"
    )
    test_parser.add_argument(
        "-d",
        "--detail-file",
        type=str,
        default=None,
        help="The prefix to files where details of energy, force and virial accuracy/accuracy per atom will be written",
    )
    test_parser.add_argument(
        "--shuffle-test", action="store_true", default=False, help="Shuffle test data"
    )

    freeze_parser = subparsers.add_parser("freeze", help="Freeze a model.")
    freeze_parser.add_argument("model", help="Resumes from checkpoint.")
    freeze_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="frozen_model.pth",
        help="The frozen model path",
    )
    freeze_parser.add_argument(
        "--head",
        default=None,
        type=str,
        help="Task head to freeze if in multi-task mode.",
    )

    FLAGS = parser.parse_args(args)
    if FLAGS.command == "train":
        train(FLAGS)
    elif FLAGS.command == "test":
        test(FLAGS)
    elif FLAGS.command == "freeze":
        freeze(FLAGS)
    else:
        logging.error("Invalid command!")
        parser.print_help()


if __name__ == "__main__":
    main()
