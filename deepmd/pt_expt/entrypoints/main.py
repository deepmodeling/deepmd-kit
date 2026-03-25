# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training entrypoint for the pt_expt backend."""

import argparse
import json
import logging
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import h5py

from deepmd.pt_expt.train import (
    training,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
    get_data,
    process_systems,
)
from deepmd.utils.path import (
    DPPath,
)

log = logging.getLogger(__name__)


def get_trainer(
    config: dict[str, Any],
    init_model: str | None = None,
    restart_model: str | None = None,
    finetune_model: str | None = None,
    finetune_links: dict | None = None,
) -> training.Trainer:
    """Build a :class:`training.Trainer` from a normalised config."""
    model_params = config["model"]
    training_params = config["training"]
    type_map = model_params["type_map"]

    # ----- training data ------------------------------------------------
    training_dataset_params = training_params["training_data"]
    training_systems = process_systems(
        training_dataset_params["systems"],
        patterns=training_dataset_params.get("rglob_patterns", None),
    )
    train_data = DeepmdDataSystem(
        systems=training_systems,
        batch_size=training_dataset_params["batch_size"],
        test_size=1,
        type_map=type_map,
        trn_all_set=True,
        sys_probs=training_dataset_params.get("sys_probs", None),
        auto_prob_style=training_dataset_params.get("auto_prob", "prob_sys_size"),
    )

    # ----- validation data ----------------------------------------------
    validation_data = None
    validation_dataset_params = training_params.get("validation_data", None)
    if validation_dataset_params is not None:
        val_systems = process_systems(
            validation_dataset_params["systems"],
            patterns=validation_dataset_params.get("rglob_patterns", None),
        )
        validation_data = DeepmdDataSystem(
            systems=val_systems,
            batch_size=validation_dataset_params["batch_size"],
            test_size=1,
            type_map=type_map,
            trn_all_set=True,
        )

    # ----- stat file path -----------------------------------------------
    stat_file_path = training_params.get("stat_file", None)
    if stat_file_path is not None:
        if not Path(stat_file_path).exists():
            if stat_file_path.endswith((".h5", ".hdf5")):
                with h5py.File(stat_file_path, "w"):
                    pass
            else:
                Path(stat_file_path).mkdir()
        stat_file_path = DPPath(stat_file_path, "a")

    trainer = training.Trainer(
        config,
        train_data,
        stat_file_path=stat_file_path,
        validation_data=validation_data,
        init_model=init_model,
        restart_model=restart_model,
        finetune_model=finetune_model,
        finetune_links=finetune_links,
    )
    return trainer


def train(
    input_file: str,
    init_model: str | None = None,
    restart: str | None = None,
    finetune: str | None = None,
    model_branch: str = "",
    use_pretrain_script: bool = False,
    skip_neighbor_stat: bool = False,
    output: str = "out.json",
) -> None:
    """Run training with the pt_expt backend.

    Parameters
    ----------
    input_file : str
        Path to the JSON configuration file.
    init_model : str or None
        Path to a checkpoint to initialise weights from.
    restart : str or None
        Path to a checkpoint to restart training from.
    finetune : str or None
        Path to a pretrained checkpoint to fine-tune from.
    model_branch : str
        Branch to select from a multi-task pretrained model.
    use_pretrain_script : bool
        If True, copy descriptor/fitting params from the pretrained model.
    skip_neighbor_stat : bool
        Skip neighbour statistics calculation.
    output : str
        Where to dump the normalised config.
    """
    import torch

    from deepmd.common import (
        j_loader,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )

    log.info("Configuration path: %s", input_file)
    config = j_loader(input_file)

    # suffix fix
    if init_model is not None and not init_model.endswith(".pt"):
        init_model += ".pt"
    if restart is not None and not restart.endswith(".pt"):
        restart += ".pt"

    # update fine-tuning config
    finetune_links = None
    if finetune is not None:
        from deepmd.pt_expt.utils.finetune import (
            get_finetune_rules,
        )

        config["model"], finetune_links = get_finetune_rules(
            finetune,
            config["model"],
            model_branch=model_branch,
            change_model_params=use_pretrain_script,
        )

    # update init_model config if --use-pretrain-script
    if init_model is not None and use_pretrain_script:
        init_state_dict = torch.load(init_model, map_location=DEVICE, weights_only=True)
        if "model" in init_state_dict:
            init_state_dict = init_state_dict["model"]
        config["model"] = init_state_dict["_extra_state"]["model_params"]

    # argcheck
    config = update_deepmd_input(config, warning=True, dump="input_v2_compat.json")
    config = normalize(config)

    # neighbour stat
    if not skip_neighbor_stat:
        log.info(
            "Calculate neighbor statistics... "
            "(add --skip-neighbor-stat to skip this step)"
        )
        type_map = config["model"].get("type_map")
        train_data = get_data(config["training"]["training_data"], 0, type_map, None)
        from deepmd.pt_expt.model import (
            BaseModel,
        )

        config["model"], _min_nbor_dist = BaseModel.update_sel(
            train_data, type_map, config["model"]
        )

    with open(output, "w") as fp:
        json.dump(config, fp, indent=4)

    trainer = get_trainer(
        config,
        init_model,
        restart,
        finetune_model=finetune,
        finetune_links=finetune_links,
    )
    trainer.run()


def freeze(
    model: str,
    output: str = "frozen_model.pte",
    head: str | None = None,
) -> None:
    """Freeze a pt_expt checkpoint into a .pte exported model.

    Parameters
    ----------
    model : str
        Path to the checkpoint file (.pt).
    output : str
        Path for the output .pte file.
    head : str or None
        Head to freeze in multi-task mode (not yet supported).
    """
    import torch

    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    state_dict = torch.load(model, map_location=DEVICE, weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]

    extra_state = state_dict.get("_extra_state")
    if not isinstance(extra_state, dict) or "model_params" not in extra_state:
        raise ValueError(
            f"Unsupported checkpoint format at '{model}': missing "
            "'_extra_state.model_params' in model state dict."
        )
    model_params = extra_state["model_params"]

    if head is not None and "model_dict" in model_params:
        raise NotImplementedError(
            "Multi-task freeze is not yet supported for the pt_expt backend."
        )

    m = get_model(model_params)
    wrapper = ModelWrapper(m)
    wrapper.load_state_dict(state_dict)
    m.eval()

    model_dict = m.serialize()
    deserialize_to_file(output, {"model": model_dict}, model_params=model_params)
    log.info("Saved frozen model to %s", output)


def change_bias(
    input_file: str,
    mode: str = "change",
    bias_value: list | None = None,
    datafile: str | None = None,
    system: str = ".",
    numb_batch: int = 0,
    model_branch: str | None = None,
    output: str | None = None,
) -> None:
    """Change the output bias of a pt_expt model.

    Parameters
    ----------
    input_file : str
        Path to the model file (.pt checkpoint or .pte frozen model).
    mode : str
        ``"change"`` or ``"set"``.
    bias_value : list or None
        User-defined bias values (one per type).
    datafile : str or None
        File listing data system paths.
    system : str
        Data system path (used when *datafile* is None).
    numb_batch : int
        Number of batches for statistics (0 = all).
    model_branch : str or None
        Branch name for multi-task models.
    output : str or None
        Output file path.
    """
    import torch

    from deepmd.common import (
        expand_sys_str,
    )
    from deepmd.dpmodel.common import (
        to_numpy_array,
    )
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.training import (
        get_additional_data_requirement,
        get_loss,
        model_change_out_bias,
    )
    from deepmd.pt_expt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
        serialize_from_file,
    )
    from deepmd.pt_expt.utils.stat import (
        make_stat_input,
    )

    if input_file.endswith(".pt"):
        old_state_dict = torch.load(input_file, map_location=DEVICE, weights_only=True)
        if "model" in old_state_dict:
            model_state_dict = old_state_dict["model"]
        else:
            model_state_dict = old_state_dict
        extra_state = model_state_dict.get("_extra_state")
        if not isinstance(extra_state, dict) or "model_params" not in extra_state:
            raise ValueError(
                f"Unsupported checkpoint format at '{input_file}': missing "
                "'_extra_state.model_params' in model state dict."
            )
        model_params = extra_state["model_params"]
    elif input_file.endswith((".pte", ".pt2")):
        pte_data = serialize_from_file(input_file)
        from deepmd.pt_expt.model.model import (
            BaseModel,
        )

        model_to_change = BaseModel.deserialize(pte_data["model"])
        model_params = None
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint file with a .pt extension "
            "or a frozen model with a .pte/.pt2 extension"
        )

    if mode == "change":
        bias_adjust_mode = "change-by-statistic"
    elif mode == "set":
        bias_adjust_mode = "set-by-statistic"
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'change' or 'set'.")

    if input_file.endswith(".pt"):
        multi_task = "model_dict" in model_params
        if multi_task:
            raise NotImplementedError(
                "Multi-task change-bias is not yet supported for the pt_expt backend."
            )
        type_map = model_params["type_map"]
        model = get_model(model_params)
        wrapper = ModelWrapper(model)
        wrapper.load_state_dict(model_state_dict)
        model_to_change = model

    if input_file.endswith((".pte", ".pt2")):
        type_map = model_to_change.get_type_map()

    if bias_value is not None:
        if "energy" not in model_to_change.model_output_type():
            raise ValueError("User-defined bias is only available for energy models!")
        if len(bias_value) != len(type_map):
            raise ValueError(
                f"The number of elements in the bias ({len(bias_value)}) must match "
                f"the number of types in type_map ({len(type_map)}): {type_map}."
            )
        old_bias = model_to_change.get_out_bias()
        bias_to_set = torch.tensor(
            bias_value, dtype=old_bias.dtype, device=old_bias.device
        ).view(old_bias.shape)
        model_to_change.set_out_bias(bias_to_set)
        log.info(
            f"Change output bias of {type_map!s} "
            f"from {to_numpy_array(old_bias).reshape(-1)!s} "
            f"to {to_numpy_array(bias_to_set).reshape(-1)!s}."
        )
    else:
        if datafile is not None:
            with open(datafile) as datalist:
                all_sys = datalist.read().splitlines()
        else:
            all_sys = expand_sys_str(system)
        data_systems = process_systems(all_sys)
        data = DeepmdDataSystem(
            systems=data_systems,
            batch_size=1,
            test_size=1,
            rcut=model_to_change.get_rcut(),
            type_map=type_map,
        )
        mock_loss = get_loss({"inference": True}, 1.0, len(type_map), model_to_change)
        data.add_data_requirements(mock_loss.label_requirement)
        data.add_data_requirements(get_additional_data_requirement(model_to_change))
        if numb_batch != 0:
            nbatches = numb_batch
        else:
            # Cap at the minimum across systems so no system wraps and
            # overweights short systems (matching PT behavior).
            nbatches = min(data.get_nbatches())
        sampled_data = make_stat_input(data, nbatches)
        model_to_change = model_change_out_bias(
            model_to_change, sampled_data, _bias_adjust_mode=bias_adjust_mode
        )

    if input_file.endswith(".pt"):
        output_path = (
            output if output is not None else input_file.replace(".pt", "_updated.pt")
        )
        wrapper = ModelWrapper(model_to_change)
        if "model" in old_state_dict:
            old_state_dict["model"] = wrapper.state_dict()
            old_state_dict["model"]["_extra_state"] = extra_state
        else:
            old_state_dict = wrapper.state_dict()
            old_state_dict["_extra_state"] = extra_state
        torch.save(old_state_dict, output_path)
    elif input_file.endswith((".pte", ".pt2")):
        output_path = (
            output
            if output is not None
            else input_file.replace(".pte", "_updated.pte").replace(
                ".pt2", "_updated.pt2"
            )
        )
        model_dict = model_to_change.serialize()
        deserialize_to_file(output_path, {"model": model_dict})
    log.info(f"Saved model to {output_path}")


def main(args: list[str] | argparse.Namespace | None = None) -> None:
    """Entry point for the pt_expt backend CLI.

    Parameters
    ----------
    args : list[str] | argparse.Namespace | None
        Command-line arguments or pre-parsed namespace.
    """
    from deepmd.loggers.loggers import (
        set_log_handles,
    )
    from deepmd.main import (
        parse_args,
    )

    if not isinstance(args, argparse.Namespace):
        FLAGS = parse_args(args=args)
    else:
        FLAGS = args

    set_log_handles(
        FLAGS.log_level,
        Path(FLAGS.log_path) if FLAGS.log_path else None,
        mpi_log=None,
    )
    log.info("DeePMD-kit backend: pt_expt (PyTorch Exportable)")

    if FLAGS.command == "train":
        train(
            input_file=FLAGS.INPUT,
            init_model=FLAGS.init_model,
            restart=FLAGS.restart,
            finetune=FLAGS.finetune,
            model_branch=FLAGS.model_branch,
            use_pretrain_script=FLAGS.use_pretrain_script,
            skip_neighbor_stat=FLAGS.skip_neighbor_stat,
            output=FLAGS.output,
        )
    elif FLAGS.command == "freeze":
        if Path(FLAGS.checkpoint_folder).is_dir():
            checkpoint_path = Path(FLAGS.checkpoint_folder)
            # pt_expt training saves a symlink "model.ckpt.pt" → latest ckpt
            default_ckpt = checkpoint_path / "model.ckpt.pt"
            if default_ckpt.exists():
                FLAGS.model = str(default_ckpt)
            else:
                raise FileNotFoundError(
                    f"Cannot find checkpoint in '{checkpoint_path}'. "
                    "Expected 'model.ckpt.pt' (created by pt_expt training)."
                )
        else:
            model_path = Path(FLAGS.checkpoint_folder)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint path '{model_path}' does not exist."
                )
            FLAGS.model = str(model_path)
        if not FLAGS.output.endswith((".pte", ".pt2")):
            FLAGS.output = str(Path(FLAGS.output).with_suffix(".pte"))
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
    elif FLAGS.command == "compress":
        from deepmd.pt_expt.entrypoints.compress import (
            enable_compression,
        )

        if not FLAGS.input.endswith((".pte", ".pt2")):
            FLAGS.input = str(Path(FLAGS.input).with_suffix(".pte"))
        if not FLAGS.output.endswith((".pte", ".pt2")):
            FLAGS.output = str(Path(FLAGS.output).with_suffix(".pte"))
        enable_compression(
            input_file=FLAGS.input,
            output=FLAGS.output,
            stride=FLAGS.step,
            extrapolate=FLAGS.extrapolate,
            check_frequency=FLAGS.frequency,
            training_script=FLAGS.training_script,
        )
    else:
        raise RuntimeError(
            f"Unsupported command '{FLAGS.command}' for the pt_expt backend."
        )
