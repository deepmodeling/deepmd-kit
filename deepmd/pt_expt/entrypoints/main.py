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
    )
    return trainer


def train(
    input_file: str,
    init_model: str | None = None,
    restart: str | None = None,
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
    skip_neighbor_stat : bool
        Skip neighbour statistics calculation.
    output : str
        Where to dump the normalised config.
    """
    from deepmd.common import (
        j_loader,
    )

    log.info("Configuration path: %s", input_file)
    config = j_loader(input_file)

    # suffix fix
    if init_model is not None and not init_model.endswith(".pt"):
        init_model += ".pt"
    if restart is not None and not restart.endswith(".pt"):
        restart += ".pt"

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

    trainer = get_trainer(config, init_model, restart)
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
    deserialize_to_file(output, {"model": model_dict})
    log.info("Saved frozen model to %s", output)


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
    else:
        raise RuntimeError(
            f"Unsupported command '{FLAGS.command}' for the pt_expt backend."
        )
