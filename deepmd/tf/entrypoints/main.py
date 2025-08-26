# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD-Kit entry point module."""

import argparse
import logging
from pathlib import (
    Path,
)
from typing import (
    Optional,
    Union,
)

from deepmd.backend.suffix import (
    format_model_suffix,
)
from deepmd.common import (
    expand_sys_str,
)
from deepmd.main import (
    get_ll,
    main_parser,
    parse_args,
)
from deepmd.tf.common import (
    clear_session,
)
from deepmd.tf.entrypoints import (
    compress,
    convert,
    freeze,
    train_dp,
    transfer,
)
from deepmd.tf.loggers import (
    set_log_handles,
)
from deepmd.tf.nvnmd.entrypoints.train import (
    train_nvnmd,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

__all__ = ["get_ll", "main", "main_parser", "parse_args"]

log = logging.getLogger(__name__)


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
    """Change model out bias according to the input data.

    Parameters
    ----------
    input_file : str
        The input checkpoint folder or frozen model file
    mode : str, optional
        The mode for changing energy bias, by default "change"
    bias_value : Optional[list], optional
        The user defined value for each type, by default None
    datafile : Optional[str], optional
        The path to the datafile, by default None
    system : str, optional
        The system dir, by default "."
    numb_batch : int, optional
        The number of frames for bias changing, by default 0
    model_branch : Optional[str], optional
        Model branch chosen for changing bias if multi-task model, by default None
    output : Optional[str], optional
        The model after changing bias, by default None
    """
    import os
    from pathlib import (
        Path,
    )

    from deepmd.tf.train.trainer import (
        DPTrainer,
    )
    from deepmd.tf.utils.argcheck import (
        normalize,
    )
    from deepmd.tf.utils.compat import (
        update_deepmd_input,
    )

    input_path = Path(input_file)

    # Check if input is a checkpoint directory or frozen model
    if input_path.is_dir():
        # Checkpoint directory
        checkpoint_folder = str(input_path)
        # Check for valid checkpoint early
        if not (input_path / "checkpoint").exists():
            raise RuntimeError(f"No valid checkpoint found in {checkpoint_folder}")
    elif input_file.endswith((".pb", ".pbtxt")):
        # Frozen model - for now, not supported
        raise NotImplementedError(
            "Bias changing for frozen models (.pb/.pbtxt) is not yet implemented. "
            "Please provide a checkpoint directory instead. "
            "You can train a model to create checkpoints, then use this command "
            "to modify the bias, and finally freeze the modified model."
        )
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint directory or frozen model file (.pb/.pbtxt)"
        )

    bias_adjust_mode = "change-by-statistic" if mode == "change" else "set-by-statistic"

    if bias_value is not None:
        raise NotImplementedError(
            "User-defined bias setting is not yet implemented for TensorFlow models. "
            "Please use the data-based bias adjustment mode."
        )

    # Load data systems for bias calculation
    if datafile is not None:
        with open(datafile) as datalist:
            all_sys = datalist.read().splitlines()
    else:
        all_sys = expand_sys_str(system)

    # Load the data systems
    data = DeepmdDataSystem(
        systems=all_sys,
        batch_size=1,
        test_size=1,
        rcut=None,
        set_prefix="set",
    )

    # Read the checkpoint to get the model configuration
    checkpoint_path = Path(checkpoint_folder)

    # Find the input.json file or create a minimal config
    # We need this to reconstruct the model
    input_json_path = checkpoint_path / "input.json"
    if not input_json_path.exists():
        # Look for input.json in parent directories or common locations
        for parent in checkpoint_path.parents:
            potential_input = parent / "input.json"
            if potential_input.exists():
                input_json_path = potential_input
                break
        else:
            raise RuntimeError(
                f"Cannot find input.json configuration file needed to load the model. "
                f"Please ensure input.json is available in {checkpoint_folder} or its parent directories."
            )

    # Load the configuration
    with open(input_json_path) as f:
        import json

        jdata = json.load(f)

    # Update and normalize the configuration
    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata)

    # Determine output path
    if output is None:
        output = str(checkpoint_path) + "_bias_updated"

    # Create trainer to access model methods
    from deepmd.tf.train.run_options import (
        RunOptions,
    )

    run_opt = RunOptions(
        init_model=checkpoint_folder,
        restart=None,
        finetune=None,
        init_frz_model=None,
        train_data=all_sys,
        valid_data=None,
    )

    trainer = DPTrainer(jdata, run_opt)

    # Get the type map from the model
    type_map = data.get_type_map()
    if len(type_map) == 0:
        # If data doesn't have type_map, get from model
        type_map = trainer.model.get_type_map()

    log.info(f"Changing bias for model with type_map: {type_map}")
    log.info(f"Using bias adjustment mode: {bias_adjust_mode}")

    # Use the trainer's change energy bias functionality
    trainer._change_energy_bias(
        data,
        checkpoint_folder,  # Use checkpoint as frozen model path for compatibility
        type_map,
        bias_adjust_mode=bias_adjust_mode,
    )

    # Save the updated model
    import shutil

    shutil.copytree(checkpoint_folder, output, dirs_exist_ok=True)
    trainer.save_checkpoint(os.path.join(output, "model.ckpt"))

    log.info(f"Bias changing complete. Updated model saved to {output}")
    log.info(
        f"You can now freeze this model using: dp freeze -c {output} -o model_updated.pb"
    )


def main(args: Optional[Union[list[str], argparse.Namespace]] = None) -> None:
    """DeePMD-Kit entry point.

    Parameters
    ----------
    args : list[str] or argparse.Namespace, optional
        list of command line arguments, used to avoid calling from the subprocess,
        as it is quite slow to import tensorflow; if Namespace is given, it will
        be used directly

    Raises
    ------
    RuntimeError
        if no command was input
    """
    if args is not None:
        clear_session()

    if not isinstance(args, argparse.Namespace):
        args = parse_args(args=args)

    # do not set log handles for None, it is useless
    # log handles for train will be set separately
    # when the use of MPI will be determined in `RunOptions`
    if args.command not in (None, "train"):
        set_log_handles(args.log_level, Path(args.log_path) if args.log_path else None)

    dict_args = vars(args)

    if args.command == "train":
        train_dp(**dict_args)
    elif args.command == "freeze":
        dict_args["output"] = format_model_suffix(
            dict_args["output"], preferred_backend=args.backend, strict_prefer=True
        )
        freeze(**dict_args)
    elif args.command == "transfer":
        transfer(**dict_args)
    elif args.command == "compress":
        dict_args["input"] = format_model_suffix(
            dict_args["input"], preferred_backend=args.backend, strict_prefer=True
        )
        dict_args["output"] = format_model_suffix(
            dict_args["output"], preferred_backend=args.backend, strict_prefer=True
        )
        compress(**dict_args)
    elif args.command == "convert-from":
        convert(**dict_args)
    elif args.command == "change-bias":
        change_bias(
            input_file=dict_args["INPUT"],
            mode=dict_args["mode"],
            bias_value=dict_args["bias_value"],
            datafile=dict_args["datafile"],
            system=dict_args["system"],
            numb_batch=dict_args["numb_batch"],
            model_branch=dict_args["model_branch"],
            output=dict_args["output"],
        )
    elif args.command == "train-nvnmd":  # nvnmd
        train_nvnmd(**dict_args)
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")

    if args is not None:
        clear_session()
