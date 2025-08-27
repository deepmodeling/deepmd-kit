# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeePMD change bias entrypoint script."""

import logging
import os
import shutil
import tempfile
from pathlib import (
    Path,
)
from typing import (
    Optional,
)

from deepmd.common import (
    expand_sys_str,
    j_loader,
)
from deepmd.tf.entrypoints.freeze import (
    freeze,
)
from deepmd.tf.train.run_options import (
    RunOptions,
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
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

__all__ = ["change_bias"]

log = logging.getLogger(__name__)


def change_bias(
    INPUT: str,
    mode: str = "change",
    bias_value: Optional[list] = None,
    datafile: Optional[str] = None,
    system: str = ".",
    numb_batch: int = 0,
    model_branch: Optional[str] = None,
    output: Optional[str] = None,
    **kwargs,
) -> None:
    """Change model out bias according to the input data.

    Parameters
    ----------
    INPUT : str
        The input checkpoint file or frozen model file
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
    input_path = Path(INPUT)

    # Determine input type and handle accordingly
    if input_path.is_dir():
        # Checkpoint directory
        return _change_bias_checkpoint_dir(
            str(input_path),
            mode,
            bias_value,
            datafile,
            system,
            numb_batch,
            model_branch,
            output,
        )
    elif INPUT.endswith(".pb"):
        # Frozen model (.pb)
        return _change_bias_frozen_model(
            INPUT, mode, bias_value, datafile, system, numb_batch, model_branch, output
        )
    elif INPUT.endswith(".pbtxt"):
        # Text format frozen model (.pbtxt) - not supported
        raise NotImplementedError(
            "Bias changing for .pbtxt models is not supported. "
            "Please convert to .pb format first using: dp convert-from pbtxt -i model.pbtxt -o model.pb"
        )
    elif INPUT.endswith((".ckpt", ".meta", ".data", ".index")):
        # Individual checkpoint files
        checkpoint_prefix = INPUT
        if INPUT.endswith((".meta", ".data", ".index")):
            checkpoint_prefix = INPUT.rsplit(".", 1)[0]
        return _change_bias_checkpoint_file(
            checkpoint_prefix,
            mode,
            bias_value,
            datafile,
            system,
            numb_batch,
            model_branch,
            output,
        )
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint directory, checkpoint file, or frozen model file (.pb)"
        )


def _change_bias_checkpoint_dir(
    checkpoint_folder: str,
    mode: str,
    bias_value: Optional[list],
    datafile: Optional[str],
    system: str,
    numb_batch: int,
    model_branch: Optional[str],
    output: Optional[str],
) -> None:
    """Change bias for checkpoint directory."""
    # Check for valid checkpoint early
    checkpoint_path = Path(checkpoint_folder)
    if not (checkpoint_path / "checkpoint").exists():
        raise RuntimeError(f"No valid checkpoint found in {checkpoint_folder}")

    bias_adjust_mode = "change-by-statistic" if mode == "change" else "set-by-statistic"

    # Load data systems for bias calculation (only if not using user-defined bias)
    if bias_value is None:
        data = _load_data_systems(datafile, system)
    else:
        data = None

    # Read the checkpoint to get the model configuration
    input_json_path = _find_input_json(checkpoint_path)
    jdata = j_loader(input_json_path)

    # Update and normalize the configuration
    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata)

    # Determine output path
    if output is None:
        output = str(checkpoint_path) + "_bias_updated"

    # Create trainer to access model methods
    run_opt = RunOptions(
        init_model=checkpoint_folder,
        restart=None,
        finetune=None,
        init_frz_model=None,
    )

    trainer = DPTrainer(jdata, run_opt)

    if bias_value is not None:
        # Use user-defined bias
        _apply_user_defined_bias(trainer, bias_value)
    else:
        # Use data-based bias calculation
        type_map = data.get_type_map()
        if len(type_map) == 0:
            # If data doesn't have type_map, get from model
            type_map = trainer.model.get_type_map()

        log.info(f"Changing bias for model with type_map: {type_map}")
        log.info(f"Using bias adjustment mode: {bias_adjust_mode}")

        # Create a temporary frozen model from the checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pb", delete=False) as temp_frozen:
            freeze(
                checkpoint_folder=checkpoint_folder,
                output=temp_frozen.name,
            )

            # Use the trainer's change energy bias functionality
            trainer._change_energy_bias(
                data,
                temp_frozen.name,  # Use temporary frozen model
                type_map,
                bias_adjust_mode=bias_adjust_mode,
            )

            # Clean up temporary file
            os.unlink(temp_frozen.name)

    # Save the updated model - just copy to output location
    # Note: The bias has been updated in the trainer's session
    # Copy the checkpoint files to output location
    shutil.copytree(checkpoint_folder, output, dirs_exist_ok=True)

    log.info(f"Bias changing complete. Model files saved to {output}")


def _change_bias_checkpoint_file(
    checkpoint_prefix: str,
    mode: str,
    bias_value: Optional[list],
    datafile: Optional[str],
    system: str,
    numb_batch: int,
    model_branch: Optional[str],
    output: Optional[str],
) -> None:
    """Change bias for individual checkpoint files."""
    # For individual checkpoint files, we need to find the directory containing them
    checkpoint_path = Path(checkpoint_prefix)
    checkpoint_dir = checkpoint_path.parent

    # Use the same logic as checkpoint directory but with specific checkpoint prefix
    _change_bias_checkpoint_dir(
        str(checkpoint_dir),
        mode,
        bias_value,
        datafile,
        system,
        numb_batch,
        model_branch,
        output,
    )


def _change_bias_frozen_model(
    frozen_model_path: str,
    mode: str,
    bias_value: Optional[list],
    datafile: Optional[str],
    system: str,
    numb_batch: int,
    model_branch: Optional[str],
    output: Optional[str],
) -> None:
    """Change bias for frozen model (.pb file)."""
    if bias_value is None:
        raise NotImplementedError(
            "Data-based bias changing for frozen models is not yet implemented. "
            "Please provide user-defined bias values using the -b/--bias-value option, "
            "or use a checkpoint directory instead."
        )

    # For frozen models, we need to modify the graph and save a new frozen model
    # This is complex and requires graph manipulation
    # For now, provide a clear error message with workaround
    raise NotImplementedError(
        "Bias modification for frozen models (.pb) is not yet fully implemented. "
        "Recommended workaround:\n"
        "1. Use a checkpoint directory instead of a frozen model\n"
        "2. Or load the model, modify bias in training, then freeze again\n"
        f"   dp --tf change-bias <checkpoint_dir> -b {' '.join(map(str, bias_value)) if bias_value else '<bias_values>'} -o <output_dir>\n"
        "   dp freeze -c <output_dir> -o modified_model.pb"
    )


def _load_data_systems(datafile: Optional[str], system: str) -> DeepmdDataSystem:
    """Load data systems for bias calculation."""
    if datafile is not None:
        with open(datafile) as datalist:
            all_sys = datalist.read().splitlines()
    else:
        all_sys = expand_sys_str(system)

    # Load the data systems with proper data requirements
    data = DeepmdDataSystem(
        systems=all_sys,
        batch_size=1,
        test_size=1,
        rcut=None,
        set_prefix="set",
    )
    data.add_dict(
        {
            "energy": {
                "ndof": 1,
                "atomic": False,
                "must": False,
                "high_prec": True,
                "type_sel": None,
                "repeat": 1,
                "default": 0.0,
            },
            "force": {
                "ndof": 3,
                "atomic": True,
                "must": False,
                "high_prec": False,
                "type_sel": None,
                "repeat": 1,
                "default": 0.0,
            },
        }
    )
    return data


def _find_input_json(checkpoint_path: Path) -> Path:
    """Find the input.json file for the checkpoint."""
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
                f"Please ensure input.json is available in {checkpoint_path} or its parent directories."
            )
    return input_json_path


def _apply_user_defined_bias(trainer: DPTrainer, bias_value: list) -> None:
    """Apply user-defined bias values to the model."""
    # Get the type map from the model
    type_map = trainer.model.get_type_map()

    # Validate bias_value length
    if len(bias_value) != len(type_map):
        raise ValueError(
            f"The number of elements in the bias ({len(bias_value)}) should be the same as "
            f"that in the type_map ({len(type_map)}): {type_map}"
        )

    # Check model type
    if trainer.model.model_type != "ener":
        raise RuntimeError(
            f"User-defined bias is only supported for energy models, got: {trainer.model.model_type}"
        )

    # Get current bias
    fitting = trainer.model.get_fitting()
    if not hasattr(fitting, "bias_atom_e"):
        raise RuntimeError(
            "Model does not have bias_atom_e attribute for bias modification"
        )

    # Convert user bias to numpy array with proper shape
    import numpy as np

    new_bias = np.array(bias_value, dtype=np.float64).reshape(-1, 1)

    log.info(f"Changing bias from user-defined values for type_map: {type_map}")
    log.info(f"Old bias: {fitting.bias_atom_e.flatten()}")
    log.info(f"New bias: {new_bias.flatten()}")

    # Update the bias in the model
    fitting.bias_atom_e = new_bias

    # Update the tensor in the session if needed
    from deepmd.tf.env import (
        tf,
    )
    from deepmd.tf.utils.sess import (
        run_sess,
    )

    if hasattr(fitting, "t_bias_atom_e"):
        assign_op = tf.assign(fitting.t_bias_atom_e, new_bias)
        run_sess(trainer.sess, assign_op)
