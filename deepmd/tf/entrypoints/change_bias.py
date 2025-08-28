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

import numpy as np

from deepmd.common import (
    expand_sys_str,
    j_loader,
)
from deepmd.tf.entrypoints.freeze import (
    freeze,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.infer import (
    DeepPotential,
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
from deepmd.tf.utils.sess import (
    run_sess,
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
    log_level: int = 0,
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
    log_level : int, optional
        The log level for output, by default 0
    """
    # Determine input type and handle accordingly
    if INPUT.endswith(".pb"):
        # Frozen model (.pb)
        return _change_bias_frozen_model(
            INPUT,
            mode,
            bias_value,
            datafile,
            system,
            numb_batch,
            model_branch,
            output,
            log_level,
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
            log_level,
        )
    else:
        raise RuntimeError(
            "The model provided must be a checkpoint file or frozen model file (.pb)"
        )


def _change_bias_checkpoint_file(
    checkpoint_prefix: str,
    mode: str,
    bias_value: Optional[list],
    datafile: Optional[str],
    system: str,
    numb_batch: int,
    model_branch: Optional[str],
    output: Optional[str],
    log_level: int,
) -> None:
    """Change bias for individual checkpoint files."""
    # Reset the default graph to avoid variable conflicts
    tf.reset_default_graph()

    checkpoint_path = Path(checkpoint_prefix)
    checkpoint_dir = checkpoint_path.parent

    # Check for valid checkpoint and find the actual checkpoint path
    checkpoint_state_file = checkpoint_dir / "checkpoint"
    if not checkpoint_state_file.exists():
        raise RuntimeError(f"No valid checkpoint found in {checkpoint_dir}")

    # Get the latest checkpoint path from the checkpoint state file
    checkpoint_state = tf.train.get_checkpoint_state(str(checkpoint_dir))
    if checkpoint_state is None or checkpoint_state.model_checkpoint_path is None:
        raise RuntimeError(f"No valid checkpoint state found in {checkpoint_dir}")

    # The model_checkpoint_path from get_checkpoint_state is the full path to the checkpoint
    actual_checkpoint_path = checkpoint_state.model_checkpoint_path

    bias_adjust_mode = "change-by-statistic" if mode == "change" else "set-by-statistic"

    # Read the checkpoint to get the model configuration
    input_json_path = _find_input_json(checkpoint_dir)
    jdata = j_loader(input_json_path)

    # Update and normalize the configuration
    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata)

    # Determine output path - should be a single model file
    if output is None:
        output = str(checkpoint_path.with_suffix(".pb"))
    elif not output.endswith(".pb"):
        output = output + ".pb"

    # Create trainer to access model methods
    run_opt = RunOptions(
        init_model=actual_checkpoint_path,  # Use the actual checkpoint file path
        restart=None,
        finetune=None,
        init_frz_model=None,
        log_level=log_level,
    )

    trainer = DPTrainer(jdata, run_opt)

    # Load data for bias calculation using trainer data requirements
    data = _load_data_systems(datafile, system, trainer)

    # Get stop_batch and origin_type_map like in train.py
    stop_batch = jdata.get("training", {}).get("numb_steps", 0)
    origin_type_map = jdata["model"].get("origin_type_map", None)
    if origin_type_map is not None and not origin_type_map:
        # get the type_map from data if not provided
        origin_type_map = data.get_type_map()

    try:
        # Build the model graph first with proper parameters, then initialize session
        # and restore variables from checkpoint - following train.py pattern
        trainer.build(data, stop_batch, origin_type_map=origin_type_map)
        trainer._init_session()

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

            # Read current bias values from the session (after variables are restored)
            _apply_data_based_bias(trainer, data, type_map, bias_adjust_mode)

        # Save the updated variables back to checkpoint format first
        # Create a separate directory for updated checkpoint to avoid polluting original
        updated_checkpoint_dir = checkpoint_dir / f"{checkpoint_path.name}_updated"
        updated_checkpoint_dir.mkdir(exist_ok=True)

        # Copy the input.json file to the new directory
        updated_input_json_path = updated_checkpoint_dir / "input.json"
        shutil.copy2(input_json_path, updated_input_json_path)

        updated_checkpoint_prefix = str(updated_checkpoint_dir / checkpoint_path.name)
        if hasattr(trainer, "saver") and trainer.saver is not None:
            log.info(f"Saving updated checkpoint to {updated_checkpoint_prefix}")
            trainer.saver.save(trainer.sess, updated_checkpoint_prefix)

            # Create a new checkpoint state file in the updated directory
            updated_checkpoint_state_file = updated_checkpoint_dir / "checkpoint"
            with open(updated_checkpoint_state_file, "w") as f:
                f.write(f'model_checkpoint_path: "{checkpoint_path.name}"\n')
                f.write(f'all_model_checkpoint_paths: "{checkpoint_path.name}"\n')

        # Then save the updated model as a frozen model using the updated checkpoint directory
        freeze(
            checkpoint_folder=str(updated_checkpoint_dir),
            output=output,
        )

        log.info(f"Bias changing complete. Model saved to {output}")

    finally:
        # Ensure session is properly closed
        if hasattr(trainer, "sess") and trainer.sess is not None:
            trainer.sess.close()


def _change_bias_frozen_model(
    frozen_model_path: str,
    mode: str,
    bias_value: Optional[list],
    datafile: Optional[str],
    system: str,
    numb_batch: int,
    model_branch: Optional[str],
    output: Optional[str],
    log_level: int,
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


def _load_data_systems(
    datafile: Optional[str], system: str, trainer: DPTrainer
) -> DeepmdDataSystem:
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
    # Use the data requirements from the trainer model instead of hardcoding them
    data.add_data_requirements(trainer.data_requirements)
    return data


def _find_input_json(checkpoint_dir: Path) -> Path:
    """Find the input.json file for the checkpoint."""
    input_json_path = checkpoint_dir / "input.json"
    if not input_json_path.exists():
        # Look for input.json in parent directories or common locations
        for parent in checkpoint_dir.parents:
            potential_input = parent / "input.json"
            if potential_input.exists():
                input_json_path = potential_input
                break
        else:
            raise RuntimeError(
                f"Cannot find input.json configuration file needed to load the model. "
                f"Please ensure input.json is available in {checkpoint_dir} or its parent directories."
            )
    return input_json_path


def _apply_data_based_bias(
    trainer: DPTrainer, data: DeepmdDataSystem, type_map: list, bias_adjust_mode: str
) -> None:
    """Apply data-based bias calculation by reading current bias from session."""
    from deepmd.tf.env import (
        tf,
    )
    from deepmd.tf.fit.ener import (
        change_energy_bias_lower,
    )

    # Get the fitting object which contains the bias tensor
    fitting = trainer.model.get_fitting()
    if not hasattr(fitting, "t_bias_atom_e"):
        raise RuntimeError(
            "Model does not have t_bias_atom_e tensor for bias modification"
        )

    # Read current bias values from the session (these are the restored values)
    current_bias = run_sess(trainer.sess, fitting.t_bias_atom_e)

    log.info(f"Current bias values from session: {current_bias.flatten()}")

    # Create a temporary frozen model to use with change_energy_bias_lower
    with tempfile.NamedTemporaryFile(suffix=".pb", delete=False) as temp_frozen:
        freeze(
            checkpoint_folder=str(Path(trainer.run_opt.init_model).parent),
            output=temp_frozen.name,
        )

        try:
            # Create DeepPotential object for evaluation
            dp = DeepPotential(temp_frozen.name)

            # Use change_energy_bias_lower with the current bias values from session
            new_bias = change_energy_bias_lower(
                data,
                dp,
                type_map,  # origin_type_map
                type_map,  # full_type_map
                current_bias,  # Use the restored bias values
                bias_adjust_mode=bias_adjust_mode,
                ntest=1,
            )

            # Update the bias in the session
            if len(new_bias.shape) == 1:
                # 1D tensor, keep bias as 1D
                new_bias_tensor = new_bias.flatten()
            else:
                # 2D tensor, reshape to match
                new_bias_tensor = new_bias.reshape(-1, 1)

            assign_op = tf.assign(fitting.t_bias_atom_e, new_bias_tensor)
            run_sess(trainer.sess, assign_op)

            # Also update the numpy array in the fitting object for consistency
            fitting.bias_atom_e = new_bias

        finally:
            # Clean up temporary file
            os.unlink(temp_frozen.name)


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

    # Convert user bias to numpy array with proper shape matching the tensor
    new_bias = np.array(bias_value, dtype=np.float64)

    # Check the shape of the existing bias tensor to match it
    if hasattr(fitting, "t_bias_atom_e"):
        existing_shape = fitting.t_bias_atom_e.get_shape().as_list()
        if len(existing_shape) == 1:
            # 1D tensor, keep bias as 1D
            new_bias = new_bias.flatten()
        else:
            # 2D tensor, reshape to match
            new_bias = new_bias.reshape(-1, 1)
    else:
        # If no tensor, use the fitting.bias_atom_e shape
        new_bias = new_bias.reshape(fitting.bias_atom_e.shape)

    log.info(f"Changing bias from user-defined values for type_map: {type_map}")
    log.info(f"Old bias: {fitting.bias_atom_e.flatten()}")
    log.info(f"New bias: {new_bias.flatten()}")

    # Update the bias in the model
    fitting.bias_atom_e = new_bias

    # Update the tensor in the session if needed
    if hasattr(fitting, "t_bias_atom_e"):
        assign_op = tf.assign(fitting.t_bias_atom_e, new_bias)
        run_sess(trainer.sess, assign_op)
