#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Debug script for model training.

Equivalent to: dp --pt train input_torch.json

This script can be run directly in VSCode with debugging capabilities.
"""

import logging
import os
import sys
import time
from pathlib import (
    Path,
)

# Add the deepmd-kit root to Python path
deepmd_root = Path(__file__).parent.parent
sys.path.insert(0, str(deepmd_root))


def train_model() -> float:
    """Train the model using the same parameters as the CLI command.

    dp --pt train input_torch.json

    Returns
    -------
    float
        Elapsed time for the training in seconds.
    """
    # Import here to avoid module-level import restriction
    from deepmd.pt.entrypoints.main import (
        train,
    )

    # Setup logging with timestamp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # Set working directory to examples/water/se_e3_tebd
    work_dir = deepmd_root / "examples" / "water" / "se_e3_tebd"
    original_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        log.debug(f"Changed to working directory: {work_dir}")

        # Training parameters
        input_file = "input_torch.json"
        init_model = None  # Start training from scratch
        restart = None  # No restart
        finetune = None  # No finetuning
        init_frz_model = None  # No frozen model initialization
        model_branch = "default"
        skip_neighbor_stat = True  # Calculate neighbor statistics
        use_pretrain_script = False  # Don't use pretrain script
        force_load = False  # Don't force load incompatible models
        compile_model = False  # Don't compile model (JIT will be used automatically)
        output = "out.json"  # Output configuration file

        # Check if input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Training input file '{input_file}' not found in {work_dir}"
            )

        log.debug(f"Input file: {input_file}")
        log.debug(f"Output config: {output}")
        log.debug(f"Skip neighbor stat: {skip_neighbor_stat}")
        log.debug(f"Compile model: {compile_model}")

        log.debug("Starting model training...")

        # Record time usage
        start_time = time.monotonic()

        # Call the training function
        train(
            input_file=input_file,
            init_model=init_model,
            restart=restart,
            finetune=finetune,
            init_frz_model=init_frz_model,
            model_branch=model_branch,
            skip_neighbor_stat=skip_neighbor_stat,
            use_pretrain_script=use_pretrain_script,
            force_load=force_load,
            compile_model=compile_model,
            output=output,
        )

        elapsed_time = time.monotonic() - start_time

        # Print results (keep these as info level - these are the main results)
        log.info("Model training completed successfully!")
        log.info(f"Output configuration saved to: {output}")
        log.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        return elapsed_time

    except Exception as e:
        log.error(f"Error during training: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    train_model()
