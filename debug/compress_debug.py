#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Debug script for model compression.

Equivalent to: dp --pt compress -i no.pth -o yes.pth -t input_torch.json

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


def compress_model() -> float:
    """Compress the model using the same parameters as the CLI command.

    dp --pt compress -i no.pth -o yes.pth -t input_torch.json

    Returns
    -------
    float
        Elapsed time for the compression in seconds.
    """
    # Import here to avoid module-level import restriction
    from deepmd.pt.entrypoints.compress import (
        enable_compression,
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

        # Model compression parameters
        input_file = "no.pth"
        output_file = "yes.pth"
        training_script = "input_torch.json"
        stride = 0.01  # default value
        extrapolate = 5  # default value
        check_frequency = -1  # default value (disabled)

        # Check if input files exist
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Input model file '{input_file}' not found in {work_dir}"
            )

        if not os.path.exists(training_script):
            raise FileNotFoundError(
                f"Training script '{training_script}' not found in {work_dir}"
            )

        log.debug(f"Input model: {input_file}")
        log.debug(f"Output model: {output_file}")
        log.debug(f"Training script: {training_script}")
        log.debug(f"Stride: {stride}")
        log.debug(f"Extrapolate: {extrapolate}")
        log.debug(f"Check frequency: {check_frequency}")

        log.debug("Starting model compression...")

        # Record time usage
        start_time = time.monotonic()

        # Call the compression function
        enable_compression(
            input_file=input_file,
            output=output_file,
            stride=stride,
            extrapolate=extrapolate,
            check_frequency=check_frequency,
            training_script=training_script,
        )

        elapsed_time = time.monotonic() - start_time

        # Print results (keep these as info level - these are the main results)
        log.info("Model compression completed successfully!")
        log.info(f"Compressed model saved to: {output_file}")
        log.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        return elapsed_time

    except Exception as e:
        log.error(f"Error during compression: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    compress_model()
