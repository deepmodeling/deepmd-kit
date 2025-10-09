#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Debug script for model compression.

Equivalent to: dp --pt compress -i no.pth -o yes.pth -t input_torch.json

This script can be run directly in VSCode with debugging capabilities.
"""

import logging
import os
import sys
from pathlib import (
    Path,
)

# Add the deepmd-kit root to Python path
deepmd_root = Path(__file__).parent.parent
sys.path.insert(0, str(deepmd_root))


def compress_model() -> None:
    """Compress the model using the same parameters as the CLI command.

    dp --pt compress -i no.pth -o yes.pth -t input_torch.json
    """
    # Import here to avoid module-level import restriction
    from deepmd.pt.entrypoints.compress import (
        enable_compression,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    # Set working directory to examples/water/se_e3_tebd
    work_dir = deepmd_root / "examples" / "water" / "se_e3_tebd"
    original_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        log.info(f"Changed to working directory: {work_dir}")

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

        log.info(f"Input model: {input_file}")
        log.info(f"Output model: {output_file}")
        log.info(f"Training script: {training_script}")
        log.info(f"Stride: {stride}")
        log.info(f"Extrapolate: {extrapolate}")
        log.info(f"Check frequency: {check_frequency}")

        log.info("Starting model compression...")

        # Call the compression function
        enable_compression(
            input_file=input_file,
            output=output_file,
            stride=stride,
            extrapolate=extrapolate,
            check_frequency=check_frequency,
            training_script=training_script,
        )

        log.info("Model compression completed successfully!")
        log.info(f"Compressed model saved to: {output_file}")

    except Exception as e:
        log.error(f"Error during compression: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    compress_model()
