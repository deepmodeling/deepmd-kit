#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Debug script for model inference/testing.

Equivalent to: dp --pt test -m model.ckpt.pt -s data -n 100 -f test_debug.txt

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


def test_model() -> None:
    """Test the model using the same parameters as the CLI command.

    dp --pt test -m model.ckpt.pt -s . -n 100 -f test_debug.txt
    """
    # Import here to avoid module-level import restriction
    from deepmd.entrypoints.test import (
        test,
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

        # Test parameters
        model_file = "no.pth"  # Model file to test
        system_dir = "../data/data_3"  # Directory contains test data
        datafile = None  # Not using a datafile list
        train_json = None  # Not using training data for testing
        valid_json = None  # Not using validation data for testing
        numb_test = 100  # Number of test frames (0 means all)
        rand_seed = None  # No random seed
        shuffle_test = False  # Don't shuffle test data
        detail_file = "test_debug.txt"  # Output file for test details
        atomic = False  # Don't compute per-atom quantities
        head = None  # No specific task head for multi-task models

        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file '{model_file}' not found in {work_dir}"
            )

        # Set environment variable to limit batch size for testing
        os.environ["DP_INFER_BATCH_SIZE"] = "1024"

        log.info(f"Model: {model_file}")
        log.info(f"System directory: {system_dir}")
        log.info(f"Number of test frames: {numb_test}")
        log.info(f"Detail file: {detail_file}")
        log.info(f"Atomic output: {atomic}")

        log.info("Starting model testing...")

        # Record time usage
        start_time = time.time()
        # Call the test function
        test(
            model=model_file,
            system=system_dir,
            datafile=datafile,
            train_json=train_json,
            valid_json=valid_json,
            numb_test=numb_test,
            rand_seed=rand_seed,
            shuffle_test=shuffle_test,
            detail_file=detail_file,
            atomic=atomic,
            head=head,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time

        log.info("Model testing completed successfully!")
        log.info(f"Test results saved to: {detail_file}")
        log.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    except Exception as e:
        log.error(f"Error during testing: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    test_model()
