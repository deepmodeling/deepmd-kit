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

import numpy as np

# Add the deepmd-kit root to Python path
deepmd_root = Path(__file__).parent.parent
sys.path.insert(0, str(deepmd_root))


def test_model() -> float:
    """Test the model using the same parameters as the CLI command.

    dp --pt test -m model.ckpt.pt -s . -n 100 -f test_debug.txt

    Returns
    -------
    float
        Elapsed time for the testing in seconds.
    """
    # Import here to avoid module-level import restriction
    from deepmd.entrypoints.test import (
        test,
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

        log.debug(f"Model: {model_file}")
        log.debug(f"System directory: {system_dir}")
        log.debug(f"Number of test frames: {numb_test}")
        log.debug(f"Detail file: {detail_file}")
        log.debug(f"Atomic output: {atomic}")

        log.debug("Starting model testing...")

        # Record time usage
        start_time = time.monotonic()
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
        end_time = time.monotonic()
        elapsed_time = end_time - start_time

        # Print results (keep these as info level - these are the main results)
        log.info("Model testing completed successfully!")
        log.info(f"Test results saved to: {detail_file}")
        log.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        return elapsed_time

    except Exception as e:
        log.error(f"Error during testing: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Run testing 10 times and calculate average timing
    num_runs = 10
    times = []

    print(f"Running model testing {num_runs} times...")  # noqa: T201
    print("=" * 50)  # noqa: T201

    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}")  # noqa: T201
        print("-" * 20)  # noqa: T201
        elapsed_time = test_model()
        times.append(elapsed_time)

    # Calculate and display statistics
    print("\n" + "=" * 50)  # noqa: T201
    print("Timing Summary:")  # noqa: T201
    print("=" * 50)  # noqa: T201

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Average time: {avg_time:.2f} seconds")  # noqa: T201
    print(f"Min time: {min_time:.2f} seconds")  # noqa: T201
    print(f"Max time: {max_time:.2f} seconds")  # noqa: T201
    print(f"Std deviation: {np.std(times):.2f} seconds")  # noqa: T201
    print(f"All times: {[f'{t:.2f}' for t in times]}")  # noqa: T201
