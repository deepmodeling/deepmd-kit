#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inference performance profiling script with TensorBoard visualization.

This script focuses on identifying performance hotspots in DeePMD-kit inference
by breaking down the computation into detailed components and visualizing results.
"""

import logging
import os
import sys
import time
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
import torch  # noqa: TID253
from torch.profiler import record_function  # noqa: TID253
from torch.utils.tensorboard import SummaryWriter  # noqa: TID253

# Add the deepmd-kit root to Python path
deepmd_root = Path(__file__).parent.parent
sys.path.insert(0, str(deepmd_root))


def load_single_configuration(data_dir: str, frame_idx: int = 0) -> dict[str, Any]:
    """Load a single configuration from the dataset.

    Parameters
    ----------
    data_dir : str
        Path to the data directory containing set.000/
    frame_idx : int, optional
        Index of the frame to load (default: 0)

    Returns
    -------
    dict
        Dictionary containing coord, box, atom_types, and optional energy/force
    """
    set_dir = Path(data_dir) / "set.000"

    # Load data
    coord = np.load(set_dir / "coord.npy")[frame_idx : frame_idx + 1]  # Keep batch dim
    box = np.load(set_dir / "box.npy")[frame_idx : frame_idx + 1]  # Keep batch dim

    # Load atom types
    type_map_file = Path(data_dir) / "type_map.raw"
    type_file = Path(data_dir) / "type.raw"

    if type_map_file.exists():
        with open(type_map_file) as f:
            type_map = [line.strip() for line in f]
    else:
        type_map = None

    if type_file.exists():
        with open(type_file) as f:
            atom_types = [int(line.strip()) for line in f]
    else:
        raise FileNotFoundError(f"Atom type file not found: {type_file}")

    # Optionally load reference data
    data = {
        "coord": coord,
        "box": box,
        "atom_types": np.array(atom_types),
        "type_map": type_map,
    }

    # Load energy and force if available (for comparison)
    energy_file = set_dir / "energy.npy"
    force_file = set_dir / "force.npy"

    if energy_file.exists():
        data["energy"] = np.load(energy_file)[frame_idx : frame_idx + 1]
    if force_file.exists():
        data["force"] = np.load(force_file)[frame_idx : frame_idx + 1]

    return data


def inference_single_config(
    model_file: str,
    enable_profiling: bool = False,
) -> float:
    """Perform inference on a single configuration with comprehensive TensorBoard logging.

    Parameters
    ----------
    model_file : str
        Path to the model checkpoint file.
    enable_profiling : bool, optional
        Whether to enable PyTorch profiling, by default False

    Returns
    -------
    float
        Elapsed time for the inference in seconds.
    """
    # Import DeepPot for simplified inference
    from deepmd.infer import (
        DeepPot,
    )

    # Setup logging with timestamp
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # Setting working directory
    work_dir = deepmd_root / "examples" / "water" / "se_e3_tebd"
    original_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        log.debug(f"Changed to working directory: {work_dir}")

        log_dir = "./profile_logs"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

        # Test parameters
        data_dir = "../data/data_3"  # Directory contains test data
        frame_idx = 0  # Use first frame

        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file '{model_file}' not found in {work_dir}"
            )

        log.debug(f"Loading model: {model_file}")

        # Initialize model using DeepPot interface (outside profiling for cleaner results)
        dp = DeepPot(model_file, auto_batch_size=1024)

        log.debug(f"Loading single configuration from: {data_dir}")

        # Load single configuration (outside profiling)
        data = load_single_configuration(data_dir, frame_idx)
        coord = data["coord"]
        box = data["box"]
        atom_types = data["atom_types"]

        log.debug("Configuration info:")
        log.debug(f"  Number of atoms: {len(atom_types)}")
        log.debug(f"  Coordinate shape: {coord.shape}")
        log.debug(f"  Box shape: {box.shape}")
        log.debug(f"  Atom types shape: {atom_types.shape}")
        log.debug(f"  Unique atom types: {np.unique(atom_types)}")

        if data.get("type_map"):
            log.debug(f"  Type map: {data['type_map']}")

        log.debug("Starting single configuration inference...")

        # Use profiler if enabled
        if enable_profiling:
            log.info("PyTorch profiling enabled...")

            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=3, warmup=3, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    writer.get_logdir()
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            ) as prof:
                # Warmup and active phases for profiling
                for phase in range(9):  # 3 wait + 3 warmup + 3 active
                    if phase == 6:  # Start active profiling
                        log.debug("Starting profiling phase...")

                    # Record time usage
                    start_time = time.monotonic()

                    # 3: Use record_function to label the core inference step
                    with record_function("Inference (DeepPot.eval)"):
                        # Perform inference using DeepPot.eval()
                        e, f, v = dp.eval(coord, box, atom_types)

                    elapsed_time = time.monotonic() - start_time

                    if phase == 6:  # End active profiling
                        log.debug("Ending profiling phase...")

                    # Mark profiler step
                    prof.step()

                # Save profiling summaries to a log file instead of showing on screen
                profiling_output_path = "profile_summary.log"
                with open(profiling_output_path, "w") as pf:
                    pf.write("=== PyTorch Profiling Summary ===\n")
                    pf.write("Top 10 CPU operations by total time:\n")
                    cpu_summary = prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=10
                    )
                    pf.write(f"{cpu_summary}\n\n")

                    pf.write("Top 10 CUDA operations by total time:\n")
                    cuda_summary = prof.key_averages().table(
                        sort_by="cuda_time_total", row_limit=10
                    )
                    pf.write(f"{cuda_summary}\n\n")

                    pf.write("Top 10 memory allocations:\n")
                    memory_summary = prof.key_averages().table(
                        sort_by="cpu_memory_usage", row_limit=10
                    )
                    pf.write(f"{memory_summary}\n")

                log.info("Profile logs saved to ./profile_logs/")
                log.info(
                    "To view detailed results, run: tensorboard --logdir=./profile_logs"
                )
            writer.close()
        else:
            # Regular inference without profiling
            # Record time usage
            start_time = time.monotonic()

            # Perform inference using DeepPot.eval()
            e, f, v = dp.eval(coord, box, atom_types)

            elapsed_time = time.monotonic() - start_time

        # Print results (keep these as info level - these are the main results)
        log.info("\n=== Inference Results ===")
        predicted_energy = e.reshape(-1)[0]
        log.info(f"Predicted energy: {predicted_energy:.6f}")

        if "energy" in data:
            reference_energy = data["energy"][0]
            energy_diff = abs(predicted_energy - reference_energy)
            log.info(f"Reference energy: {reference_energy:.6f}")
            log.info(f"Energy difference: {energy_diff:.6f}")

        predicted_force = f
        log.info(f"Force norm: {np.linalg.norm(predicted_force):.6f}")

        if "force" in data:
            reference_force = data["force"].reshape(predicted_force.shape)
            force_diff = np.linalg.norm(predicted_force - reference_force)
            log.info(f"Reference force norm: {np.linalg.norm(reference_force):.6f}")
            log.info(f"Force RMSE: {force_diff / np.sqrt(predicted_force.size):.6f}")

        predicted_virial = v.reshape(-1)
        log.info(f"Predicted virial: {predicted_virial}")

        log.info("Inference completed successfully!")
        log.info(f"Elapsed time: {elapsed_time:.6f} seconds")

        return elapsed_time

    except Exception as e:
        log.error(f"Error during inference: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Set this to True to enable PyTorch profiling
    ENABLE_PROFILING = True

    # Run inference and calculate average timing
    # If profiling is enabled, force single run
    num_runs = 1 if ENABLE_PROFILING else 10
    times = []

    model_name = "no"

    print(f"Running inference {num_runs} times...")  # noqa: T201
    if ENABLE_PROFILING:
        print("PyTorch profiling ENABLED (single run forced)")  # noqa: T201
    print("=" * 50)  # noqa: T201

    for i in range(num_runs):
        print(f"\nRun {i + 1}/{num_runs}")  # noqa: T201
        print("-" * 20)  # noqa: T201

        # Enable profiling if requested (will only run once anyway)
        elapsed_time = inference_single_config(
            model_file=f"{model_name}.pth", enable_profiling=ENABLE_PROFILING
        )
        times.append(elapsed_time)

    # Calculate and display statistics
    print("\n" + "=" * 50)  # noqa: T201
    print("Timing Summary:")  # noqa: T201
    print("=" * 50)  # noqa: T201

    # Drop the first run to avoid cold start bias
    if len(times) > 1:
        times = times[1:]

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"Average time: {avg_time:.6f} seconds")  # noqa: T201
    print(f"Min time: {min_time:.6f} seconds")  # noqa: T201
    print(f"Max time: {max_time:.6f} seconds")  # noqa: T201
    print(f"Std deviation: {np.std(times):.6f} seconds")  # noqa: T201
    print(f"All times: {[f'{t:.6f}' for t in times]}")  # noqa: T201
