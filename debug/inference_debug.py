#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Debug script for single configuration model inference.

This script loads only one configuration from the dataset and performs inference.
Perfect for profiling and debugging individual forward passes.
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


def inference_single_config() -> None:
    """Perform inference on a single configuration."""
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

    # Set working directory to examples/water/se_e3_tebd
    work_dir = deepmd_root / "examples" / "water" / "se_e3_tebd"
    original_cwd = os.getcwd()

    try:
        os.chdir(work_dir)
        log.info(f"Changed to working directory: {work_dir}")

        # Test parameters
        model_file = "no.pth"  # Model file to test
        data_dir = "../data/data_3"  # Directory contains test data
        frame_idx = 0  # Use first frame

        # Check if model file exists
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"Model file '{model_file}' not found in {work_dir}"
            )

        log.info(f"Loading model: {model_file}")

        # Initialize model using DeepPot interface
        dp = DeepPot(model_file, auto_batch_size=True)

        log.info(f"Loading single configuration from: {data_dir}")

        # Load single configuration
        data = load_single_configuration(data_dir, frame_idx)

        coord = data["coord"]
        box = data["box"]
        atom_types = data["atom_types"]

        log.info("Configuration info:")
        log.info(f"  Number of atoms: {len(atom_types)}")
        log.info(f"  Coordinate shape: {coord.shape}")
        log.info(f"  Box shape: {box.shape}")
        log.info(f"  Atom types shape: {atom_types.shape}")
        log.info(f"  Unique atom types: {np.unique(atom_types)}")

        if data.get("type_map"):
            log.info(f"  Type map: {data['type_map']}")

        log.info("Starting single configuration inference...")

        # Record time usage
        start_time = time.time()

        # Perform inference using DeepPot.eval()
        e, f, v = dp.eval(coord, box, atom_types)

        elapsed_time = time.time() - start_time

        # Print results
        log.info("\n=== Inference Results ===")
        predicted_energy = e.reshape(-1)[0]
        log.info(f"Predicted energy: {predicted_energy:.6f}")

        if "energy" in data:
            reference_energy = data["energy"][0]
            energy_diff = abs(predicted_energy - reference_energy)
            log.info(f"Reference energy: {reference_energy:.6f}")
            log.info(f"Energy difference: {energy_diff:.6f}")

        predicted_force = f
        log.info(f"Predicted force shape: {predicted_force.shape}")
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

    except Exception as e:
        log.error(f"Error during inference: {e}")
        raise
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    inference_single_config()
