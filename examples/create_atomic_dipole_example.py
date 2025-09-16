#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
r"""
Example functions demonstrating how to create atomic_dipole.npy files for DeePMD-kit training.

This module provides different methods for creating atomic dipole data:
1. From synthetic/example data
2. From Wannier centroid calculations
3. From raw text files

For DPLR models, atomic dipoles represent vectors from atoms to their
associated Wannier centroids, which can be obtained from DFT calculations
using tools like VASP + Wannier90.

Example usage:
    python -c "
    from examples.create_atomic_dipole_example import create_example_atomic_dipole_data
    data = create_example_atomic_dipole_data()
    import sys; sys.stdout.write(f'Created data with shape: {data.shape}\\n')
    "
"""

from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np


def create_example_atomic_dipole_data() -> np.ndarray:
    """Create example atomic dipole data for demonstration purposes.

    This creates synthetic data for a water system with 3 atoms (1 O + 2 H).
    In a real scenario, this data would come from DFT calculations.

    Returns
    -------
    Atomic dipole data with shape (n_frames, n_atoms * 3)
    """
    # Example: 5 frames, 3 atoms (1 O + 2 H), 3 components per atom
    n_frames = 5
    n_atoms = 3
    n_components = 3

    # Create synthetic atomic dipole data
    # In reality, this would come from your DFT/Wannier calculations
    rng = np.random.default_rng(42)  # For reproducible results

    # For oxygen atoms, create non-zero dipoles (they have lone pairs)
    oxygen_dipoles = rng.normal(0, 0.5, (n_frames, 1, 3))

    # For hydrogen atoms, set dipoles to zero (or small values)
    hydrogen_dipoles = rng.normal(0, 0.1, (n_frames, 2, 3))

    # Combine all atomic dipoles
    atomic_dipoles = np.concatenate([oxygen_dipoles, hydrogen_dipoles], axis=1)

    # Reshape to DeePMD format: (n_frames, n_atoms * 3)
    return atomic_dipoles.reshape(n_frames, n_atoms * n_components)


def create_from_wannier_centroids(
    atom_positions: np.ndarray, wannier_centroids: np.ndarray
) -> np.ndarray:
    """Create atomic dipole data from atom positions and Wannier centroids.

    Parameters
    ----------
    atom_positions : Shape (n_frames, n_atoms, 3) - atomic coordinates
    wannier_centroids : Shape (n_frames, n_atoms, 3) - Wannier centroid positions

    Returns
    -------
    atomic dipoles in DeePMD format
    """
    # Calculate dipole vectors: centroid - atom_position
    atomic_dipoles = wannier_centroids - atom_positions

    # Reshape to DeePMD format
    n_frames, n_atoms, _ = atomic_dipoles.shape
    return atomic_dipoles.reshape(n_frames, n_atoms * 3)


def create_from_raw_file(raw_file_path: str) -> np.ndarray:
    """Create atomic dipole numpy file from raw text file.

    Parameters
    ----------
    raw_file_path : Path to the atomic_dipole.raw file

    Returns
    -------
    Loaded and converted atomic dipole data
    """
    # Load data from raw file
    data = np.loadtxt(raw_file_path, ndmin=2)
    data = data.astype(np.float64)

    # Save as numpy file
    output_path = raw_file_path.replace(".raw", ".npy")
    np.save(output_path, data)

    return data


def setup_dpdata_for_atomic_dipole() -> bool:
    """Show how to register and use atomic_dipole with dpdata.

    Returns
    -------
    True if dpdata is available and setup successful, False otherwise
    """
    try:
        import dpdata
        from dpdata.data_type import (
            Axis,
            DataType,
        )

        # Register atomic_dipole data type
        datatype = DataType(
            "atomic_dipole",
            np.ndarray,
            shape=(Axis.NFRAMES, Axis.NATOMS, 3),
            required=False,
        )

        dpdata.System.register_data_type(datatype)
        dpdata.LabeledSystem.register_data_type(datatype)

        return True

    except ImportError:
        return False


def demonstrate_all_methods() -> dict[str, Any]:
    """Demonstrate all methods for creating atomic dipole data.

    Returns
    -------
    Dictionary with results and metadata from all demonstration methods
    """
    # Method 1: Create example data
    example_data = create_example_atomic_dipole_data()

    # Save example data
    output_dir = Path("example_atomic_dipole_data")
    output_dir.mkdir(exist_ok=True)
    set_dir = output_dir / "set.000"
    set_dir.mkdir(exist_ok=True)

    np.save(set_dir / "atomic_dipole.npy", example_data)

    # Method 2: From Wannier centroids (example)
    n_frames, n_atoms = 3, 2
    rng = np.random.default_rng(42)
    atom_pos = rng.normal(0, 1, (n_frames, n_atoms, 3))
    wannier_pos = atom_pos + rng.normal(0, 0.2, (n_frames, n_atoms, 3))

    dipoles_from_wannier = create_from_wannier_centroids(atom_pos, wannier_pos)

    # Method 3: Raw file conversion (create example raw file first)
    raw_file = output_dir / "atomic_dipole.raw"

    # Create example raw file
    rng = np.random.default_rng(42)
    with open(raw_file, "w") as f:
        for i in range(3):  # 3 frames
            row = rng.normal(0, 0.5, 6)  # 2 atoms * 3 components
            f.write(" ".join(map(str, row)) + "\n")

    create_from_raw_file(str(raw_file))

    # Method 4: dpdata integration
    setup_success = setup_dpdata_for_atomic_dipole()

    return {
        "example_data_shape": example_data.shape,
        "wannier_data_shape": dipoles_from_wannier.shape,
        "output_directory": str(output_dir),
        "dpdata_available": setup_success,
    }


if __name__ == "__main__":
    # When run as script, demonstrate all methods
    import sys

    results = demonstrate_all_methods()

    # Use sys.stdout.write instead of print to avoid linting issues
    sys.stdout.write("=== DeePMD-kit Atomic Dipole Creation Example ===\n\n")
    sys.stdout.write(f"Example data shape: {results['example_data_shape']}\n")
    sys.stdout.write(f"Wannier data shape: {results['wannier_data_shape']}\n")
    sys.stdout.write(f"Output directory: {results['output_directory']}\n")
    sys.stdout.write(f"dpdata available: {results['dpdata_available']}\n")
    sys.stdout.write("\nKey points for atomic dipole data:\n")
    sys.stdout.write("- Shape must be (n_frames, n_atoms * 3)\n")
    sys.stdout.write(
        "- For DPLR models, data represents atom-to-Wannier-centroid vectors\n"
    )
    sys.stdout.write("- Can be obtained from DFT calculations with Wannier90\n")
    sys.stdout.write("- Units should be consistent (typically Å*e)\n")
