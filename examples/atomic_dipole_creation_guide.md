# Creating atomic_dipole.npy Files for DeePMD-kit

This guide explains how to create the `atomic_dipole.npy` files required for training Deep Potential Long Range (DPLR) models and other tensor fitting models in DeePMD-kit.

## What is atomic_dipole?

The `atomic_dipole` data represents atomic-level dipole vectors. For DPLR models, these vectors point from each atom to the center of its electronic density, typically represented by Wannier centroids from first-principles calculations.

## Quick Start

Run the example script to see different methods for creating atomic dipole data:

```bash
python examples/create_atomic_dipole_example.py
```

## Data Format Requirements

- **File name**: Must be named `atomic_dipole.npy` (or `atomic_dipole.raw` for text format)
- **Shape**: `(n_frames, n_atoms * 3)` where `n_atoms * 3` represents x,y,z components for each atom
- **Data type**: `float64`
- **Units**: Any consistent unit (commonly Ångström × elementary charge)

## Methods for Creating Atomic Dipole Data

### 1. From DFT + Wannier90 Calculations (Recommended for DPLR)

For DPLR models, obtain Wannier centroids from DFT calculations:

1. Run DFT calculations with VASP, QUANTUM ESPRESSO, or similar
2. Use Wannier90 to compute maximally localized Wannier functions
3. Extract Wannier centroid positions
4. Calculate dipole vectors: `atomic_dipole = wannier_centroid - atom_position`

**Hands-on tutorial**: [CSI Princeton Workshop](https://github.com/CSIprinceton/workshop-june-2024/tree/main/hands-on-sessions/day-2-DW-DPLR/7-deep-wannier)

### 2. Using dpdata (Recommended)

```python
import numpy as np
import dpdata
from dpdata.data_type import Axis, DataType

# Register atomic_dipole data type
datatype = DataType(
    "atomic_dipole",
    np.ndarray,
    shape=(Axis.NFRAMES, Axis.NATOMS, 3),
    required=False,
)
dpdata.System.register_data_type(datatype)
dpdata.LabeledSystem.register_data_type(datatype)

# Load system and add atomic dipole data
system = dpdata.LabeledSystem("your_data_directory")
system.data["atomic_dipole"] = your_atomic_dipole_array  # shape: (n_frames, n_atoms, 3)
system.to_deepmd_npy("output_directory")
```

### 3. From Raw Text Files

Create `atomic_dipole.raw` with format:

```
# Each row = one frame, columns = atom1_x atom1_y atom1_z atom2_x atom2_y atom2_z ...
0.123 -0.456 0.789 0.234 -0.567 0.891
0.135 -0.468 0.801 0.246 -0.579 0.903
```

Convert to numpy format:

```bash
python -c "
import numpy as np
data = np.loadtxt('atomic_dipole.raw', ndmin=2)
data = data.astype(np.float64)
np.save('atomic_dipole', data)
"
```

### 4. Direct NumPy Creation

```python
import numpy as np

# Example: 10 frames, 32 atoms per frame
n_frames = 10
n_atoms = 32
atomic_dipoles = np.zeros((n_frames, n_atoms, 3))

# Fill with your calculated dipole data
# atomic_dipoles[frame_idx, atom_idx, :] = [dx, dy, dz]

# Reshape for DeePMD format and save
atomic_dipoles_flat = atomic_dipoles.reshape(n_frames, n_atoms * 3)
np.save("atomic_dipole.npy", atomic_dipoles_flat)
```

## File Organization

Place the `atomic_dipole.npy` file in your dataset directory structure:

```
your_dataset/
├── type.raw
├── type_map.raw
└── set.000/
    ├── coord.npy
    ├── box.npy
    ├── atomic_dipole.npy  # <-- Your atomic dipole data
    └── energy.npy (optional)
```

## Troubleshooting

### Shape Mismatch Errors

```
ValueError: cannot reshape array of size xxx into shape (xx,xx)
```

- Check that your data shape is `(n_frames, n_atoms * 3)`
- Verify the file is named correctly (`atomic_dipole.npy`, not `dipole.npy`)

### Zero Dipoles for Some Atoms

For atoms that don't contribute to the property (e.g., hydrogen in some models):

- Set their dipole components to zero
- Or exclude them using `atom_exclude_types` in the model configuration

## Model Configuration

When training with atomic dipole data, configure your model:

```json
{
  "model": {
    "fitting_net": {
      "type": "dipole",
      "sel_type": [0] // or "atom_exclude_types": [1] for PyTorch
    }
  },
  "loss": {
    "type": "tensor",
    "pref": 0.0,
    "pref_atomic": 1.0
  }
}
```

## Related Documentation

- [Tensor Fitting Documentation](../doc/model/train-fitting-tensor.md)
- [DPLR Model Documentation](../doc/model/dplr.md)
- [System Data Format](../doc/data/system.md)

## Example Files

See `examples/create_atomic_dipole_example.py` for complete working examples of all methods described above.
