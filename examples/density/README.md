# Charge Density Prediction Example

This example demonstrates how to train and evaluate a **charge density** model using DeePMD-kit with the PyTorch backend.

The model predicts the charge density on a set of grid points (`grid`) for a given atomic configuration (`coord`, `atype`, `box`).

______________________________________________________________________

## Directory Structure

```
.
├── dpa2/                     # DPA-2 descriptor example
│   └── input.json
├── dpa3/                     # DPA-3 descriptor example (recommended)
│   └── input.json
├── dataset/
│   └── qm9/
│       ├── C7H15NO_train/    # Training data (deepmd/npy format)
│       └── C7H15NO_val/      # Validation data (deepmd/npy format)
└── dptest_density_script.py  # Standalone evaluation script (alternative to dp test)
```

______________________________________________________________________

## Data Format

The training/validation data follows the standard **`deepmd/npy`** format, with two additional files in each `set.000/` directory:

| File              | Shape                   | Description                        |
| ----------------- | ----------------------- | ---------------------------------- |
| `coord.npy`       | `[nframes, natoms * 3]` | Atomic coordinates                 |
| `box.npy`         | `[nframes, 9]`          | Simulation cell vectors            |
| `type.raw`        | `[natoms]`              | Atom type indices                  |
| `type_map.raw`    | `[ntypes]`              | Type map (e.g., `C H N O ... X`)   |
| **`grid.npy`**    | `[nframes, ngrid, 3]`   | **Grid point coordinates**         |
| **`density.npy`** | `[nframes, ngrid, 1]`   | **Charge density labels on grids** |

> **Notes:**
>
> - `grid.npy` and `density.npy` are required for the density model. The number of grid points (`ngrid`) must match between `grid.npy` and `density.npy`, and is allowed to differ from `natoms`.
> - The **last entry of `type_map` is reserved as a virtual "grid point type"** (e.g. `X` in the example): internally, grid points are assigned this type when building the grid-to-atom neighbor list. Make sure your `type_map` contains one more entry than the real element types.

______________________________________________________________________

## Training

### 1. Choose a Configuration

Two example configurations are provided:

- **`dpa2/input.json`** — Uses the DPA-2 descriptor.
- **`dpa3/input.json`** — Uses the DPA-3 descriptor (recommended).

Key settings in `input.json`:

```json
{
  "model": {
    "type_map": [
      "Li",
      "Ni",
      "Co",
      "Mn",
      "O",
      "C",
      "H",
      "N",
      "F",
      "X"
    ],
    "descriptor": {
      "type": "dpa3"
    },
    "fitting_net": {
      "type": "density",
      "neuron": [
        240,
        240,
        240
      ]
    }
  },
  "loss": {
    "type": "grid_density",
    "start_pref_d": 1,
    "limit_pref_d": 1
  },
  "training": {
    "training_data": {
      "systems": [
        "../dataset/qm9/C7H15NO_train"
      ],
      "batch_size": "auto:128"
    },
    "validation_data": {
      "systems": [
        "../dataset/qm9/C7H15NO_val"
      ],
      "batch_size": 1,
      "numb_btch": 3
    }
  }
}
```

- The fitting net type is `"density"` and the loss type is `"grid_density"`.
- The model type `"grid_density"` is selected automatically from the fitting net type.

### 2. Run Training

```bash
cd dpa3   # or cd dpa2
dp --pt train input.json
```

The training will output:

- `model.ckpt-*.pt` — Model checkpoints
- `lcurve.out` — Training/validation loss curves (`mae_d` / `rmse_d` for density)
- `out.json` — Final training parameters

### 3. Finetune from a Pretrained Model

```bash
cd dpa3
dp --pt train input.json --finetune model.ckpt-*.pt
```

> **Note:** For density models, `change_out_bias` (the energy-bias adjustment used in standard finetuning) is **automatically skipped** because density outputs are grid-based, not atomic-based. The descriptor weights are inherited, and the fitting net adapts via normal gradient descent.

### 4. Freeze the Model

Export a trained checkpoint into a frozen model for inference:

```bash
cd dpa3
dp --pt freeze
```

This reads `model.ckpt.pt` in the current directory and generates `frozen_model.pth`.

______________________________________________________________________

## Testing / Evaluation

### Option 1: `dp test` (recommended)

`dp test` supports density models natively:

```bash
cd dpa3
dp --pt test -m frozen_model.pth -s ../dataset/qm9/C7H15NO_val -n 0
```

- `-n 0` tests **all** frames; use a positive number to test only the first N frames.
- Add `--detail-file detail` to write per-frame `[label, prediction]` pairs to `detail.density.out.<frame>` (one file per frame, one line per grid point).
- Add `--shuffle-test --rand-seed 42` to test on a shuffled subset.

Example output:

```
# testing system : ../dataset/qm9/C7H15NO_val
# number of test data : 30
DENSITY MAE             : x.xxxe+00 units
DENSITY RMSE            : x.xxxe+00 units
```

### Option 2: Standalone script

`dptest_density_script.py` evaluates a model and writes **one combined** `[prediction, label]` file (`result.d.out`), which is convenient for plotting:

```bash
python dptest_density_script.py \
    dpa3/frozen_model.pth \
    dataset/qm9/C7H15NO_val \
    --ratio 1.0 \
    --output val_result.txt
```

| Argument      | Description                                                               |
| ------------- | ------------------------------------------------------------------------- |
| `model`       | Path to the frozen model file (`.pth`)                                    |
| `data_dir`    | Root directory of deepmd/npy datasets (searched recursively)              |
| `--ratio`     | Fraction of frames to randomly sample (default: `0.1`)                    |
| `--output`    | If provided, save screen output to this file as well                      |
| `--pred-file` | File to save paired `[prediction, label]` array (default: `result.d.out`) |

> **Note:** The script registers the `grid`/`density` dpdata types with a hard-coded grid count (`125`, matching this example). If your dataset uses a different `ngrid`, edit the `shape=(Axis.NFRAMES, 125, ...)` lines at the top of the script.

### Option 3: Python interface

```python
import numpy as np
from deepmd.infer import DeepEval

dp = DeepEval("frozen_model.pth")  # dispatched to DeepDensity automatically
density = dp.eval(coord, box, atype, grid=grid)  # (nframes, ngrid)
```

______________________________________________________________________

## Notes

- **Backend:** PyTorch only (`--pt`). Make sure DeePMD-kit is installed with PyTorch support.
- **Stat file:** `input.json` specifies `"stat_file": "./qm9_charge_density.hdf5"` for caching descriptor statistics; it is generated automatically on the first run.
- **Checkpoint vs. frozen model:** evaluation requires a frozen model (`.pth`). Freeze a checkpoint first with `dp --pt freeze`.
