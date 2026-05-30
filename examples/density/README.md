# Charge Density Prediction Example

This example demonstrates how to train and evaluate a **charge density** model using DeePMD-kit with the PyTorch backend.

The model predicts the charge density on a set of grid points (`grid`) for a given atomic configuration (`coord`, `atype`, `box`).

______________________________________________________________________

## Directory Structure

```
.
├── dpa2/                     # DPA-2 descriptor example
│   └── input.json
├── dpa3/                     # DPA-3 descriptor example
│   ├── input.json
├── dataset/
│   └── qm9/
│       ├── C7H15NO_train/    # Training data (deepmd/npy format)
│       └── C7H15NO_val/      # Validation data (deepmd/npy format)
└── dptest_density_script.py  # Evaluation script for density models
```

______________________________________________________________________

## Data Format

The training/validation data follows the standard **`deepmd/npy`** format, with two additional files in each `set.000/` directory:

| File              | Shape                   | Description                        |
| ----------------- | ----------------------- | ---------------------------------- |
| `coord.npy`       | `[nframes, natoms * 3]` | Atomic coordinates                 |
| `box.npy`         | `[nframes, 9]`          | Simulation cell vectors            |
| `type.raw`        | `[natoms]`              | Atom type indices                  |
| `type_map.raw`    | `[ntypes]`              | Type map (e.g., `C H N O ...`)     |
| **`grid.npy`**    | `[nframes, ngrid, 3]`   | **Grid point coordinates**         |
| **`density.npy`** | `[nframes, ngrid, 1]`   | **Charge density labels on grids** |

> **Note:** `grid.npy` and `density.npy` are required for the density model. The number of grid points (`ngrid`) must match between `grid.npy` and `density.npy`.

______________________________________________________________________

## Training

### 1. Choose a Configuration

Two example configurations are provided:

- **`dpa2/input.json`** — Uses the DPA-2 descriptor.
- **`dpa3/input.json`** — Uses the DPA-3 descriptor (recommended).

Key parameters in `input.json`:

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

### 2. Run Training

```bash
cd dpa3   # or cd dpa2
dp --pt train input.json
```

The training will output:

- `model.ckpt-*.pt` — Model checkpoints
- `lcurve.out` — Training/validation loss curves
- `out.json` — Final training parameters

### 3. Finetune from a Pretrained Model

To finetune an existing density checkpoint:

```bash
cd dpa3
dp --pt train input.json --finetune model.ckpt-*.pt
```

> **Note:** For density models, `change_out_bias` (the energy-bias adjustment used in standard finetuning) is **automatically skipped** because density outputs are grid-based, not atomic-based. The descriptor weights are inherited, and the fitting net adapts via normal gradient descent.

### 4. Freeze the Model

To export a trained checkpoint into a frozen model for inference:

```bash
cd dpa3
dp --pt freeze -c . -o frozen_model
```

This generates `frozen_model.pth` (PyTorch backend).

______________________________________________________________________

## Testing / Evaluation

Use the provided `dptest_density_script.py` to evaluate a trained model on validation or test data.

### Basic Usage

```bash
cd /aisi/yuzhiLiu/deepmd-kit-charge/deepmd-kit-dpa3/examples/density

python dptest_density_script.py \
    dpa3/model.ckpt-*pt \
    dataset/qm9/C7H15NO_val \
    --ratio 0.1 \
    --output val_result.txt
```

Arguments:

| Argument      | Description                                                               |
| ------------- | ------------------------------------------------------------------------- |
| `model`       | Path to the model file (`.pt` checkpoint or `.pth` frozen model)          |
| `data_dir`    | Root directory of deepmd/npy datasets                                     |
| `--ratio`     | Fraction of frames to randomly sample (default: `0.1`)                    |
| `--output`    | If provided, save screen output to this file                              |
| `--pred-file` | File to save paired `[prediction, label]` array (default: `result.d.out`) |

### Evaluate the Full QM9 Dataset

```bash
python dptest_density_script.py \
    dpa3/model.ckpt-100.pt \
    dataset/qm9 \
    --ratio 0.1
```

The script recursively searches all subdirectories containing `type.raw`.

## Notes

- **Backend:** This example uses the PyTorch backend (`--pt`). Make sure you have installed DeePMD-kit with PyTorch support.
- **Stat File:** The `input.json` specifies `"stat_file": "./qm9_charge_density.hdf5"` for caching descriptor statistics. It will be generated automatically on the first run.
- **Checkpoint vs. Frozen Model:** `dptest_density_script.py` uses `DeepPot()` to load the model. If loading a training checkpoint (`.pt`) fails, freeze it first with `dp --pt freeze` and use the resulting `.pth` file.
