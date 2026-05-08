---
name: deepmd-train-se-e2-a
description: Train a DeePMD-kit model using the SE_E2_A (DeepPot-SE) descriptor with the PyTorch backend. Use when the user wants to train a classical deep potential model for a specific system, prepare training input JSON, run `dp --pt train`, monitor learning curves, freeze the model, and test it. SE_E2_A is the foundational two-body embedding descriptor suitable for most condensed-phase systems.
compatibility: Requires deepmd-kit with PyTorch backend installed. GPU recommended for production training.
license: LGPL-3.0-or-later
metadata:
  author: iProzd
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# DeePMD-kit Training: SE_E2_A

Train a deep potential model using the SE_E2_A (Smooth Edition, two-body embedding, all information) descriptor. This is the foundational DeepPot-SE architecture suitable for most condensed-phase systems.

## Quick Start

```bash
dp --pt train input.json
```

## Agent Responsibilities

1. Confirm the user has a working deepmd-kit environment with PyTorch backend.
1. Collect the minimum required information:
   - Training data paths (deepmd/npy or deepmd/hdf5 format)
   - Validation data paths
   - Element types (type_map)
   - Target number of training steps
1. Generate a complete `input.json` training configuration.
1. Explain key hyperparameters if the user is unfamiliar.
1. Run training and monitor the learning curve (`lcurve.out`).
1. Freeze the trained model to `.pth` format.
1. Optionally test the model with `dp test`.

## Workflow

### Step 1: Prepare Training Data

Training data must be in DeePMD format (deepmd/npy or deepmd/hdf5). Each system directory should contain:

```
system_dir/
├── type.raw          # atom type indices, one integer per atom
├── type_map.raw      # element names, one per line
└── set.000/
    ├── coord.npy     # coordinates (nframes, natoms*3)
    ├── energy.npy    # energies (nframes, 1)
    ├── force.npy     # forces (nframes, natoms*3)
    └── box.npy       # cell vectors (nframes, 9)
```

If the user has DFT output (VASP OUTCAR, etc.), refer to the `dpdata-cli` skill for format conversion.

### Step 2: Write input.json

A complete SE_E2_A training configuration:

```json
{
  "model": {
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "type": "se_e2_a",
      "sel": [
        46,
        92
      ],
      "rcut_smth": 0.5,
      "rcut": 6.0,
      "neuron": [
        25,
        50,
        100
      ],
      "resnet_dt": false,
      "axis_neuron": 16,
      "type_one_side": true,
      "seed": 1
    },
    "fitting_net": {
      "neuron": [
        240,
        240,
        240
      ],
      "resnet_dt": true,
      "seed": 1
    }
  },
  "learning_rate": {
    "type": "exp",
    "decay_steps": 5000,
    "start_lr": 0.001,
    "stop_lr": 3.51e-08
  },
  "loss": {
    "type": "ener",
    "start_pref_e": 0.02,
    "limit_pref_e": 1,
    "start_pref_f": 1000,
    "limit_pref_f": 1,
    "start_pref_v": 0.02,
    "limit_pref_v": 1
  },
  "training": {
    "training_data": {
      "systems": [
        "./data/train_system_0",
        "./data/train_system_1"
      ],
      "batch_size": "auto"
    },
    "validation_data": {
      "systems": [
        "./data/valid_system_0"
      ],
      "batch_size": 1,
      "numb_btch": 3
    },
    "numb_steps": 400000,
    "seed": 10,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "save_freq": 10000
  }
}
```

If you do not want to train on virial, set the virial prefactors to 0.

SE_E2_A uses different default loss prefactors compared to DPA3 (e: 0.02→1, f: 1000→1 vs. e: 0.2→20, f: 100→60, v: 0.02→1).

The meaning of each parameter can be generated through `dp doc-train-input`.
Considering the output RST documentation on the screen is very long, use `grep` to find the documentation of a specific parameter:

```sh
dp doc-train-input | grep -A 7 training/numb_steps
dp doc-train-input | grep -A 7 'model\[standard\]/descriptor\[se_e2_a\]/sel'
```

### Step 3: Run Training

```bash
dp --pt train input.json
```

To restart from a checkpoint:

```bash
dp --pt train input.json --restart model.ckpt.pt
```

To initialize from an existing model:

```bash
dp --pt train input.json --init-model model.ckpt.pt
```

### Step 4: Monitor Training

The learning curve is written to `lcurve.out` with columns:

```
#  step  rmse_val  rmse_trn  rmse_e_val  rmse_e_trn  rmse_f_val  rmse_f_trn  rmse_v_val  rmse_v_trn  lr
```

- `rmse_e_*`: energy RMSE per atom (eV/atom)
- `rmse_f_*`: force RMSE (eV/A)
- `rmse_v_*`: virial RMSE (eV/atom, only present if virial data is available)
- `lr`: current learning rate

Quick visualization:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("lcurve.out", names=True)
for name in data.dtype.names[1:-1]:
    plt.plot(data["step"], data[name], label=name)
plt.legend()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.xscale("symlog")
plt.yscale("log")
plt.grid()
plt.show()
```

### Step 5: Freeze the Model

```bash
dp --pt freeze -o model.pth
```

### Step 6: Test the Model

```bash
dp --pt test -m model.pth -s /path/to/test_system -n 30
```

## Key Hyperparameters

### Descriptor

| Parameter       | Description                         | Typical Value    |
| --------------- | ----------------------------------- | ---------------- |
| `rcut`          | Cutoff radius (A)                   | 6.0              |
| `rcut_smth`     | Smooth cutoff start (A)             | 0.5              |
| `sel`           | Max neighbors per type              | System-dependent |
| `neuron`        | Embedding net sizes                 | [25, 50, 100]    |
| `axis_neuron`   | Axis matrix dimension               | 16               |
| `type_one_side` | Share embedding across center types | true             |

### Fitting Net

| Parameter   | Description            | Typical Value   |
| ----------- | ---------------------- | --------------- |
| `neuron`    | Hidden layer sizes     | [240, 240, 240] |
| `resnet_dt` | Use timestep in ResNet | true            |

### Loss Prefactors

| JSON keys                       | Description              | Start | Limit |
| ------------------------------- | ------------------------ | ----- | ----- |
| `start_pref_e` / `limit_pref_e` | Energy weight            | 0.02  | 1     |
| `start_pref_f` / `limit_pref_f` | Force weight             | 1000  | 1     |
| `start_pref_v` / `limit_pref_v` | Virial weight (optional) | 0.02  | 1     |

Here, `start_pref_*` and `limit_pref_*` set the initial and final loss weights; the loss shifts from force-dominated early training to balanced energy+force later. For virial, set to 0 if not training on virial data.

### Training

| Parameter     | Description           | Typical Value       |
| ------------- | --------------------- | ------------------- |
| `numb_steps`  | Total training steps  | 400000-1000000      |
| `batch_size`  | Frames per step       | "auto" or "auto:32" |
| `start_lr`    | Initial learning rate | 0.001               |
| `stop_lr`     | Final learning rate   | 3.51e-8             |
| `decay_steps` | LR decay interval     | 5000                |

### Setting `sel`

`sel` is a list with one entry per element type, specifying the maximum number of neighbors of that type within `rcut`. To determine appropriate values:

```bash
dp --pt neighbor-stat -s /path/to/data -r 6.0 -t O H
```

Use values slightly above the reported maximum.

## Agent Checklist

- [ ] Training data exists and is in deepmd format
- [ ] `type_map` matches the elements in the data
- [ ] `sel` is appropriate for the system (use `dp neighbor-stat` if unsure)
- [ ] `rcut` is reasonable for the system (typically 6.0-9.0 A)
- [ ] Training completes without NaN in `lcurve.out`
- [ ] Model is frozen to `.pth` after training
- [ ] Test RMSE values are reported to the user

## References

- [SE_E2_A descriptor documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/train-se-e2-a.html)
- [Training documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training.html)
- [Training advanced options](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training-advanced.html)
- [DeePMD-kit GitHub](https://github.com/deepmodeling/deepmd-kit)
