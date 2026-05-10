---
name: deepmd-train-dpa3
description: Train a DeePMD-kit model using the DPA3 descriptor with the PyTorch backend. Use when the user wants to train a state-of-the-art deep potential model based on message passing on Line Graph Series (LiGS). DPA3 provides high accuracy and strong generalization, suitable for large atomic models (LAM) and diverse chemical systems. Supports both fixed and dynamic neighbor selection.
compatibility: Requires deepmd-kit with PyTorch backend installed. GPU strongly recommended. Custom OP library required for LAMMPS deployment.
license: LGPL-3.0-or-later
metadata:
  author: iProzd
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# DeePMD-kit Training: DPA3

Train a deep potential model using the DPA3 descriptor, an advanced message-passing architecture operating on Line Graph Series (LiGS). DPA3 is designed as a large atomic model (LAM) with high fitting accuracy and robust generalization across diverse chemical and materials systems.

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
   - Model size preference (L3/L6/L12 layers)
1. Generate a complete `input.json` training configuration.
1. Decide whether to use fixed or dynamic neighbor selection based on system diversity.
1. Run training and monitor the learning curve.
1. Freeze the trained model and optionally test it.

## Workflow

### Step 1: Prepare Training Data

Same format as other DeePMD models. Each system directory should contain:

```
system_dir/
├── type.raw
├── type_map.raw
└── set.000/
    ├── coord.npy
    ├── energy.npy
    ├── force.npy
    ├── box.npy
    └── virial.npy
```

DPA3 also supports the mixed type data format for multi-element systems.

### Step 2: Write input.json

#### Standard DPA3 (fixed selection)

```json
{
  "model": {
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "type": "dpa3",
      "repflow": {
        "n_dim": 128,
        "e_dim": 64,
        "a_dim": 32,
        "nlayers": 6,
        "e_rcut": 6.0,
        "e_rcut_smth": 5.3,
        "e_sel": 120,
        "a_rcut": 4.0,
        "a_rcut_smth": 3.5,
        "a_sel": 30,
        "axis_neuron": 4,
        "fix_stat_std": 0.3,
        "a_compress_rate": 1,
        "a_compress_e_rate": 2,
        "a_compress_use_split": true,
        "update_angle": true,
        "smooth_edge_update": true,
        "edge_init_use_dist": true,
        "use_exp_switch": true,
        "update_style": "res_residual",
        "update_residual": 0.1,
        "update_residual_init": "const"
      },
      "activation_function": "silut:10.0",
      "use_tebd_bias": false,
      "precision": "float32",
      "concat_output_tebd": false,
      "seed": 1
    },
    "fitting_net": {
      "neuron": [
        240,
        240,
        240
      ],
      "resnet_dt": true,
      "precision": "float32",
      "activation_function": "silut:10.0",
      "seed": 1
    }
  },
  "learning_rate": {
    "type": "exp",
    "decay_steps": 5000,
    "start_lr": 0.001,
    "stop_lr": 3e-05
  },
  "loss": {
    "type": "ener",
    "start_pref_e": 0.2,
    "limit_pref_e": 20,
    "start_pref_f": 100,
    "limit_pref_f": 60,
    "start_pref_v": 0.02,
    "limit_pref_v": 1
  },
  "optimizer": {
    "type": "AdamW",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "weight_decay": 0.001
  },
  "training": {
    "stat_file": "./dpa3.hdf5",
    "training_data": {
      "systems": [
        "./data/train_0",
        "./data/train_1",
        "./data/train_2"
      ],
      "batch_size": 1
    },
    "validation_data": {
      "systems": [
        "./data/valid_0"
      ],
      "batch_size": 1
    },
    "numb_steps": 1000000,
    "gradient_max_norm": 5.0,
    "seed": 10,
    "disp_file": "lcurve.out",
    "disp_freq": 100,
    "save_freq": 2000
  }
}
```

If you do not want to train on virial, set the virial prefactors to 0.

DPA3 uses different default loss prefactors compared to SE_E2_A. See the comparison table in the "Key Differences from SE_E2_A" section below.

The meaning of each parameter can be generated through `dp doc-train-input`.
Considering the output RST documentation on the screen is very long, use `grep` to find the documentation of a specific parameter:

```sh
dp doc-train-input | grep -A 7 training/numb_steps
dp doc-train-input | grep -A 7 'model\[standard\]/descriptor\[dpa3\]/repflow/e_sel'
```

#### DPA3 with Dynamic Selection

For systems with highly variable neighbor counts (e.g., multi-element datasets), use dynamic selection by modifying the `repflow` section:

```json
"repflow": {
  "e_sel": 1200,
  "a_sel": 300,
  "use_dynamic_sel": true,
  "sel_reduce_factor": 10.0
}
```

When `use_dynamic_sel` is true, the effective selection is `e_sel / sel_reduce_factor` and `a_sel / sel_reduce_factor` (i.e., 120 and 30 in this example), but the model dynamically adapts to varying neighbor counts.

### Step 3: Run Training

```bash
dp --pt train input.json
```

To restart from a checkpoint:

```bash
dp --pt train input.json --restart model.ckpt.pt
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

### Step 5: Freeze the Model

```bash
dp --pt freeze -o model.pth
```

### Step 6: Test the Model

```bash
dp --pt test -m model.pth -s /path/to/test_system -n 30
```

## Model Size Guide

Choose the number of layers based on accuracy vs. cost trade-off:

| Model         | nlayers | n_dim | e_dim | a_dim | Relative Cost | Use Case                           |
| ------------- | ------- | ----- | ----- | ----- | ------------- | ---------------------------------- |
| DPA3-L3       | 3       | 256   | 128   | 32    | 1x            | Quick prototyping, smaller systems |
| DPA3-L3-small | 3       | 128   | 64    | 32    | 0.8x          | Fast iteration, limited GPU memory |
| DPA3-L6       | 6       | 256   | 128   | 32    | 2x            | Recommended for production         |
| DPA3-L6-small | 6       | 128   | 64    | 32    | 1.4x          | Good accuracy/cost balance         |

Benchmark RMSE (averaged over 6 representative systems, 0.5M steps):

| Model                     | Energy (meV/atom) | Force (meV/A) | Virial (meV/atom) |
| ------------------------- | ----------------- | ------------- | ----------------- |
| DPA3-L3 (256/128/32)      | 5.74              | 85.4          | 43.1              |
| DPA3-L3-small (128/64/32) | 6.99              | 93.6          | 46.7              |
| DPA3-L6 (256/128/32)      | 4.85              | 79.9          | 39.7              |
| DPA3-L6-small (128/64/32) | 5.11              | 77.7          | 41.2              |
| DPA2-L6 (reference)       | 12.12             | 109.3         | 83.1              |

## Key Differences from SE_E2_A

| Aspect            | SE_E2_A              | DPA3                            |
| ----------------- | -------------------- | ------------------------------- |
| Architecture      | Two-body embedding   | Message passing on LiGS         |
| Default precision | float64              | float32                         |
| Optimizer         | Adam                 | AdamW (with weight_decay)       |
| Loss prefactors   | e: 0.02→1, f: 1000→1 | e: 0.2→20, f: 100→60, v: 0.02→1 |
| stop_lr           | 3.51e-8              | 3e-5                            |
| Gradient clipping | Not used             | gradient_max_norm: 5.0          |
| Virial training   | Optional             | Recommended                     |
| Model compression | Supported            | Not supported                   |
| Activation        | tanh (default)       | silut:10.0                      |

## Key Hyperparameters

### Repflow (Descriptor)

| Parameter         | Description                      | Default        |
| ----------------- | -------------------------------- | -------------- |
| `n_dim`           | Node embedding dimension         | 128 or 256     |
| `e_dim`           | Edge embedding dimension         | 64 or 128      |
| `a_dim`           | Angle embedding dimension        | 32             |
| `nlayers`         | Number of message passing layers | 3 or 6         |
| `e_rcut`          | Edge cutoff radius (A)           | 6.0            |
| `e_rcut_smth`     | Edge smooth cutoff start         | 5.3            |
| `e_sel`           | Max edge neighbors               | 120            |
| `a_rcut`          | Angle cutoff radius (A)          | 4.0            |
| `a_rcut_smth`     | Angle smooth cutoff start        | 3.5            |
| `a_sel`           | Max angle neighbors              | 30             |
| `update_style`    | Residual update style            | "res_residual" |
| `update_residual` | Residual scaling factor          | 0.1            |

### Activation Function

DPA3 uses `silut:10.0` by default. For datasets where training is unstable, consider switching to `tanh`:

```json
"descriptor": {
  "type": "dpa3",
  "repflow": { ... },
  "activation_function": "tanh"
},
"fitting_net": {
  "activation_function": "tanh"
}
```

### Optimizer

DPA3 uses AdamW by default (decoupled weight decay):

```json
"optimizer": {
  "type": "AdamW",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "weight_decay": 0.001
}
```

### Gradient Clipping

Recommended for DPA3 to stabilize training:

```json
"training": {
  "gradient_max_norm": 5.0
}
```

## Agent Checklist

- [ ] Training data exists and is in deepmd format
- [ ] `type_map` matches the elements in the data
- [ ] Precision is set to `float32` (DPA3 default, not float64)
- [ ] AdamW optimizer is configured with weight_decay
- [ ] `gradient_max_norm` is set (recommended: 5.0)
- [ ] `stop_lr` is 3e-5 (not 3.51e-8 as in SE_E2_A)
- [ ] Virial loss prefactors are included if virial data is available
- [ ] `stat_file` is set to cache statistics (avoids recomputation on restart)
- [ ] Training completes without NaN in `lcurve.out`
- [ ] Model is frozen to `.pth` after training

## References

- [DPA3 descriptor documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa3.html)
- [DPA3 paper](https://arxiv.org/abs/2506.01686)
- [Training documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training.html)
- [DeePMD-kit GitHub](https://github.com/deepmodeling/deepmd-kit)
