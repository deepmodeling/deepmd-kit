# DPA3 training reference

Read this file only after the user chooses DPA3, or when DPA3 is the best fit for the task. Keep the shared data checks, train/monitor/freeze/test workflow in `../SKILL.md`; this file only records DPA3-specific choices.

## When to choose DPA3

Use DPA3 for high-accuracy training on diverse systems, large atomic model workflows, DPA-3/LiGS requests, dynamic neighbor selection, or pretrained DPA3-style experiments. Assume PyTorch unless the user provides a different supported backend.

## Extra inputs to collect

- Model size preference: L3/L6 and small/full dimensions.
- Whether neighbor counts vary strongly across systems, which decides fixed vs. dynamic selection.
- Whether virial labels are available. Virial training is recommended when present.
- GPU memory budget, because DPA3 is usually heavier than se_e2_a.

## Model-specific JSON sections

Merge these sections into the common training input described by `../SKILL.md`.

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
    "gradient_max_norm": 5.0
  }
}
```

Set the virial prefactors to `0` when virial labels are unavailable or should not be trained.

## Dynamic selection

For systems with highly variable neighbor counts, use dynamic selection in `repflow`:

```json
{
  "e_sel": 1200,
  "a_sel": 300,
  "use_dynamic_sel": true,
  "sel_reduce_factor": 10.0
}
```

With `use_dynamic_sel: true`, the effective selection is `e_sel / sel_reduce_factor` and `a_sel / sel_reduce_factor` (120 and 30 above), while the model adapts to varying neighbor counts.

## Model size guide

| Model         | `nlayers` | `n_dim` | `e_dim` | `a_dim` | Relative cost | Use case                           |
| ------------- | --------- | ------- | ------- | ------- | ------------- | ---------------------------------- |
| DPA3-L3       | 3         | 256     | 128     | 32      | 1x            | Quick prototyping, smaller systems |
| DPA3-L3-small | 3         | 128     | 64      | 32      | 0.8x          | Fast iteration, limited GPU memory |
| DPA3-L6       | 6         | 256     | 128     | 32      | 2x            | Recommended for production         |
| DPA3-L6-small | 6         | 128     | 64      | 32      | 1.4x          | Good accuracy/cost balance         |

Benchmark RMSE averaged over 6 representative systems at 0.5M steps:

| Model                     | Energy (meV/atom) | Force (meV/A) | Virial (meV/atom) |
| ------------------------- | ----------------- | ------------- | ----------------- |
| DPA3-L3 (256/128/32)      | 5.74              | 85.4          | 43.1              |
| DPA3-L3-small (128/64/32) | 6.99              | 93.6          | 46.7              |
| DPA3-L6 (256/128/32)      | 4.85              | 79.9          | 39.7              |
| DPA3-L6-small (128/64/32) | 5.11              | 77.7          | 41.2              |
| DPA2-L6 (reference)       | 12.12             | 109.3         | 83.1              |

## Key differences from se_e2_a

| Aspect            | se_e2_a              | DPA3                            |
| ----------------- | -------------------- | ------------------------------- |
| Architecture      | Two-body embedding   | Message passing on LiGS         |
| Default precision | `float64`            | `float32`                       |
| Optimizer         | Adam                 | AdamW with `weight_decay`       |
| Loss prefactors   | e: 0.02→1, f: 1000→1 | e: 0.2→20, f: 100→60, v: 0.02→1 |
| `stop_lr`         | 3.51e-8              | 3e-5                            |
| Gradient clipping | Usually not used     | `gradient_max_norm: 5.0`        |
| Virial training   | Optional             | Recommended                     |
| Model compression | Supported            | Not supported                   |
| Activation        | `tanh` default       | `silut:10.0`                    |

## DPA3 checklist

- [ ] Precision is `float32` for descriptor and fitting net.
- [ ] AdamW is configured with `weight_decay`.
- [ ] `gradient_max_norm` is set, typically 5.0.
- [ ] `stop_lr` is 3e-5, not the se_e2_a value.
- [ ] Dynamic selection is enabled only when variable neighbor counts justify it.
- [ ] `stat_file` is set to cache statistics for restart.

## References

- [DPA3 descriptor documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa3.html)
- [DPA3 paper](https://arxiv.org/abs/2506.01686)
