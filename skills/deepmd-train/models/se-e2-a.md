# se_e2_a training reference

Read this file only after the user chooses se_e2_a/DeepPot-SE, or when this model is the best fit for the task. Keep the shared data checks, train/monitor/freeze/test workflow in `../SKILL.md`; this file only records se_e2_a-specific choices.

## When to choose se_e2_a

Use se_e2_a for a robust DeepPot-SE baseline, broad compatibility, limited compute, or small-to-medium systems where a mature production model is preferred over a larger attention/message-passing model.

## Extra inputs to collect

- Cutoff radius, usually `rcut: 6.0` A unless the chemistry requires longer interactions.
- Neighbor selection `sel`, one value per element type. If unsure, run:

```bash
dp --pt neighbor-stat -s /path/to/data -r 6.0 -t O H
```

Use values slightly above the reported maximum.

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
  }
}
```

Set the virial prefactors to `0` when virial labels are unavailable or should not be trained.

## Key hyperparameters

### Descriptor

| Parameter       | Description                         | Typical value    |
| --------------- | ----------------------------------- | ---------------- |
| `rcut`          | Cutoff radius (A)                   | 6.0              |
| `rcut_smth`     | Smooth cutoff start (A)             | 0.5              |
| `sel`           | Max neighbors per type              | System-dependent |
| `neuron`        | Embedding net sizes                 | `[25, 50, 100]`  |
| `axis_neuron`   | Axis matrix dimension               | 16               |
| `type_one_side` | Share embedding across center types | `true`           |

### Fitting and training defaults

| Parameter               | Typical value           | Note                                |
| ----------------------- | ----------------------- | ----------------------------------- |
| `fitting_net.neuron`    | `[240, 240, 240]`       | Standard fitting network            |
| `fitting_net.resnet_dt` | `true`                  | Use timestep in ResNet              |
| `numb_steps`            | `400000`-`1000000`      | Match data size and accuracy target |
| `batch_size`            | `"auto"` or `"auto:32"` | Use common training data section    |

### Loss prefactors

| JSON keys                       | Start | Limit | Note                             |
| ------------------------------- | ----- | ----- | -------------------------------- |
| `start_pref_e` / `limit_pref_e` | 0.02  | 1     | Energy weight                    |
| `start_pref_f` / `limit_pref_f` | 1000  | 1     | Force-dominated early training   |
| `start_pref_v` / `limit_pref_v` | 0.02  | 1     | Use 0 when virial is not trained |

## se_e2_a checklist

- [ ] `sel` has one entry per element in `type_map` and is backed by `neighbor-stat` when unknown.
- [ ] `rcut` is reasonable for the chemistry, typically 6.0-9.0 A.
- [ ] Virial prefactors match whether virial labels are present.
- [ ] Model is frozen and tested using the backend command selected in the top-level workflow.

## References

- [se_e2_a descriptor documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/train-se-e2-a.html)
- [Training advanced options](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training-advanced.html)
