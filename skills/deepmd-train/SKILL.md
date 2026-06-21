---
name: deepmd-train
description: Train DeePMD-kit models with progressive disclosure. Use when the user wants to train a DeePMD-kit potential, prepare an input.json, choose between model families such as se_e2_a/DeepPot-SE and DPA3, run `dp train`, monitor learning curves, freeze checkpoints, or test trained models. Start with model selection and read only the selected model reference under `models/` when model-specific configuration is needed.
compatibility: Requires deepmd-kit installed. The selected backend and model may require PyTorch, TensorFlow, JAX, Paddle, GPU support, or custom OP libraries.
license: LGPL-3.0-or-later
metadata:
  author: iProzd
  version: '1.1'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# DeePMD-kit Training

Use this skill to guide DeePMD-kit model training without loading every model-specific recipe up front.
The workflow is intentionally progressive:

1. Understand the user's data, target accuracy, compute budget, and deployment backend.
1. Choose an appropriate model family.
1. Read only the reference file for the selected model under [`models/`](models/).
1. Generate or edit `input.json`, run training, monitor, freeze, and test.

## Progressive disclosure protocol

Do not start by reading every model document. First classify the request:

- If the user already named a model, read only that model reference.
- If the user asks for a recommendation, collect the decision inputs below, choose a model, then read only the selected reference.
- If model-specific parameters are not needed yet, stay in this top-level workflow.

Available model references:

| Model reference                          | Read when                                                                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [`models/se-e2-a.md`](models/se-e2-a.md) | The user wants a classical DeepPot-SE baseline, broad compatibility, or a smaller/established production model.                          |
| [`models/dpa3.md`](models/dpa3.md)       | The user wants a high-accuracy DPA3/LAM workflow, large/diverse datasets, dynamic neighbor selection, or pretrained DPA3-style training. |

## Model selection

Ask only for missing information that changes the choice. Prefer reasonable defaults when the answer is obvious from context.

Key inputs:

- Data format and size: deepmd/npy, deepmd/hdf5, mixed type, number of systems/frames/elements.
- Target: quick baseline, production accuracy, large atomic model, transfer/fine-tuning, or deployment in MD.
- Compute: CPU/GPU, available memory, single-node vs. distributed training.
- Backend/deployment: PyTorch/TensorFlow/JAX/Paddle training; LAMMPS, Python inference, or other downstream use.
- Labels: energy/force only or also virial/stress.
- System diversity: single chemistry/phase vs. diverse multi-domain datasets.

Recommended defaults:

- Choose **se_e2_a** for a robust baseline, small to medium systems, compatibility-focused workflows, or when compute is limited.
- Choose **DPA3** for high accuracy on diverse datasets, LAM-style training, or when the user explicitly asks for DPA3, DPA-3, LiGS, dynamic neighbor selection, or pretrained DPA3 variants.

## Common workflow

### 1. Confirm environment

```bash
dp --version
```

For PyTorch training, use `dp --pt ...`; for TensorFlow, use `dp ...`; for other backends, confirm the installed backend first.

### 2. Confirm training data

Training data should be in DeePMD format, typically deepmd/npy or deepmd/hdf5. If the user has raw electronic-structure outputs, convert them first with dpdata before writing the training input.

Minimum information needed to build `input.json`:

- `type_map`
- training system paths
- validation system paths
- whether virial labels are present and should be trained
- target number of steps or accuracy/time budget
- model choice

### 3. Read the selected model reference

After selecting a model, read the corresponding file under [`models/`](models/) and apply its model-specific configuration, hyperparameters, and caveats.

### 4. Train

```bash
dp --pt train input.json
```

Use the backend-specific command if not using PyTorch.

Restart from a checkpoint when needed:

```bash
dp --pt train input.json --restart model.ckpt.pt
```

### 5. Monitor

Training progress is usually written to `lcurve.out`. Check for:

- decreasing validation RMSE
- NaN or exploding losses
- train/validation divergence
- learning-rate schedule behaving as expected

### 6. Freeze and test

```bash
dp --pt freeze -o model.pth
dp --pt test -m model.pth -s /path/to/test_system -n 30
```

Adjust the backend flags and output extension for non-PyTorch models.

## Agent checklist

- [ ] Model was selected before reading model-specific details.
- [ ] Only the selected model reference was loaded.
- [ ] Training/validation data paths exist or are clearly marked as placeholders.
- [ ] `type_map` matches the data and model/pretrained checkpoint.
- [ ] Virial loss is enabled only when virial labels are available and desired.
- [ ] Backend command matches the selected model and installed DeePMD-kit environment.
- [ ] The generated `input.json` is valid JSON.
- [ ] Training was monitored via `lcurve.out` or equivalent logs.
- [ ] Final model was frozen and tested when requested.

## References

- [Training documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/training.html)
- [Training input documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/train/train-input.html)
- [Model documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/index.html)
