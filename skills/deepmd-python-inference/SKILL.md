---
name: deepmd-python-inference
description: Run Python inference with DeePMD-kit models using the DeepPot API. Use when the user wants to load a trained/frozen DeePMD model (.pth or .pb) or a built-in pretrained model (e.g., DPA-3.2-5M) in Python, predict energy/force/virial for atomic configurations, evaluate descriptors, or calculate model deviation between multiple models. Also covers using `dp test` CLI for batch evaluation against labeled data.
compatibility: Requires deepmd-kit Python package installed. PyTorch backend for .pth models, TensorFlow for .pb models.
license: LGPL-3.0-or-later
metadata:
  author: iProzd
  version: '1.0'
  repository: https://github.com/deepmodeling/deepmd-kit
---

# DeePMD-kit Python Inference

Load a trained DeePMD-kit model in Python and predict energy, forces, and virial for atomic configurations. Also covers CLI-based testing with `dp test`.

## Quick Start

```python
from deepmd.infer import DeepPot
import numpy as np

dp = DeepPot("model.pth")
coord = np.array([[1, 0, 0], [0, 0, 1.5], [1, 0, 3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1, 0, 1]
e, f, v = dp.eval(coord, cell, atype)
```

## Agent Responsibilities

1. Determine the model source:
   - Frozen model file (`.pth` for PyTorch, `.pb` for TensorFlow)
   - Built-in pretrained model name (e.g., `DPA-3.2-5M`)
   - Checkpoint file (requires freezing first)
1. Determine the inference task:
   - Single-frame prediction (energy, force, virial)
   - Batch prediction over multiple frames
   - Descriptor evaluation
   - Model deviation calculation
   - CLI-based testing against labeled data
1. Help the user prepare input arrays in the correct format.
1. Run inference and report results.

## Python API: DeepPot

### Load a Model

```python
from deepmd.infer import DeepPot

# From a frozen PyTorch model
dp = DeepPot("model.pth")

# From a frozen TensorFlow model
dp = DeepPot("graph.pb")

# From a built-in pretrained model (auto-downloads if not cached)
dp = DeepPot("DPA-3.2-5M")
```

Built-in pretrained model names include `DPA-3.2-5M`, `DPA-3.1-3M`, `DPA3-Omol-Large`, etc. DeePMD-kit will automatically download and cache the model on first use.

### Predict Energy, Forces, and Virial

```python
import numpy as np
from deepmd.infer import DeepPot

dp = DeepPot("model.pth")

# Prepare inputs
# coord: (nframes, natoms * 3) in Angstrom
# cell: (nframes, 9) cell vectors in Angstrom, row-major
# atype: list of atom type indices (length natoms)

coord = np.array(
    [
        [
            0.0,
            0.0,
            0.0,  # atom 0 (O)
            0.0,
            0.0,
            1.0,  # atom 1 (H)
            0.0,
            1.0,
            0.0,
        ]  # atom 2 (H)
    ]
).reshape([1, -1])

cell = np.diag([10.0, 10.0, 10.0]).reshape([1, -1])

# atype indices correspond to type_map order in the model
# e.g., if type_map = ["O", "H"], then O=0, H=1
atype = [0, 1, 1]

e, f, v = dp.eval(coord, cell, atype)

print(f"Energy (eV): {e}")  # shape: (nframes, 1)
print(f"Forces (eV/A): {f}")  # shape: (nframes, natoms, 3)
print(f"Virial (eV): {v}")  # shape: (nframes, 9)
```

### Non-periodic Systems

For non-periodic (isolated) systems, pass `cell=None`:

```python
e, f, v = dp.eval(coord, None, atype)
```

### Batch Prediction

Process multiple frames at once:

```python
nframes = 10
natoms = 3

coords = np.random.rand(nframes, natoms * 3)
cells = np.tile(np.diag([10.0, 10.0, 10.0]).reshape([1, -1]), (nframes, 1))
atype = [0, 1, 1]

e, f, v = dp.eval(coords, cells, atype)
# e: (nframes, 1)
# f: (nframes, natoms, 3)
# v: (nframes, 9)
```

### Evaluate Descriptors

Extract the descriptor (atomic environment representation) from the model:

```python
descriptors = dp.eval_descriptor(coord, cell, atype)
# shape: (nframes, natoms, ndesc)
```

This can also be done via CLI:

```bash
dp eval-desc -m model.pth -s /path/to/system -o desc_output
```

### Calculate Model Deviation

Compare predictions from multiple models to estimate uncertainty:

```python
from deepmd.infer import calc_model_devi, DeepPot

coord = np.array([[1, 0, 0], [0, 0, 1.5], [1, 0, 3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1, 0, 1]

graphs = [DeepPot("model_0.pth"), DeepPot("model_1.pth")]
model_devi = calc_model_devi(coord, cell, atype, graphs)
```

Important: avoid loading the same model multiple times in a loop, as this can cause memory leaks.

## CLI Testing: dp test

Test a frozen model against labeled data:

```bash
# Basic test
dp --pt test -m model.pth -s /path/to/test_system -n 30

# Test with detailed output
dp --pt test -m model.pth -s /path/to/test_system -n 30 -d test_detail
```

### dp test Options

| Option           | Description                        |
| ---------------- | ---------------------------------- |
| `-m MODEL`       | Path to the frozen model file      |
| `-s SYSTEM`      | Path to the test data system       |
| `-n NUMB`        | Number of test frames              |
| `-d DETAIL`      | Output prefix for detailed results |
| `--shuffle-test` | Shuffle test frames                |

### Output

`dp test` prints RMSE values for energy, force, and virial:

```
Energy RMSE        : 1.234e-03 eV
Energy RMSE/Natoms : 6.427e-06 eV
Force  RMSE        : 2.345e-02 eV/A
Virial RMSE        : 5.678e-02 eV
Virial RMSE/Natoms : 2.957e-04 eV
```

With `-d test_detail`, per-frame predictions are saved to files for further analysis.

## Complete Example: Train, Freeze, and Inference

```python
import subprocess
import numpy as np
from deepmd.infer import DeepPot

# Step 1: Train (run in shell)
# dp --pt train input.json

# Step 2: Freeze (run in shell)
# dp --pt freeze -o model.pth

# Step 3: Python inference
dp = DeepPot("model.pth")

# Load test data from deepmd format
coord = np.load("test_system/set.000/coord.npy")  # (nframes, natoms*3)
cell = np.load("test_system/set.000/box.npy")  # (nframes, 9)
atype_raw = np.loadtxt("test_system/type.raw", dtype=int).tolist()

# Predict
e, f, v = dp.eval(coord, cell, atype_raw)

# Compare with reference
ref_energy = np.load("test_system/set.000/energy.npy")
ref_force = np.load("test_system/set.000/force.npy")

natoms = len(atype_raw)
energy_rmse = np.sqrt(np.mean((e.flatten() - ref_energy.flatten()) ** 2)) / natoms
force_rmse = np.sqrt(np.mean((f.reshape(-1) - ref_force.reshape(-1)) ** 2))

print(f"Energy RMSE/atom: {energy_rmse:.6f} eV")
print(f"Force RMSE:       {force_rmse:.6f} eV/A")
```

## Using Pretrained Models Directly

Built-in pretrained models can be used without any training:

```python
from deepmd.infer import DeepPot
import numpy as np

# Auto-downloads DPA-3.2-5M on first use
dp = DeepPot("DPA-3.2-5M")

# Water molecule example
coord = np.array(
    [
        [0.000, 0.000, 0.117],  # O
        [0.000, 0.757, -0.469],  # H
        [0.000, -0.757, -0.469],  # H
    ]
).reshape([1, -1])

cell = np.diag([10.0, 10.0, 10.0]).reshape([1, -1])
atype = [0, 1, 1]  # Check model's type_map for correct indices

e, f, v = dp.eval(coord, cell, atype)
print(f"Energy: {e[0][0]:.6f} eV")
print(f"Forces:\n{f[0]}")
```

To download pretrained models explicitly:

```bash
dp pretrained download DPA-3.2-5M
dp pretrained download DPA-3.1-3M
dp pretrained download DPA-3.2-5M --cache-dir ./models
```

## Input Array Format Reference

| Array   | Shape                | Unit     | Description                                   |
| ------- | -------------------- | -------- | --------------------------------------------- |
| `coord` | (nframes, natoms\*3) | Angstrom | Atomic coordinates, flattened                 |
| `cell`  | (nframes, 9)         | Angstrom | Cell vectors, row-major (a1x,a1y,a1z,a2x,...) |
| `atype` | (natoms,)            | -        | Atom type indices matching model's type_map   |

| Output | Shape                | Unit | Description             |
| ------ | -------------------- | ---- | ----------------------- |
| `e`    | (nframes, 1)         | eV   | Total energy per frame  |
| `f`    | (nframes, natoms, 3) | eV/A | Forces on each atom     |
| `v`    | (nframes, 9)         | eV   | Virial tensor per frame |

## Agent Checklist

- [ ] Model file exists and is accessible (`.pth`, `.pb`, or valid pretrained name)
- [ ] `coord` array is shaped (nframes, natoms\*3) and in Angstrom
- [ ] `cell` array is shaped (nframes, 9) or `None` for non-periodic systems
- [ ] `atype` indices match the model's `type_map` ordering
- [ ] For model deviation, multiple models are loaded only once (not in a loop)
- [ ] Results are reported with correct units (eV, eV/A)

## References

- [Python inference documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/inference/python.html)
- [dp test documentation](https://docs.deepmodeling.com/projects/deepmd/en/latest/test/test.html)
- [Pretrained model download](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/pretrained.html)
- [DeepPot API reference](https://docs.deepmodeling.com/projects/deepmd/en/latest/api_py/deepmd.infer.html)
- [DeePMD-kit GitHub](https://github.com/deepmodeling/deepmd-kit)
