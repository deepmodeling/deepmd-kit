# ADAPT: Atomistic DPA Adaptation for Property Tasks

**ADAPT** is a scikit-learn-style Python package for fine-tuning pre-trained DPA models on your own materials or molecular property dataset. No DeePMD-kit JSON configs or `dp train` pipelines to write.

## Installation

```bash
pip install deepmd-kit[dpa-adapt]
```

Installs `scikit-learn`, `dpdata`, `ase`, `rdkit`, and `e3nn` alongside DeePMD-kit. For GPU PyTorch, install your preferred PyTorch build first.

## Quickstart

Five lines to fine-tune and predict on CPU:

```python
from dpa_adapt import DPAFineTuner

model = DPAFineTuner(pretrained="DPA-3.1-3M", strategy="frozen_sklearn", predictor="rf")
model.fit(train_data="data/train", target_key="bandgap")
preds = model.predict("data/test").predictions
model.freeze("model.pth")
```

For a complete runnable example (QM9 HOMO–LUMO gap, ~5 min on CPU), see [`../../examples/dpa_adapt/`](../../examples/dpa_adapt/).

## Fine-tuning strategies

The strategy is the core choice. All four share the same pre-trained DPA backbone and differ in how much of it gets updated:

| Strategy         | Core Mechanism                                  | Target Data Size | Hardware     | Primary Use Case                          |
| :--------------- | :---------------------------------------------- | :--------------- | :----------- | :---------------------------------------- |
| `frozen_sklearn` | Frozen backbone + scikit-learn regressor        | Small (\<1k)     | CPU only     | Ultra-fast benchmarking & prototyping     |
| `linear_probe`   | Frozen backbone + gradient-descent linear head  | Medium (1k–10k)  | CPU / GPU    | Balanced efficiency for linear properties |
| `finetune`       | End-to-end full parameter fine-tuning           | Large (>10k)     | GPU required | Maximum accuracy on large datasets        |
| `mft`            | Multi-task co-training (property + force field) | Small / low-data | GPU required | Mitigating representation collapse        |

```python
# frozen_sklearn — CPU, no dp train, three predictor choices
model = DPAFineTuner(
    pretrained="DPA-3.1-3M",
    strategy="frozen_sklearn",
    predictor="rf",  # "rf" | "linear" | "mlp"
    pooling="mean",  # "mean" | "sum" | "mean+std" | "mean+std+max+min"
)
model.fit(train_data="/data/train", target_key="homo")

# linear_probe / finetune — same interface, different depth
model = DPAFineTuner(
    pretrained="DPA-3.1-3M", strategy="linear_probe", property_name="homo"
)
model.fit(train_data="/data/train", valid_data="/data/valid", target_key="homo")

# mft — downstream property head + auxiliary force-field head jointly
model = DPAFineTuner(
    pretrained="/path/to/DPA-3.1-3M.pt",
    strategy="mft",
    property_name="homo",
    aux_branch="MP_traj_v024_alldata_mixu",
)
model.fit(train_data="/data/qm9", aux_data="/data/spice2")
```

## Data preparation

Your data must be in `deepmd/npy` format. `auto_convert` detects the input format automatically:

```python
from dpa_adapt import auto_convert

# Structure file → dpdata (POSCAR, OUTCAR, extxyz, cif, …)
auto_convert("POSCAR", "./npy")
auto_convert("calcs/**/OUTCAR", "./npy", fmt="vasp/outcar")  # glob → batch

# CSV with SMILES column → RDKit 3D conformers → deepmd/npy
auto_convert("data.csv", "./npy", property_name="homo", property_col="HOMO")

# Composition formula CSV + template POSCAR → random atomic substitution → deepmd/npy
# CSV: two columns, formula and property value (header optional)
# e.g.  Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1    291.9
auto_convert(
    "compositions.csv",
    "./npy",
    fmt="formula",
    poscar="template.POSCAR",
    property_name="overpotential",
    sets=3,  # random doped structures per composition (default: 1)
)
```

Lower-level helpers:

```python
from dpa_adapt import convert, attach_labels, check_data

convert("calcs/**/OUTCAR", "./npy", fmt="vasp/outcar")
attach_labels(system, head="bandgap", values=np.array([1.0, 2.0, 3.0]))
check_data("/data/system")  # → list[Issue]
```

### Context features (fparam)

fparam lets you condition the model on system-level context such as temperature, pressure, or experimental conditions.

**frozen_sklearn** — pass a dict of numpy arrays at fit and predict time:

```python
model.fit(train_data, conditions={"temperature": T_train})
model.predict(test_data, conditions={"temperature": T_test})
# ConditionManager standardizes and concatenates values to the descriptor
```

**linear_probe / finetune / mft** — place `fparam.npy` of shape `(nframes, fparam_dim)` in each `set.*/` directory alongside `coord.npy`, then declare the dimension at construction:

```python
model = DPAFineTuner(strategy="finetune", fparam_dim=2)
model.fit(train_data)  # reads fparam.npy automatically
```

## Inference and uncertainty

After training, save a portable frozen bundle and load it with `DPAPredictor` — no training dependencies required:

```python
model.freeze("model.pth")

from dpa_adapt import DPAPredictor

pred = DPAPredictor("model.pth")
result = pred.predict("/data/test")  # DotDict: .predictions
metrics = pred.evaluate("/data/test")  # DotDict: .mae, .rmse, .r2
```

Uncertainty estimation is available for `frozen_sklearn` models:

- **RF**: native out-of-bag variance, always available
- **MLP**: committee of N independently-seeded clones; set `n_committee` at load time
- **Ridge**: not supported

```python
pred = DPAPredictor("model.pth", n_committee=5)
result = pred.predict("/data/test", return_uncertainty=True)
# result.predictions  — shape (n,)
# result.uncertainty  — shape (n,), std across committee members
```

Uncertainty estimates can drive active learning (query most uncertain candidates) or feed into Bayesian optimization over composition space.

## Cross-validation

Formula-grouped splitting prevents same-composition leakage between folds:

```python
from dpa_adapt import cross_validate, train_test_split, load_dataset

systems = load_dataset("/data/root", label_key="energy")
train, valid, test = train_test_split(systems, group_by="formula", seed=42)

result = cross_validate(model, systems, label_key="energy", cv=5, group_by="formula")
# → {"aggregate": {"mae_mean": ..., "rmse_std": ...}, ...}
```

## Python API

```python
from dpa_adapt import (
    DPAFineTuner,  # fine-tune (strategies: frozen_sklearn, linear_probe, finetune, mft)
    DPAPredictor,  # inference from frozen bundles
    extract_descriptors,  # standalone descriptor extraction
    cross_validate,  # leak-proof cross-validation
    train_test_split,  # formula-grouped splitting
    auto_convert,  # format-sniffing data conversion
    smiles_to_npy,  # CSV+SMILES → deepmd/npy
    formula_csv_to_npy,  # composition formula CSV + POSCAR → deepmd/npy
    convert,  # structure file → deepmd/npy
    batch_convert,  # glob-based batch conversion
    check_data,  # data sanity checks
    attach_labels,  # inject label arrays
    load_dataset,  # label-filtered data loading
)
```

Standalone descriptor extraction:

```python
X = extract_descriptors(
    "/data/systems",
    pretrained="/path/to/DPA-3.1-3M.pt",
    pooling="mean+std",
)
# → np.ndarray (n_frames, feat_dim * 2)
```

## CLI

| Command                     | Description                                                          |
| --------------------------- | -------------------------------------------------------------------- |
| `dpaad fit`                 | Fine-tune (`--strategy frozen_sklearn\|linear_probe\|finetune\|mft`) |
| `dpaad predict`             | Predict with a frozen `.pth` bundle                                  |
| `dpaad evaluate`            | Evaluate against stored labels                                       |
| `dpaad extract-descriptors` | Extract pooled DPA descriptors to `.npy`                             |
| `dpaad cv`                  | Cross-validate                                                       |
| `dpaad data convert`        | Convert structure / CSV / formula → `deepmd/npy`                     |
| `dpaad data validate`       | Sanity-check `deepmd/npy` directories                                |
| `dpaad data attach-labels`  | Inject `.npy` label arrays                                           |

```bash
# Data conversion
dpaad data convert --input POSCAR --output ./npy
dpaad data convert --input data.csv --output ./npy --property-name homo
dpaad data convert --input comps.csv --output ./npy \
    --fmt formula --poscar template.POSCAR --sets 3

# Fine-tune
dpaad fit --train-data ./npy/train --pretrained DPA-3.1-3M \
    --strategy frozen_sklearn --predictor rf --target-key homo --output model.pth

# MFT
dpaad fit --train-data /data/qm9 --aux-data /data/spice2 \
    --pretrained /path/to/DPA-3.1-3M.pt --strategy mft --target-key homo

# Predict / evaluate
dpaad predict --model model.pth --data ./npy/test
dpaad evaluate --model model.pth --data ./npy/test
```

`dpaad --help` does not load torch — all heavy imports are lazy.
