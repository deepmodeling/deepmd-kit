# DPA-ADAPT: Atomistic DPA Adaptation for Property Tasks

**DPA-ADAPT** (`dpa-adapt`, Python import `dpa_adapt`) is a toolkit for adapting pretrained DPA models to downstream atomistic property prediction tasks. The main CLI is `dpa-adapt`; the optional short alias is `dpaad`. No DeePMD-kit JSON configs or `dp train` pipelines to write.

## Installation

```bash
pip install deepmd-kit[dpa-adapt]
```

Installs `scikit-learn`, `dpdata`, `ase`, `rdkit`, and `e3nn` alongside DeePMD-kit. For GPU PyTorch, install your preferred PyTorch build first.

## Quickstart

For a complete runnable example (QM9 HOMO–LUMO gap, ~5 min on CPU), see [`../../examples/dpa_adapt/`](../../examples/dpa_adapt/).

## Fine-tuning strategies

The strategy is the core choice. All four share the same pre-trained DPA backbone and differ in how much of it gets updated:

| Strategy         | Core Mechanism                                  | Target Data Size | Hardware     | Primary Use Case                          |
| :--------------- | :---------------------------------------------- | :--------------- | :----------- | :---------------------------------------- |
| `frozen_sklearn` | Frozen backbone + scikit-learn regressor        | Small (\<1k)     | CPU only     | Ultra-fast benchmarking & prototyping     |
| `frozen_head`    | Frozen backbone + DeepMD property fitting head  | Medium (1k–10k)  | CPU / GPU    | Train only the property head while keeping the pretrained DPA backbone frozen |
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
model.fit(train_data="/data/train/*", target_key="homo")
pred = model.predict(data=str("/data/test"))
metrics = model.evaluate(data=str("/data/test"))

# frozen_head / finetune — same interface, different depth
#
# Both strategies delegate to `dp --pt train` under the hood and accept the
# same set of training parameters.  frozen_head freezes the DPA backbone and
# only trains a new property fitting head; finetune updates the full network
# end-to-end.
#
# All parameters below are deepmd-kit native — they map directly to fields
# in the generated input.json.
model = DPAFineTuner(
    pretrained="DPA-3.1-3M",
    strategy="frozen_head",          # "frozen_head" | "finetune"

    # ---- task definition ----
    property_name="homo",            # label key under set.*/
    task_dim=1,                      # output dimensionality
    intensive=True,                  # True = intensive (mean-pooled), False = extensive
    init_branch="SPICE2",            # checkpoint branch for descriptor init

    # ---- fitting net (optional, deepmd-kit defaults: neuron=[240,240,240], tanh, resnet_dt=True) ----
    fitting_net_params={             # deepmd-kit fitting_net fields, e.g.
        "neuron": [128, 128, 128],   #   hidden layer sizes
        "activation_function": "relu",
        "resnet_dt": True,
        # "numb_fparam": …,          #   auto-set from fparam_dim, do not pass manually
    },

    # ---- learning rate schedule (deepmd-kit "exp" scheduler) ----
    learning_rate=1e-3,              # start_lr
    stop_lr=1e-5,                    # end_lr
    decay_steps=1000,                # steps between LR decays
    warmup_steps=0,                  # linear LR warmup (0 = disabled)

    # ---- training loop ----
    max_steps=100_000,               # total training steps
    batch_size="auto:512",           # deepmd-kit batch_size spec
    loss_function="mse",             # "mse" | "smooth_mae"

    # ---- context features (optional) ----
    fparam_dim=0,                    # > 0 reads set.*/fparam.npy automatically

    # ---- reproducibility ----
    seed=42,

    # ---- output ----
    output_dir="./dpa_output",       # input.json, checkpoints, logs
    save_freq=10_000,                # checkpoint save interval (steps)
    disp_freq=1_000,                 # log display interval (steps)
)

model.fit(train_data="/data/train", valid_data="/data/valid")
pred = model.predict(data=str("/data/test"))
metrics = model.evaluate(data=str("/data/test"))

# mft — downstream property head + auxiliary force-field head jointly
#
# Jointly trains a downstream property head with an auxiliary force/energy
# head on a shared DPA descriptor, preventing representation collapse.
# Accepts all frozen_head/finetune params plus MFT-specific ones below.
model = DPAFineTuner(
    pretrained="/path/to/DPA-3.1-3M.pt",
    strategy="mft",

    # ---- task definition (same as frozen_head/finetune) ----
    property_name="homo",            # label key under set.*/ (property mode only)
    task_dim=1,
    intensive=True,
    init_branch="SPICE2",

    # ---- MFT-specific ----
    aux_branch="MP_traj_v024_alldata_mixu",  # checkpoint branch for aux force head
    aux_prob=0.5,                    # aux sampling weight (downstream = 1.0 - aux_prob)
    downstream_task_type="property", # "property" | "ener" (legacy default)
    aux_type_map=None,               # element symbols for aux data (auto-detect if None)
    downstream_type_map=None,        # element symbols for downstream data
    aux_batch_size=None,             # batch size for aux head (None = auto)
    downstream_batch_size=None,      # batch size for downstream head (None = auto)

    # ---- fitting net (optional, for aux head; downstream uses property defaults) ----
    fitting_net_params=None,         # aux fitting net overrides (None = auto-read from ckpt)

    # ---- learning rate schedule (deepmd-kit "exp" scheduler) ----
    learning_rate=1e-3,
    stop_lr=1e-5,
    decay_steps=None,                # None → auto: 1000 for property, 5000 for ener
    warmup_steps=0,

    # ---- training loop ----
    max_steps=50_000,
    batch_size="auto:32",

    # ---- context features (optional) ----
    fparam_dim=0,

    # ---- reproducibility ----
    seed=42,

    # ---- output ----
    output_dir="./mft_output",
    save_freq=10_000,
    disp_freq=1_000,
)
model.fit(train_data="/data/train", aux_data="/data/spice2")
pred = model.predict(data=str("/data/test"))
metrics = model.evaluate(data=str("/data/test"))
```

## Data preparation

DPA-ADAPT trains on `deepmd/npy` data. Use `dpa-adapt data convert` (or the Python
`convert` helper) to route common inputs into the right conversion pipeline:

- **SMILES CSV**: a `.csv` file with a `SMILES`/`smiles` column. RDKit generates 3D
  conformers, or existing `.mol`/`.sdf`/`.xyz`/`.pdb` files can be supplied with
  `mol_dir`.
- **Formula CSV + POSCAR template**: pass `fmt="formula"` and `poscar=...` to create
  doped structures by random substitution on the host-element sublattice.
- **Structure files / trajectories**: POSCAR, OUTCAR, `*.xyz`, `vasprun.xml`, ABACUS,
  CP2K, Gaussian, LAMMPS, ASE, `deepmd/raw`, `deepmd/npy`, LMDB, and other dpdata
  formats. Omit `fmt` when dpdata can infer it; set `fmt` explicitly for ambiguous
  inputs.

```python
from dpa_adapt import convert

# Structure file / trajectory → dpdata → deepmd/npy
convert("POSCAR", "./npy")
convert("OUTCAR", "./npy", fmt="vasp/outcar")

# Glob patterns: one match is converted as one system; multiple matches are batched.
convert("calcs/**/OUTCAR", "./npy_root", fmt="vasp/outcar")

# CSV with a SMILES column → RDKit 3D conformers → deepmd/npy.
# property_col names the input target column and output label name.
convert(
    "molecules.csv",
    "./npy",
    fmt="smiles",          # optional when a SMILES/smiles column is present
    smiles_col="SMILES",
    property_col="HOMO",
    train_ratio=0.9,
)

# CSV + pre-generated molecular structures: skip RDKit conformer generation.
convert(
    "molecules.csv",
    "./npy",
    fmt="smiles",
    smiles_col="SMILES",
    property_col="GAP",
    mol_dir="./mol_files",
    mol_template="id{row}.sdf",
)

# Composition formula CSV + template POSCAR → random atomic substitution → deepmd/npy.
# CSV: header required; defaults are formula_col="formula" and property_col="Property".
# e.g.  formula,Property
#       Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1,291.9
convert(
    "compositions.csv",
    "./npy",
    fmt="formula",
    poscar="template.POSCAR",
    formula_col="formula",
    property_col="bandgap",
    sets=3,        # random doped structures per composition row (default: 1)
    seed=42,
)
```

CLI equivalents:

```bash
# SMILES table
dpa-adapt data convert --input molecules.csv --output ./npy \
  --fmt smiles --smiles-col SMILES --property-col HOMO --train-ratio 0.9

# Formula table + POSCAR template
dpa-adapt data convert --input compositions.csv --output ./npy --fmt formula \
  --poscar template.POSCAR --formula-col formula --property-col bandgap --sets 3

# Structure file or glob of calculation outputs
dpa-adapt data convert --input POSCAR --output ./npy
dpa-adapt data convert --input "calcs/**/OUTCAR" --output ./npy_root --fmt vasp/outcar
```

Lower-level helpers:

```python
from dpa_adapt import convert, attach_labels, check_data

convert("OUTCAR", "./npy", fmt="vasp/outcar")
convert("calcs/**/OUTCAR", "./npy_root", fmt="vasp/outcar")

# Single system
attach_labels("./npy/", head="bandgap", values=np.array([1.0, 2.0, 3.0]))

# Multiple systems: values[i] → sorted(glob("npy/*/"))[i]
labels = np.load("labels.npy")  # shape (n_systems,)
attach_labels("./npy/", head="bandgap", values=labels)

check_data("/data/system")  # → list[Issue]
```

For the full option list and supported dpdata formats, see
[`input_formats.md`](input_formats.md).

### Context features (fparam)

fparam lets you condition the model on system-level context such as temperature, humidity, pressure, or any per-frame scalar.  All strategies use the same interface: place `fparam.npy` of shape `(n_frames, fparam_dim)` in each `set.*/` directory alongside `coord.npy` and declare the dimension at construction.

```python
# works identically for frozen_sklearn, frozen_head, finetune, and mft
model = DPAFineTuner(strategy="frozen_sklearn", fparam_dim=2)
model.fit(train_data="data/train", target_key="property")
# fparam.npy is read automatically — no conditions= dict needed
```

| Strategy | How fparam is used |
|---|---|
| `frozen_sklearn` | columns are standardized via `ConditionManager` and concatenated to the descriptor |
| `frozen_head` / `finetune` / `mft` | passed into the fitting net as `numb_fparam` |

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

Formula-grouped splitting prevents same-composition leakage between folds.
`group_by` accepts `"formula"` (uses each system's directory name as the group
key — requires directories named by formula, e.g. `H2O/`, `CH4/`) or a list
of labels the same length as `systems`:

```python
from dpa_adapt import cross_validate, train_test_split, load_dataset

systems = load_dataset("/data/root", label_key="energy")

# Case 1: directory names are formulas (e.g. data/H2O/, data/CH4/)
train, valid, test = train_test_split(systems, group_by="formula", seed=42)

# Case 2: directory names are not formulas (e.g. QM9's sys_0000, sys_0001, …)
formulas = ["H2O", "H2O", "CH4", "CH4", ...]  # one label per system
train, valid, test = train_test_split(systems, group_by=formulas, seed=42)

# Cross-validate (same group_by options apply)
result = cross_validate(model, systems, label_key="energy", cv=5, group_by=formulas)
# → {"aggregate": {"mae_mean": ..., "rmse_std": ...}, ...}
```

## Python API

```python
from dpa_adapt import (
    DPAFineTuner,  # fine-tune (strategies: frozen_sklearn, frozen_head, finetune, mft)
    DPAPredictor,  # inference from frozen bundles
    extract_descriptors,  # standalone descriptor extraction
    cross_validate,  # leak-proof cross-validation
    train_test_split,  # formula-grouped splitting
    convert,  # format-sniffing data conversion
    smiles_to_npy,  # CSV+SMILES → deepmd/npy
    formula_to_npy,  # composition formula CSV + POSCAR → deepmd/npy
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

| Command | Description |
|---------|-------------|
| `dpa-adapt fit` / `dpaad fit` | Fine-tune (`--strategy frozen_sklearn\|frozen_head\|finetune\|mft`) |
| `dpa-adapt predict` / `dpaad predict` | Predict with a frozen `.pth` bundle |
| `dpa-adapt evaluate` / `dpaad evaluate` | Evaluate against stored labels |
| `dpa-adapt extract-descriptors` / `dpaad extract-descriptors` | Extract pooled DPA descriptors to `.npy` |
| `dpa-adapt cv` / `dpaad cv` | Cross-validate |
| `dpa-adapt data convert` / `dpaad data convert` | Convert structure / CSV / formula → `deepmd/npy` |
| `dpa-adapt data validate` / `dpaad data validate` | Sanity-check `deepmd/npy` directories |
| `dpa-adapt data attach-labels` / `dpaad data attach-labels` | Inject `.npy` label arrays |

```bash
# Data conversion
# Structure file
dpa-adapt data convert --input POSCAR --output ./npy

# SMILES CSV: --property-col names the input target column and output label name.
dpaad data convert --input data.csv --output ./npy --fmt smiles \
  --property-col homo

# Formula CSV + POSCAR template
dpa-adapt data convert --input comps.csv --output ./npy --fmt formula \
  --poscar template.POSCAR --formula-col formula --property-col bandgap --sets 3

# Fine-tune
dpa-adapt fit --train-data ./npy/train --pretrained DPA-3.1-3M \
  --strategy frozen_sklearn --predictor rf --target-key homo --output model.pth

# MFT
dpaad fit --train-data /data/qm9 --aux-data /data/spice2 \
  --pretrained /path/to/DPA-3.1-3M.pt --strategy mft --target-key homo

# Predict / evaluate
dpa-adapt predict --model model.pth --data ./npy/test --output pred.npy
dpa-adapt evaluate --model model.pth --data ./npy/test
```

`dpa-adapt --help` and `dpaad --help` do not load torch — all heavy imports are lazy.
