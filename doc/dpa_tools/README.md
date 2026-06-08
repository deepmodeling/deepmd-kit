# dpa_tools

`dpa_tools` is a **scikit-learn-style Python API** for fine-tuning pre-trained DPA
series models on your own dataset. You construct a
`DPAFineTuner`, call `fit(...)` then `predict(...)`, and pick a transfer-learning
strategy — no DeePMD-kit JSON configs or `dp train` pipelines to write. The usual
goal is adapting a large pre-trained model to a downstream materials or molecular
property (energy, band gap, HOMO–LUMO gap, …) from a modest labeled dataset.

It ships as the `dpa_tools` package alongside `deepmd-kit`,
and the same workflow is also exposed on the command line as `dp dpa`.

## Installation

```bash
pip install deepmd-kit[dpa-tools]
```

The `dpa-tools` extra installs the Python dependencies used by this package,
including `scikit-learn`, `dpdata`, `torch`, `rdkit`, and `e3nn`. For a
CUDA/GPU PyTorch build, install the desired PyTorch variant first, then install
this extra.

## Quickstart

Fine-tune a frozen-descriptor + scikit-learn head and predict — under 10 lines:

```python
from dpa_tools import DPAFineTuner

# `pretrained` accepts a built-in model name (auto-downloaded) or a local .pt path
model = DPAFineTuner(pretrained="DPA-3.1-3M", strategy="frozen_sklearn", predictor="rf")
model.fit(train_data="data/train", target_key="bandgap")  # fine-tune on labeled structures

preds = model.predict("data/test").predictions            # predict on new structures
model.freeze("model.dp-sklearn.pth")                      # save a reusable bundle
```

Your data must be in `deepmd/npy` format (see [Data preparation](#data-preparation)
to convert structure files, VASP output, SMILES CSVs, or composition formulas).
For a complete,
runnable example that fits a QM9 HOMO–LUMO-gap model on CPU in **under 5
minutes**, see [`demo/`](demo/) — it ships with 50 pre-processed molecules so you
only need a pre-trained checkpoint.

## Fine-tuning strategies

The strategy is the main choice you make. All four adapt the same pre-trained
DPA backbone; they differ in how much of it they train:

| Strategy | What it does | Best for |
|----------|--------------|----------|
| `frozen_sklearn` (default) | Freeze the backbone, extract descriptors once, fit a scikit-learn head (RF / Ridge / MLP) | Small data (<1k samples), CPU-only, fastest iteration |
| `linear_probe` | Freeze the backbone, train only a property fitting net | Medium data, GPU available |
| `finetune` | Fine-tune the full network | Larger data, GPU available |
| `mft` | Multi-task: property head + an auxiliary force-field head trained jointly | Prevents representation collapse on small property datasets |

```python
# frozen_sklearn (CPU, no dp train): extract once, fit a scikit-learn head
model = DPAFineTuner(
    pretrained="DPA-3.1-3M",     # built-in name → auto-downloaded; or a local path
    strategy="frozen_sklearn",
    predictor="rf",              # "rf" | "linear"/"ridge" | "mlp"
    pooling="mean",              # "mean" | "sum" | "mean+std" | "mean+std+max+min"
)
model.fit(train_data="/data/train", target_key="homo")
model.predict("/data/test")
model.freeze("model.dp-sklearn.pth")

# mft: multi-task fine-tuning (downstream property head + auxiliary force-field head)
model = DPAFineTuner(
    pretrained="/path/to/DPA-3.1-3M.pt",
    strategy="mft",
    property_name="homo",
    aux_branch="MP_traj_v024_alldata_mixu",
)
model.fit(train_data="/data/qm9", aux_data="/data/spice2")
```

## Python API

```python
from dpa_tools import (
    DPAFineTuner,          # fine-tune (strategies: frozen_sklearn, linear_probe, finetune, mft)
    DPAPredictor,          # read-only inference from frozen bundles
    extract_descriptors,   # standalone descriptor extraction
    cross_validate,        # leak-proof cross-validation
    train_test_split,      # formula-grouped data splitting
    # data tools
    auto_convert,          # sniff input → route to SMILES, formula, or dpdata pipeline
    smiles_to_npy,         # CSV+SMILES → deepmd/npy (train/valid split)
    formula_to_npy,        # CSV+composition formula + POSCAR → deepmd/npy (random doping)
    convert,               # structure file → deepmd/npy (via dpdata)
    batch_convert,         # glob-based batch conversion
    check_data,            # data sanity checks
    attach_labels,         # inject external label arrays
    load_dataset,          # label-filtered data loading
)
```

### DPAPredictor

Load a frozen bundle for inference, with no training dependencies:

```python
pred = DPAPredictor("model.dp-sklearn.pth")
result = pred.predict("/data/test")           # DotDict with .predictions
metrics = pred.evaluate("/data/test")         # DotDict with .mae, .rmse, .r2

# uncertainty: RF native, MLP via committee, Ridge raises
result = pred.predict("/data/test", return_uncertainty=True)
# → .predictions, .uncertainty
```

### Descriptor extraction

Get pooled DPA descriptors as a NumPy array (e.g. to feed your own model):

```python
X = extract_descriptors(
    "/data/systems",
    pretrained="/path/to/DPA-3.1-3M.pt",
    pooling="mean+std",
)
# → np.ndarray (n_frames, feat_dim * 2)
```

### Data preparation

One command auto-detects the input format — CSV with a SMILES column routes
through RDKit (3D conformer generation), `fmt="formula"` routes through
composition-based random doping from a template POSCAR, and everything else
goes through dpdata:

```python
from dpa_tools import auto_convert

# CSV with SMILES → RDKit generates 3D coords, writes train/valid deepmd/npy
auto_convert("data.csv", "./npy", property_name="homo", property_col="HOMO")

# Structure file → auto-detected by dpdata (POSCAR, OUTCAR, extxyz, cif, …)
auto_convert("POSCAR", "./npy")

# Composition formula CSV + template POSCAR → random doping → deepmd/npy
auto_convert("compositions.csv", "./npy", fmt="formula", poscar="template.POSCAR")

# Lower-level helpers
convert("POSCAR", "out_dir", fmt="extxyz", type_map=["Cu", "O"])
convert("calcs/**/OUTCAR", "npy_root", fmt="vasp/outcar")  # glob → batch mode
attach_labels(system, head="bandgap", values=np.array([1.0, 2.0, 3.0]))
check_data("/data/system")   # → list[Issue]
```

### Cross-validation & splitting

Formula-grouped to prevent same-molecule leakage between folds:

```python
from dpa_tools import cross_validate, train_test_split, load_dataset

systems = load_dataset("/data/root", label_key="energy")
train, valid, test = train_test_split(systems, group_by="formula", seed=42)

result = cross_validate(model, systems, label_key="energy", cv=5, group_by="formula")
# → {"aggregate": {"mae_mean": ..., "rmse_std": ...}, ...}
```

## CLI

The same workflow is available under `dp dpa` (two-level nesting for data tools):

| Command | Description |
|---------|-------------|
| `dp dpa fit` | Fine-tune a model with any strategy (`--strategy frozen_sklearn\|linear_probe\|finetune\|mft`) |
| `dp dpa predict` | Predict with a frozen `.pth` bundle |
| `dp dpa evaluate` | Evaluate a frozen `.pth` against stored labels |
| `dp dpa extract-descriptors` | Extract pooled DPA descriptors to `.npy` |
| `dp dpa cv` | Cross-validate (metric estimation, no model output) |
| `dp dpa data convert` | Convert a structure/CSV file or glob → `deepmd/npy` (auto-sniffs SMILES vs. structure, or `--fmt formula` for composition formulas) |
| `dp dpa data validate` | Sanity-check `deepmd/npy` directories |
| `dp dpa data attach-labels` | Inject `.npy` label arrays into a system |

```bash
# Convert data (format auto-detected)
dp dpa data convert --input data.csv --output ./npy --property-name homo   # CSV+SMILES
dp dpa data convert --input POSCAR --output ./npy                          # structure file
dp dpa data convert --input "calcs/**/OUTCAR" --output ./npy_root          # glob → batch
dp dpa data convert --input comps.csv --output ./npy --fmt formula \\      # formula CSV
    --poscar template.POSCAR --sets 3

# Fine-tune
dp dpa fit --train-data ./npy/train --pretrained DPA-3.1-3M \
  --strategy frozen_sklearn --predictor rf --target-key homo --output model.pth

# Multi-task fine-tuning (MFT)
dp dpa fit --train-data /data/qm9 --aux-data /data/spice2 \
  --pretrained /path/to/DPA-3.1-3M.pt --strategy mft --target-key homo

# Predict / evaluate with a frozen bundle
dp dpa predict --model model.pth --data ./npy/test --output preds.npy
dp dpa evaluate --model model.pth --data ./npy/test
```

`dp dpa --help` does not load torch — the parser is pure argparse in
`deepmd/main.py`, and the handlers (and the DPA stack) are imported lazily only
when a `dp dpa ...` command actually runs.

