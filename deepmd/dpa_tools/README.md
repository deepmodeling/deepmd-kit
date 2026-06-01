# dpa_tools

Fine-tuning, descriptor extraction, cross-validation, and data utilities for
DPA-3 pretrained models.  Lives as a self-contained subpackage of `deepmd-kit`
at `deepmd.dpa_tools`.

## Relationship with deepmd-kit

`dpa_tools` sits on top of deepmd-kit without modifying any existing module:

- **Model loading**: `_backend.py` is the single choke point that imports
  `deepmd.pt.model.model.get_model` and `deepmd.pt.train.wrapper.ModelWrapper`
  to load DPA-3 checkpoints and extract descriptors.  No other file in
  `dpa_tools` touches `deepmd.pt.*` directly.
- **Training**: shells out to `dp --pt train` / `dp --pt freeze` /
  `dp --pt test`, auto-generating `input.json` config files.
- **Inference**: deepmd-kit's built-in `DeepProperty` handles neural-network
  models; dpa_tools adds a lightweight frozen-descriptor + sklearn-head path.
- **CLI**: registered as `dp dpa` subcommand group via `deepmd/main.py`.
  Torch and all DPA dependencies are loaded lazily — only when a `dp dpa ...`
  command actually runs.
- **Lazy import**: `import deepmd.dpa_tools` does **not** trigger a `torch`
  import.  `dp dpa --help` is equally lightweight.

## Python API

```python
from deepmd.dpa_tools import (
    DPAFineTuner,       # train (frozen sklearn / finetune / linear probe)
    DPAPredictor,       # read-only inference from frozen bundles
    MFTFineTuner,       # multi-task fine-tuning
    DPATrainer,         # single-task dp --pt train wrapper
    extract_descriptors, # standalone descriptor extraction
    cross_validate,     # leak-proof cross-validation
    train_test_split,   # formula-grouped data splitting
    # data tools
    convert,            # structure file → deepmd/npy
    batch_convert,      # glob-based batch conversion
    check_data,         # data sanity checks
    attach_labels,      # inject external label arrays
    load_dataset,       # label-filtered data loading
)
```

### DPAFineTuner

Four training strategies:

| Strategy | Description | Best for |
|----------|------------|----------|
| `frozen_sklearn` | Freeze descriptor, extract once, fit sklearn head (RF/Ridge/MLP) | Small data (<1k samples), CPU inference |
| `linear_probe` | Freeze backbone, train property fitting net only | Medium data, GPU |
| `finetune` | Full-network fine-tuning | Larger data, GPU |
| `scratch` | Train from random init (experimental) | Large-scale data only |

```python
model = DPAFineTuner(
    pretrained="/path/to/DPA-3.1-3M.pt",
    strategy="frozen_sklearn",
    predictor="rf",
    pooling="mean",
)
model.fit(train_data="/data/train", target_key="homo")
model.predict("/data/test")
model.freeze("model.dp-sklearn.pth")
```

### DPAPredictor

```python
pred = DPAPredictor("model.dp-sklearn.pth")
result = pred.predict("/data/test")           # DotDict with .predictions
metrics = pred.evaluate("/data/test")         # DotDict with .mae, .rmse, .r2

# uncertainty: RF native, MLP via committee, Ridge raises
result = pred.predict("/data/test", return_uncertainty=True)
# → .predictions, .uncertainty
```

### MFTFineTuner

Joint downstream property head + auxiliary force-field head (arXiv:2601.08486):

```python
mft = MFTFineTuner(
    pretrained="/path/to/DPA-3.1-3M.pt",
    downstream_task_type="property",
    property_name="homo",
    aux_branch="MP_traj_v024_alldata_mixu",
)
mft.fit(train_data="/data/qm9", aux_data="/data/spice2")
mft.evaluate("/data/qm9_test")
```

### Descriptor extraction

```python
X = extract_descriptors(
    "/data/systems",
    pretrained="/path/to/DPA-3.1-3M.pt",
    pooling="mean+std",
)
# → np.ndarray (n_frames, feat_dim * 2)
```

### Cross-validation

Formula-grouped to prevent same-molecule leakage:

```python
from deepmd.dpa_tools import cross_validate, train_test_split

systems = load_dataset("/data/root", label_key="energy")
train, valid, test = train_test_split(systems, group_by="formula", seed=42)

result = cross_validate(model, systems, label_key="energy", cv=5, group_by="formula")
# → {"aggregate": {"mae_mean": ..., "rmse_std": ...}, ...}
```

### Data tools

```python
convert("POSCAR", "output_dir", fmt="vasp/poscar", type_map=["Cu", "O"])
batch_convert("calcs/**/OUTCAR", "npy_root", fmt="vasp/outcar")
check_data("/data/system")   # → list[Issue]
attach_labels(system, head="bandgap", values=np.array([1.0, 2.0, 3.0]))
```

## CLI

All commands live under `dp dpa` with two-level nesting:

```
dp dpa
  extract-descriptors   extract pooled DPA descriptors to .npy
  fit                   train a model (any strategy)
  mft                   multi-task fine-tuning
  cv                    cross-validate frozen_sklearn baseline
  predict               predict with a frozen .pth bundle
  evaluate              evaluate a frozen .pth against stored labels
  data
    convert             structure file → deepmd/npy
    batch-convert       glob-based batch conversion
    validate            sanity-check deepmd/npy directories
    attach-labels       inject .npy labels into a system
```

`dp dpa --help` does not load torch.  The parser is pure argparse in
`deepmd/main.py`; the handler import happens lazily in
`deepmd/entrypoints/main.py` only when `dp dpa ...` is invoked.

```bash
dp dpa fit \
  --train-data /data/train \
  --pretrained /path/to/DPA-3.1-3M.pt \
  --strategy frozen_sklearn \
  --predictor rf \
  --target-key homo

dp dpa extract-descriptors \
  --data /data/sys1 /data/sys2 \
  --pretrained /path/to/DPA-3.1-3M.pt \
  --pooling mean+std \
  --output features.npy

dp dpa mft \
  --data /data/qm9 \
  --aux-data /data/spice2 \
  --pretrained /path/to/DPA-3.1-3M.pt \
  --property-name homo

dp dpa data convert --input POSCAR --output npy_dir --fmt vasp/poscar
dp dpa data validate --data /data/sys1 /data/sys2
```

## Installation

```bash
pip install deepmd-kit[dpa-tools]
```

The `dpa-tools` extra brings in `scikit-learn`.  `torch` and `dpdata` are
already provided by deepmd-kit's core dependencies.

## Internal architecture

```
deepmd/dpa_tools/
├── __init__.py           # public API, lazy imports (no torch at import time)
├── _backend.py            # single choke point for deepmd.pt.* calls
├── cli.py                 # dp dpa subcommand handlers
├── finetuner.py           # DPAFineTuner (training + descriptor extraction)
├── predictor.py           # DPAPredictor (read-only inference + uncertainty)
├── mft.py                 # MFTFineTuner (multi-task fine-tuning)
├── trainer.py             # DPATrainer (dp --pt train subprocess wrapper)
├── cv.py                  # cross-validation + data splitting
├── conditions.py          # scalar condition manager (T, P)
├── config/
│   └── manager.py         # MFT input.json generation
├── data/
│   ├── loader.py          # polymorphic data loading
│   ├── dataset.py         # label-filtered loading
│   ├── convert.py         # format conversion
│   ├── validate.py        # data sanity checks
│   ├── desc_cache.py      # two-tier descriptor cache
│   ├── type_map.py        # automatic type-map resolution
│   └── errors.py          # DPADataError
└── utils/
    ├── dotdict.py         # DotDict
    └── sklearn_heads.py   # sklearn regressor factory
```

Key design points:
- `_backend.py` is the **only** file that imports `deepmd.pt.*` — every call
  into deepmd internals goes through it
- `_DescriptorExtraction` encapsulates the fragile chain
  `wrapper.model["Default"]` → `set_eval_descriptor_hook` → `forward_common`
  → `eval_descriptor()`
- `dp --pt train/test/freeze` always runs as a subprocess, keeping
  dpa_tools decoupled from deepmd-kit's training entry points
- `dpdata.System` is the universal internal data format
