# DPA Tools Quickstart Demo

Fit a frozen DPA-3.1 descriptor + Ridge regressor on the QM9 GDB9
HOMO-LUMO gap in **under 5 minutes on CPU** with just 50 molecules.

Pre-processed data for 50 QM9 molecules (mol_id 1–50, HOMO-LUMO gap)
is included in `demo/data/`.  To regenerate from raw GDB9, see
`scripts/prepare_data.py`.

## Step 1 — Prerequisites

- Python 3.10+ with `dpdata`, `numpy`, and `deepmd-kit` installed
- **DPA-3.1-3M pretrained checkpoint** — download from the DPA-3.1
  release page or from DeepModeling.  Set the path via the
  `DPA_MODEL_PATH` environment variable or pass it with `--model`.

```bash
# One-time setup
export DPA_MODEL_PATH=/path/to/DPA-3.1-3M.pt
```

## Step 2 — Fit & evaluate

Trains a frozen DPA descriptor + sklearn `Ridge` regressor and evaluates
on the held-out test set.

```bash
python fit_evaluate.py --model $DPA_MODEL_PATH
```

Or with `dp dpa fit` (same underlying API):

```bash
dp dpa fit --pretrained $DPA_MODEL_PATH --train-data data/train \
    --valid-data data/test --target-key gap \
    --model-branch Domains_Drug --predictor linear --pooling mean
```

## Expected output

```
Fitting …
Evaluating …

==================================================
MAE  : ~0.2–0.4 eV
R²   : ~0.85–0.95
RMSE : ~0.3–0.5 eV
N    : 10
==================================================
Frozen model → frozen_model.pth
```

(Results may vary slightly depending on the DPA-3.1-3M checkpoint version.)

---

This demo uses 50 molecules and runs on CPU in under 5 minutes.
