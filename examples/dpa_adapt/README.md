# ADAPT example

This directory contains a small ready-to-run example for `dpa_adapt`.
The example uses 50 pre-processed QM9 molecules to fine-tune and evaluate a
DPA-based HOMO–LUMO gap predictor.

The processed data is already included, so you can run the demo directly.

## Directory layout

```text
examples/dpa_adapt/
├── data/                         # ready-to-use processed data
│   ├── train/                    # 40 training systems in deepmd/npy format
│   ├── test/                     # 10 test systems in deepmd/npy format
│   ├── train_labels.npy
│   └── test_labels.npy
├── scripts/
│   ├── run_evaluate_sklearn.py   # frozen_sklearn demo: DPA-3.1-3M + Ridge
│   ├── run_evaluate_finetune.py  # frozen_head demo: DPA-3.1-3M fine-tuning
│   └── prepare_data.py           # regenerate data/ from raw GDB9 data
└── README.md
```

## Run the example

Two evaluation scripts are provided, demonstrating different adaptation strategies.

From this directory, run either (or both):

```bash
# frozen_sklearn strategy — extract DPA features, fit a Ridge regressor
python scripts/run_evaluate_sklearn.py

# frozen_head strategy — fine-tune the prediction head with gradient steps
python scripts/run_evaluate_finetune.py
```

### `run_evaluate_sklearn.py`

Uses the `frozen_sklearn` strategy with the `Domains_Drug` model branch.
DPA-3.1-3M features are extracted from the training systems and a Ridge (`linear`)
regressor is fitted on top. Prints MAE, RMSE, and R² on the test set.

### `run_evaluate_finetune.py`

Uses the `frozen_head` strategy. A fresh prediction head is trained on top of
frozen DPA-3.1-3M features with `learning_rate=1e-3`, `batch_size=128`,
`max_steps=5`. Prints predictions and evaluation metrics (MAE, RMSE, R²) on the
test set.

## About the included data

The `data/` directory already contains the processed example dataset. Each system
is stored in `deepmd/npy` format and each `set.000/` directory contains a
`gap.npy` label file. The label key used by the example is `gap`.

In normal use, you do not need to run any data preparation step.

## Regenerating the data

`scripts/prepare_data.py` is provided only for reproducibility. It rebuilds the
included `data/` directory from raw GDB9/QM9 files.

Run it only if you want to recreate the processed data:

```bash
python scripts/prepare_data.py
```

The script downloads `gdb9.tar.gz`, extracts the raw SDF and CSV files into
`raw/`, converts the first 50 molecules to `deepmd/npy`, and writes HOMO–LUMO gap
labels as `gap.npy`.
