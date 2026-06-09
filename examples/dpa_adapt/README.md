# ADAPT example

This directory contains a small ready-to-run example for `dpa_adapt`.
The example uses 50 pre-processed QM9 molecules to fine-tune and evaluate a
DPA-based HOMO–LUMO gap predictor.

The processed data is already included, so you can run the demo directly.

## Directory layout

```text
examples/dpa_adapt/
├── data/                    # ready-to-use processed data
│   ├── train/               # 40 training systems in deepmd/npy format
│   ├── test/                # 10 test systems in deepmd/npy format
│   ├── train_labels.npy
│   └── test_labels.npy
├── scripts/
│   ├── run_evaluate.py      # run the included training/evaluation demo
│   └── prepare_data.py      # regenerate data/ from raw GDB9 data
└── README.md
```

## Run the example

From this directory, run:

```bash
python scripts/run_evaluate.py
```

The script uses the included `data/train/` and `data/test/` systems. It trains a
small `frozen_sklearn` model and prints evaluation metrics on the test set.

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
