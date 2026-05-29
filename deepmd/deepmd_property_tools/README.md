# DeePMD Property Tools

`deepmd_property_tools` is a Uni-Mol-tools-like interface for DeePMD-kit molecular property training and prediction.

It wraps DeePMD-kit data generation, DPA3 property training, fine-tuning, freezing, and `DeepProperty` inference behind a small API:

## Installation

Install the package from this directory:

```bash
pip install .
```

For local development with tests:

```bash
pip install ".[test]"
python -m pytest tests -v
```

```python
from deepmd_property_tools import PropertyTrain, PropertyPredict

clf = PropertyTrain(
    task="regression",
    property_name="Property",
    property_col="Property",
    save_path="./exp",
    finetune="DPA-3.2-5M",
)
clf.fit({"dataset": "DATA/dataset_demo.csv", "mol_dir": "DATA/mol_convert"})

predictor = PropertyPredict(load_model="./exp/model.ckpt-10.pt")
y_pred = predictor.predict(
    {"dataset": "DATA/dataset_demo.csv", "mol_dir": "DATA/mol_convert"},
    save_path="./pred",
)
```

## Data format

For CSV + MOL workflows, row `i` in the CSV maps to `mol_convert/id{i}.mol` by default. The selected property column is converted to a DeePMD property fitting target.

```text
DATA/
  dataset_demo.csv
  mol_convert/
    id0.mol
    id1.mol
```

Direct coordinate data is also supported:

```python
clf.fit(
    {
        "atoms": [["C", "H", "H", "H", "H"], ["O", "H", "H"]],
        "coordinates": [coords0, coords1],
        "target": [0.1, 0.2],
    }
)
```

## Command Line

The package exposes an entry point after installation:

```bash
deepmd-property-tools --help
```

Train from CSV + MOL inputs:

```bash
deepmd-property-tools train \
    --dataset DATA/dataset_demo.csv \
    --mol-dir DATA/mol_convert \
    --save-path exp_property
```

Predict with a checkpoint file or an experiment directory:

```bash
deepmd-property-tools predict \
    --model exp_property \
    --dataset DATA/dataset_demo.csv \
    --mol-dir DATA/mol_convert \
    --save-path pred_property
```

## Notes

This package does not reimplement DeePMD models. It is a convenience layer that calls DeePMD-kit training and inference APIs internally.
