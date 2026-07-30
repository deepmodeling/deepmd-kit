# Group Property Example

This example demonstrates the `group_property` architecture. One supervised
sample is a group of component structures rather than a single frame. DeePMD
embeds each component, masks padded atoms with `pool_mask.npy`, weights
components with `weight.npy`, and predicts one group-level `target_property`.

The bundled dataset is intentionally tiny: two training groups and one
validation group generated from three real rows of a public component-based
molecular dataset.

## Directory layout

```text
examples/group_property/
|-- data/
|   |-- grouped_assembly_subset.csv
|   |-- train/                 # 2 grouped training systems
|   |-- valid/                 # 1 grouped validation system
|   |-- train_systems.txt
|   `-- valid_systems.txt
|-- scripts/
|   |-- prepare_data.py        # regenerate data/train and data/valid
|   `-- run_dpa_adapt_api.py   # dpa-adapt API route
|-- train/
|   `-- input_torch.json       # DeePMD group_property training input
`-- README.md
```

## Run

To run the DeePMD training example:

```bash
cd train
dp --pt train input_torch.json
```

To run the dpa-adapt API route:

```bash
python scripts/run_dpa_adapt_api.py
```

To regenerate the grouped DeePMD data from the CSV subset:

```bash
python scripts/prepare_data.py
```

Set `DPA_GROUP_PROPERTY_CSV=/path/to/dataset.csv` and optionally
`DPA_GROUP_PROPERTY_ROWS` / `DPA_GROUP_PROPERTY_VALID` to use a larger compatible
CSV split.
