# Change the model output bias for trained model {{ tensorflow_icon }} {{ pytorch_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}
:::

The output bias of a trained model typically originates from the statistical results of the training dataset.

There are several scenarios where one might want to adjust the output bias after the model is trained,
such as zero-shot testing (similar to the procedure before the first step in fine-tuning)
or manually setting the output bias.

The `dp change-bias` command supports the following methods for adjusting the bias:

::::{tab-set}

:::{tab-item} TensorFlow Backend {{ tensorflow_icon }}

**Changing bias using provided systems for trained checkpoint models:**

```sh
dp --tf change-bias model.ckpt -s data_dir -o model_updated.pb
```

**Changing bias using provided systems for trained frozen models:**

```sh
dp --tf change-bias model.pb -s data_dir -o model_updated.pb
```

**Changing bias using user input for energy model:**

```sh
dp --tf change-bias model.ckpt -b -92.523 -187.66 -o model_updated.pb
```

For multitask models, where `--model-branch` must be specified:

```sh
dp --tf change-bias model.ckpt -s data_dir -o model_updated.pb --model-branch model_1
```

:::

:::{tab-item} PyTorch Backend {{ pytorch_icon }}

**Changing bias using provided systems for trained `.pt`/`.pth` models:**

```sh
dp --pt change-bias model.pt -s data_dir -o model_updated.pt
```

**Changing bias using user input for energy model:**

```sh
dp --pt change-bias model.pt -b -92.523 -187.66 -o model_updated.pt
```

For multitask models, where `--model-branch` must be specified:

```sh
dp --pt change-bias multi_model.pt -s data_dir -o model_updated.pt --model-branch model_1
```

:::

::::

## Common Parameters

Both backends support the same command-line options:

- `-s/--system`: Specify data directory for automatic bias calculation
- `-b/--bias-value`: Provide user-defined bias values (e.g., `-b -92.523 -187.66`)
- `-n/--numb-batch`: Number of frames to use for bias calculation (0 = all data)
- `-m/--mode`: Bias calculation mode (`change` or `set`)
- `-o/--output`: Output model file path
- `--model-branch`: Model branch for multitask models

The `-b/--bias-value` option specifies user-defined energy bias for each type, separated by space, in an order consistent with the `type_map` in the model.

## Backend-Specific Details

### TensorFlow {{ tensorflow_icon }}

- **Supported input formats**:
  - Checkpoint files (`.ckpt`, `.meta`, `.data`, `.index`)
  - Frozen models (`.pb`)
- **Output format**: Frozen model (`.pb`)
- **Special features**:
  - Creates updated checkpoint files in a separate directory for continued training
  - Variables are properly restored from checkpoint before bias modification

### PyTorch {{ pytorch_icon }}

- **Supported input formats**:
  - Saved models (`.pt`)
  - TorchScript models (`.pth`)
- **Output format**: Same as input format (`.pt` or `.pth`)
- **Special features**:
  - Direct model state modification
  - Preserves all model metadata
