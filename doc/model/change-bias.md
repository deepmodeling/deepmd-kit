# Change the model output bias for trained model {{ tensorflow_icon }} {{ pytorch_icon }}

> [!NOTE]
> **Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}

The output bias of a trained model typically originates from the statistical results of the training dataset.

There are several scenarios where one might want to adjust the output bias after the model is trained,
such as zero-shot testing (similar to the procedure before the first step in fine-tuning)
or manually setting the output bias.

## The two statistic modes, precisely

The model energy decomposes as `E = E_model + E_bias`, where `E_model` is
whatever the model computes (a learned network, an analytical term such as
ZBL bridging, or a `linear_ener` combination of models) and `E_bias` is the
per-type output bias.

- **`set` (`set-by-statistic`)** assigns `E_bias` directly: either the
  user-given values (`-b`), or the per-type least-squares statistic of the
  **raw data labels**. It is independent of `E_model` by definition — it
  ignores a trained network, and it equally ignores an analytical
  contribution such as the ZBL term of a bridged model. The result is
  reproducible and idempotent for a given dataset, but it contains **no
  compensation for `E_model`**: after `set`, predictions on the calibration
  data are offset by the data mean of `E_model`.
- **`change` (`change-by-statistic`)** assigns `E_bias` from the residual:
  the per-type statistic of the labels **minus the complete model
  prediction** (including any analytical bridging term), added to the
  existing bias. Use this mode for a self-consistent calibration of a
  trained (or bridged) model.

For a bridged model — or any model whose `E_model` is significantly nonzero
on the calibration data — `set` therefore yields a bias that double-counts
nothing and compensates nothing; if you want the model predictions to match
the calibration labels, use `change`.

The `dp change-bias` command supports the following methods for adjusting the bias:

::::{tab-set}

:::{tab-item} TensorFlow Backend {{ tensorflow_icon }}

**Changing bias using provided systems for trained checkpoint:**

```sh
dp --tf change-bias model.ckpt -s data_dir -o model_updated.pb
```

**Changing bias using user input for energy model:**

```sh
dp --tf change-bias model.ckpt -b -92.523 -187.66 -o model_updated.pb
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
