# Change the model output bias for trained model {{ pytorch_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}
:::

The output bias of a trained model typically originates from the statistical results of the training dataset.

There are several scenarios where one might want to adjust the output bias after the model is trained,
such as zero-shot testing (similar to the procedure before the first step in fine-tuning)
or manually setting the output bias.

The `dp --pt change-bias` command supports the following methods for adjusting the bias:

::::{tab-set}

:::{tab-item} Changing bias using provided systems for trained `.pt`/`.pth` models:

```sh
dp --pt change-bias model.pt -s data_dir -o model_updated.pt
```

For multitask models, where `--model-branch` must be specified:

```sh
dp --pt change-bias multi_model.pt -s data_dir -o model_updated.pt --model-branch model_1
```

:::

:::{tab-item} Changing bias using user input for **energy model**:

```sh
dp --pt change-bias model.pt -b -92.523 -187.66 -o model_updated.pt
```

Here, `-b` specifies user-defined energy bias for each type, separated by space,
in an order consistent with the `type_map` in the model.

:::

::::
