# Descriptor DPA-2 {{ pytorch_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, DP {{ dpmodel_icon }}
:::

The DPA-2 model implementation. See https://arxiv.org/abs/2312.15492 for more details.

Training example: `examples/water/dpa2/input_torch_medium.json`, see [README](../../examples/water/dpa2/README.md) for inputs in different levels.

## Data format

DPA-2 supports both the [standard data format](../data/system.md) and the [mixed type data format](../data/system.md#mixed-type).
