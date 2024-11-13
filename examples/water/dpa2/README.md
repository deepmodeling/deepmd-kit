## Inputs for DPA-2 model

This directory contains the input files for training the DPA-2 model (currently supporting PyTorch backend only). Depending on your precision/efficiency requirements, we provide three different levels of model complexity:

- `input_torch_small.json`: Our smallest DPA-2 model, optimized for speed.
- `input_torch_medium.json` (Recommended): Our well-performing DPA-2 model, balancing efficiency and precision. This is a good starting point for most users.
- `input_torch_large.json`: Our most complex model with the highest precision, suitable for very intricate data structures.

For detailed differences in their configurations, please refer to the table below:

| Input                     | Repformer layers | Three-body embedding in Repinit | Pair-wise attention in Repformer | Tuned sub-structures in [#4089](https://github.com/deepmodeling/deepmd-kit/pull/4089) | Description                                                                  |
| ------------------------- | ---------------- | ------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `input_torch_small.json`  | 3                | ✓                               | ✗                                | ✓                                                                                     | Smallest DPA-2 model, optimized for speed.                                   |
| `input_torch_medium.json` | 6                | ✓                               | ✓                                | ✓                                                                                     | Recommended well-performing DPA-2 model, balancing efficiency and precision. |
| `input_torch_large.json`  | 12               | ✓                               | ✓                                | ✓                                                                                     | Most complex model with the highest precision.                               |

`input_torch_compressible.json` is derived from `input_torch_small.json` and makes the `repinit` part compressible.
