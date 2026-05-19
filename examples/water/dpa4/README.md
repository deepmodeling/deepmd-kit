# Input for DPA4 / SeZM: Smooth equivariant Zone-bridging Model (PyTorch)

This directory stores a minimal configuration for training DPA4 on the water
example dataset. `model.type: dpa4` and `descriptor.type: dpa4` are the
preferred DPA-series names; `SeZM` and `sezm` are equivalent compatibility
aliases for the same PyTorch implementation.

Run:

```bash
cd examples/water/dpa4
dp --pt train input.json
```
