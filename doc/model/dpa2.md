# Descriptor DPA-2 {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

The DPA-2 model implementation. See https://arxiv.org/abs/2312.15492 for more details.

Training example: `examples/water/dpa2/input_torch_medium.json`, see [README](../../examples/water/dpa2/README.md) for inputs in different levels.

## Requirements of installation {{ pytorch_icon }}

If one wants to run the DPA-2 model on LAMMPS, the customized OP library for the Python interface must be installed when [freezing the model](../freeze/freeze.md).

The customized OP library for the Python interface can be installed by setting environment variable {envvar}`DP_ENABLE_PYTORCH` to `1` during installation.

If one runs LAMMPS with MPI, the customized OP library for the C++ interface should be compiled against the same MPI library as the runtime MPI.
If one runs LAMMPS with MPI and CUDA devices, it is recommended to compile the customized OP library for the C++ interface with a [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library and CUDA,
otherwise the communication between GPU cards falls back to the slower CPU implementation.

## Limiations of the JAX backend with LAMMPS {{ jax_icon }}

When using the JAX backend, 2 or more MPI ranks are not supported. One must set `map` to `yes` using the [`atom_modify`](https://docs.lammps.org/atom_modify.html) command.

```lammps
atom_modify map yes
```

See the example `examples/water/lmp/jax_dpa2.lammps`.

## Data format

DPA-2 supports both the [standard data format](../data/system.md) and the [mixed type data format](../data/system.md#mixed-type).
