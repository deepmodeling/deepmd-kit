# Descriptor DPA3 {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

DPA3 is an advanced interatomic potential leveraging the message passing architecture.
Designed as a large atomic model (LAM), DPA3 is tailored to integrate and simultaneously train on datasets from various disciplines,
encompassing diverse chemical and materials systems across different research domains.
Its model design ensures exceptional fitting accuracy and robust generalization both within and beyond the training domain.
Furthermore, DPA3 maintains energy conservation and respects the physical symmetries of the potential energy surface,
making it a dependable tool for a wide range of scientific applications.

Reference: [DPA3 paper](https://arxiv.org/abs/2506.01686).

Training example: `examples/water/dpa3/input_torch.json`.

## Hyperparameter tests

We systematically conducted DPA3 training on six representative DFT datasets (available at [AIS-Square](https://www.aissquare.com/datasets/detail?pageType=datasets&name=DPA3_hyperparameter_search&id=316)):
metallic systems (`Alloy`, `AlMgCu`, `W`), covalent material (`Boron`), molecular system (`Drug`), and liquid water (`Water`).
Under consistent training conditions (0.5M training steps, batch_size "auto:128"),
we rigorously evaluated the impacts of some critical hyperparameters on validation accuracy.

The comparative analysis focused on average RMSEs (Root Mean Square Error) for both energy, force and virial predictions across all six systems,
with results tabulated below to guide scenario-specific hyperparameter selection:

| Model            | comment         | nlayers | n_dim   | e_dim  | a_dim | e_sel   | a_sel  | start_lr | stop_lr  | loss prefactors           | rmse_e (meV/atom) | rmse_f (meV/Å) | rmse_v (meV/atom) | Training wall time (h) |
| ---------------- | --------------- | ------- | ------- | ------ | ----- | ------- | ------ | -------- | -------- | ------------------------- | ----------------- | -------------- | ----------------- | ---------------------- |
| DPA3-L3          | Default         | 3       | 256     | 128    | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.74              | 85.4           | 43.1              | 9.8                    |
|                  | Small dimension | 3       | **128** | **64** | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 6.99              | 93.6           | 46.7              | 8.0                    |
|                  | Large sel       | 3       | 256     | 128    | 32    | **154** | **48** | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.70              | 83.7           | 43.4              | 14.1                   |
| DPA3-L6          | Default         | 6       | 256     | 128    | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 4.85              | 79.9           | 39.7              | 19.2                   |
|                  | Small dimension | 6       | **128** | **64** | 32    | 120     | 30     | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 5.11              | 77.7           | 41.2              | 14.1                   |
|                  | Large sel       | 6       | 256     | 128    | 32    | **154** | **48** | 1e-3     | 3e-5     | 0.2\|20, 100\|60, 0.02\|1 | 4.76              | 78.4           | 40.2              | 31.8                   |
| DPA2-L6 (medium) | Default         | 6       | -       | -      | -     | -       | -      | 1e-3     | 3.51e-08 | 0.02\|1, 1000\|1, 0.02\|1 | 12.12             | 109.3          | 83.1              | 12.2                   |

The loss prefactors (0.2|20, 100|60, 0.02|1) correspond to (`start_pref_e`|`limit_pref_e`, `start_pref_f`|`limit_pref_f`, `start_pref_v`|`limit_pref_v`) respectively.
Virial RMSEs were averaged exclusively for systems containing virial labels (`Alloy`, `AlMgCu`, `W`, and `Boron`).

Note that we set `float32` in all DPA3 models, while `float64` in other models by default.

## Requirements of installation from source code {{ pytorch_icon }}

To run the DPA3 model on LAMMPS via source code installation
(users can skip this step if using [easy installation](../install/easy-install.md)),
the custom OP library for Python interface integration must be compiled and linked
during the [model freezing process](../freeze/freeze.md).

The customized OP library for the Python interface can be installed by setting environment variable {envvar}`DP_ENABLE_PYTORCH` to `1` during installation.

If one runs LAMMPS with MPI, the customized OP library for the C++ interface should be compiled against the same MPI library as the runtime MPI.
If one runs LAMMPS with MPI and CUDA devices, it is recommended to compile the customized OP library for the C++ interface with a [CUDA-Aware MPI](https://developer.nvidia.com/mpi-solutions-gpus) library and CUDA,
otherwise the communication between GPU cards falls back to the slower CPU implementation.

## Limitations of the JAX backend with LAMMPS {{ jax_icon }}

When using the JAX backend, 2 or more MPI ranks are not supported. One must set `map` to `yes` using the [`atom_modify`](https://docs.lammps.org/atom_modify.html) command.

```lammps
atom_modify map yes
```

See the example `examples/water/lmp/jax_dpa.lammps`.

## Data format

DPA3 supports both the [standard data format](../data/system.md) and the [mixed type data format](../data/system.md#mixed-type).

## Type embedding

Type embedding is within this descriptor with the same dimension as the node embedding: {ref}`n_dim <model[standard]/descriptor[dpa3]/repflow/n_dim>` argument.

## Model compression

Model compression is not supported in this descriptor.
