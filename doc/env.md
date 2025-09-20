# Runtime environment variables

:::{note}
For build-time environment variables, see [Install from source code](./install/install-from-source.md).
:::

## All interfaces

:::{envvar} DP_INTER_OP_PARALLELISM_THREADS

**Alias**: `TF_INTER_OP_PARALLELISM_THREADS`
**Default**: `0`

Control parallelism within TensorFlow (when TensorFlow is built against Eigen) and PyTorch native OPs for CPU devices.
See [How to control the parallelism of a job](./troubleshooting/howtoset_num_nodes.md) for details.
:::

:::{envvar} DP_INTRA_OP_PARALLELISM_THREADS

**Alias**: `TF_INTRA_OP_PARALLELISM_THREADS`\*\*
**Default**: `0`

Control parallelism within TensorFlow (when TensorFlow is built against Eigen) and PyTorch native OPs.
See [How to control the parallelism of a job](./troubleshooting/howtoset_num_nodes.md) for details.
:::

## Environment variables of dependencies

- If OpenMP is used, [OpenMP environment variables](https://www.openmp.org/spec-html/5.0/openmpch6.html) can be used to control OpenMP threads, such as [`OMP_NUM_THREADS`](https://www.openmp.org/spec-html/5.0/openmpse50.html#x289-20540006.2).
- If CUDA is used, [CUDA environment variables](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables) can be used to control CUDA devices, such as `CUDA_VISIBLE_DEVICES`.
- If ROCm is used, [ROCm environment variables](https://rocm.docs.amd.com/en/latest/conceptual/gpu-isolation.html#environment-variables) can be used to control ROCm devices.
- {{ tensorflow_icon }} If TensorFlow is used, TensorFlow environment variables can be used.
- {{ pytorch_icon }} If PyTorch is used, [PyTorch environment variables](https://pytorch.org/docs/stable/torch_environment_variables.html) can be used.
- {{ jax_icon }} [`JAX_PLATFORMS`](https://jax.readthedocs.io/en/latest/faq.html#controlling-data-and-computation-placement-on-devices) and [`XLA_FLAGS`](https://jax.readthedocs.io/en/latest/gpu_performance_tips.html#xla-performance-flags) are commonly used.

## Python interface only

:::{envvar} DP_INTERFACE_PREC

**Choices**: `high`, `low`; **Default**: `high`

Control high (double) or low (float) precision of training.
:::

:::{envvar} DP_AUTO_PARALLELIZATION

**Choices**: `0`, `1`; **Default**: `0`

{{ tensorflow_icon }} Enable auto parallelization for CPU operators.
:::

:::{envvar} DP_JIT

**Choices**: `0`, `1`; **Default**: `0`

{{ tensorflow_icon }} Enable JIT. Note that this option may either improve or decrease the performance. Requires TensorFlow to support JIT.
:::

:::{envvar} DP_INFER_BATCH_SIZE

**Default**: `1024` on CPUs and as maximum as possible until out-of-memory on GPUs

Inference batch size, calculated by multiplying the number of frames with the number of atoms.
:::

:::{envvar} DP_BACKEND

**Default**: `tensorflow`

Default backend.
:::

:::{envvar} NUM_WORKERS

**Default**: 4 or the number of cores (whichever is smaller)

{{ pytorch_icon }} Number of subprocesses to use for data loading in the PyTorch backend.
See [PyTorch documentation](https://pytorch.org/docs/stable/data.html) for details.

:::

## C++ interface only

These environment variables also apply to third-party programs using the C++ interface, such as [LAMMPS](./third-party/lammps-command.md).

:::{envvar} DP_PLUGIN_PATH

**Type**: List of paths, split by `:` on Unix and `;` on Windows

List of customized OP plugin libraries to load, such as `/path/to/plugin1.so:/path/to/plugin2.so` on Linux and `/path/to/plugin1.dll;/path/to/plugin2.dll` on Windows.

:::
