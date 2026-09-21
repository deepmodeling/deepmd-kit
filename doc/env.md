# Runtime environment variables

> [!NOTE]
> For build-time environment variables, see [Install from source code](./install/install-from-source.md).

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

**Default**: `1024` on CPUs; automatically sized on GPUs

Inference batch size, calculated by multiplying the number of frames with the number of atoms.
Setting this variable disables automatic growth; out-of-memory errors can still
reduce the batch size.

With the native PyTorch CUDA allocator, automatic multi-frame evaluation starts
with one calibration frame and limits growth using the observed workspace and
available device memory, leaving a 10% margin. This policy applies to both
`dp test` and full validation in the PyTorch and PyTorch Exportable backends.
Other GPU allocators and backends grow batches until an out-of-memory error.
:::

:::{envvar} DP_HESSIAN_HVP_BATCH

**Default**: automatically sized on CUDA devices; `1` elsewhere

{{ pytorch_icon }} Number of Hessian rows evaluated per second-order backward
pass when a PyTorch Exportable model computes a Hessian on the neighbour-graph
route. The Hessian is assembled from Hessian-vector products,
and batching them over replicated frames trades peak memory for far fewer
kernel launches. `1` evaluates one row at a time, which is how the Hessian was
computed before batching existed.

Automatic sizing runs one Hessian-vector product to measure what a replica
costs, then takes the batch the free memory affords, capped at 8. Peak memory
is linear in the batch while the speedup is not: batching recovers
kernel-launch overhead, which stops mattering once a single Hessian-vector
product already saturates the device. On one H20 with DPA-4, batching is worth
about 5x on a system of 72 neighbour pairs but only 1.2x at 5832, where it also
costs six times the memory -- so a large fixed batch lowers the system size that
fits and buys little on the systems that need it.

Setting this variable disables automatic sizing and uses the value given;
out-of-memory errors can still reduce the batch, halving it and warning which
batch was used. The result does not depend on the batch: it changes how the
Hessian is computed, not what it is.
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

:::{envvar} DP_LMDB_NUM_WORKERS

**Type**: non-negative integer

**Default**: automatically selected from the process CPU affinity and the
number of local training ranks, with limits of 32 workers per rank and
approximately 64 workers per node

Number of worker processes used to read, decode, and assemble one LMDB batch
in the PyTorch and PyTorch Exportable backends. Each process owns an
independent read-only LMDB transaction. The next batch is prefetched while the
current batch is consumed, and at most one batch is prefetched. Batches with
fewer frames than workers are decoded synchronously because process startup
and IPC cost more than their small decode workload.

Set this variable to `0` or `1` to use synchronous decoding. For multi-GPU
training, this value applies to each rank. Independent jobs do not share their
worker pools, so reduce it when the aggregate reader count across concurrent
jobs would overload the storage service. The LMDB dataset must remain immutable
while any job is reading it because readers intentionally disable LMDB locking.
:::

## C++ interface only

These environment variables also apply to third-party programs using the C++ interface, such as [LAMMPS](./third-party/lammps-command.md).

:::{envvar} DP_PLUGIN_PATH

**Type**: List of paths, split by `:` on Unix and `;` on Windows

List of customized OP plugin libraries to load, such as `/path/to/plugin1.so:/path/to/plugin2.so` on Linux and `/path/to/plugin1.dll;/path/to/plugin2.dll` on Windows.
:::

:::{envvar} DP_BACKEND_PLUGIN_PATH

**Type**: List of directories, split by `:` on Unix and `;` on Windows

List of directories used to search for C/C++ backend plugin libraries before the directory that contains `libdeepmd_cc`.
This controls backend implementation plugins, such as `libdeepmd_backend_tf.so` and `libdeepmd_backend_pt.so`, and is separate from {envvar}`DP_PLUGIN_PATH`, which loads customized OP plugins.
:::

:::{envvar} DP_PROFILER

{{ pytorch_icon }} Enable the built-in PyTorch Kineto profiler for the PyTorch C++ (inference) backend.

**Type**: string (output file stem)

**Default**: unset (disabled)

When set to a non-empty value, profiling is enabled for the lifetime of the loaded PyTorch model (e.g. during LAMMPS runs). A JSON trace file is created on finish. The final file name is constructed as:

- `<ENV_VALUE>_gpu<ID>.json` if running on GPU
- `<ENV_VALUE>.json` if running on CPU

The trace can be examined with [Chrome trace viewer](https://ui.perfetto.dev/) (alternatively chrome://tracing). It includes:

- CPU operator activities
- CUDA activities (if available)

Example:

```bash
export DP_PROFILER=result
mpirun -np 4 lmp -in in.lammps
# Produces result_gpuX.json, where X is the GPU id used by each MPI rank.
```

Tips:

- Large runs can generate sizable JSON files; consider limiting numbers of MD steps, like 20.
- Currently this feature only supports single process, or multi-process runs where each process uses a distinct GPU on the same node.
:::
