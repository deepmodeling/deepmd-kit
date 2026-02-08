# How to control the parallelism of a job?

DeePMD-kit has three levels of parallelism.
To get the best performance, one should control the number of threads used by DeePMD-kit.
One should make sure the product of the parallel numbers is less than or equal to the number of cores available.

## MPI or multiprocessing (optional)

Parallelism for MPI is optional and used for multiple nodes, multiple GPU cards, or sometimes multiple CPU cores.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

To enable MPI support for training in the TensorFlow interface, one should [install horovod](../install/install-from-source.md#install-horovod-and-mpi4py) in advance.

:::
:::{tab-item} PyTorch {{ pytorch_icon }}

Multiprocessing support for training in the PyTorch backend is implemented with [torchrun](https://pytorch.org/docs/stable/elastic/run.html).

:::
::::

Note that the parallelism mode is data parallelism, so it is not expected to see the training time per batch decreases.
See [Parallel training](../train/parallel-training.md) for details.

MPI support for inference is not directly supported by DeePMD-kit, but indirectly supported by the third-party software. For example, [LAMMPS enables running simulations in parallel](https://docs.lammps.org/Developer_parallel.html) using the MPI parallel communication standard with distributed data. That software has to build against MPI.

Set the number of processes with:

```bash
mpirun -np $num_nodes dp
```

Note that `mpirun` here should be the same as the MPI used to build software. For example, one can use `mpirun --version` and `lmp -h` to see if `mpirun` and LAMMPS has the same MPI version.

Sometimes, `$num_nodes` and the nodes information can be directly given by the HPC scheduler system, if the MPI used here is the same as the MPI used to build the scheduler system. Otherwise, one have to manually assign these information.

Each process can use at most one GPU card.

## Parallelism between independent operators

For CPU devices, TensorFlow and PyTorch use multiple streams to run independent operators (OP).

```bash
export DP_INTER_OP_PARALLELISM_THREADS=3
```

However, for GPU devices, TensorFlow and PyTorch use only one compute stream and multiple copy streams.
Note that some of DeePMD-kit OPs do not have GPU support, so it is still encouraged to set environment variables even if one has a GPU.

## Parallelism within individual operators

For CPU devices, {envvar}`DP_INTRA_OP_PARALLELISM_THREADS` controls parallelism within TensorFlow (when TensorFlow is built against Eigen) and PyTorch native OPs.

```bash
export DP_INTRA_OP_PARALLELISM_THREADS=2
```

`OMP_NUM_THREADS` is the number of threads for OpenMP parallelism.
It controls parallelism within TensorFlow (when TensorFlow is built upon Intel OneDNN) and PyTorch (when PyTorch is built upon OpenMP) native OPs and DeePMD-kit custom CPU OPs.
It may also control parallelism for NumPy when NumPy is built against OpenMP, so one who uses GPUs for training should also care this environmental variable.

```bash
export OMP_NUM_THREADS=2
```

There are several other environment variables for OpenMP, such as `KMP_BLOCKTIME`.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

See [Intel documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html) for detailed information.

:::
:::{tab-item} PyTorch {{ pytorch_icon }}

See [PyTorch documentation](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) for detailed information.

:::
::::

## Tune the performance

There is no one general parallel configuration that works for all situations, so you are encouraged to tune parallel configurations yourself after empirical testing.

Here are some empirical examples.
If you wish to use 3 cores of 2 CPUs on one node, you may set the environment variables and run DeePMD-kit as follows:

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```bash
export OMP_NUM_THREADS=3
export DP_INTRA_OP_PARALLELISM_THREADS=3
export DP_INTER_OP_PARALLELISM_THREADS=2
dp --tf train input.json
```

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```bash
export OMP_NUM_THREADS=3
export DP_INTRA_OP_PARALLELISM_THREADS=3
export DP_INTER_OP_PARALLELISM_THREADS=2
dp --pt train input.json
```

:::

::::

For a node with 128 cores, it is recommended to start with the following variables:

```bash
export OMP_NUM_THREADS=16
export DP_INTRA_OP_PARALLELISM_THREADS=16
export DP_INTER_OP_PARALLELISM_THREADS=8
```

Again, in general, one should make sure the product of the parallel numbers is less than or equal to the number of cores available.
In the above case, $16 \times 8 = 128$, so threads will not compete with each other.
