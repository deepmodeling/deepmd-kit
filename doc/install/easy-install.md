# Easy install

There are various easy methods to install DeePMD-kit. Choose one that you prefer. If you want to build by yourself, jump to the next two sections.

After your easy installation, DeePMD-kit (`dp`) and LAMMPS (`lmp`) will be available to execute. You can try `dp -h` and `lmp -h` to see the help. `mpirun` is also available considering you may want to train models or run LAMMPS in parallel.

:::{note}
Note: The off-line packages and conda packages require the [GNU C Library](https://www.gnu.org/software/libc/) 2.17 or above. The GPU version requires [compatible NVIDIA driver](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility) to be installed in advance. It is possible to force conda to [override detection](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#overriding-detected-packages) when installation, but these requirements are still necessary during runtime.
You can refer to [DeepModeling conda FAQ](https://docs.deepmodeling.com/faq/conda.html) for more information.
:::

:::{note}
Python 3.9 or above is required for Python interface.
:::

- [Install off-line packages](#install-off-line-packages)
- [Install with conda](#install-with-conda)
- [Install with docker](#install-with-docker)
- [Install Python interface with pip](#install-python-interface-with-pip)

## Install off-line packages

Both CPU and GPU version offline packages are available on [the Releases page](https://github.com/deepmodeling/deepmd-kit/releases).

Some packages are split into two files due to the size limit of GitHub. One may merge them into one after downloading:

```bash
cat deepmd-kit-2.2.9-cuda118-Linux-x86_64.sh.0 deepmd-kit-2.2.9-cuda118-Linux-x86_64.sh.1 > deepmd-kit-2.2.9-cuda118-Linux-x86_64.sh
```

One may enable the environment using

```bash
conda activate /path/to/deepmd-kit
```

## Install with conda

DeePMD-kit is available with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section), [Miniconda](https://docs.conda.io/en/latest/miniconda.html), or [miniforge](https://conda-forge.org/download/) first.
You can refer to [DeepModeling conda FAQ](https://docs.deepmodeling.com/faq/conda.html) for how to setup a conda environment.

### conda-forge channel

DeePMD-kit is available on the [conda-forge](https://conda-forge.org/) channel:

```bash
conda create -n deepmd deepmd-kit lammps horovod -c conda-forge
```

The supported platforms include Linux x86-64, macOS x86-64, and macOS arm64.
Read [conda-forge FAQ](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch) to learn how to install CUDA-enabled packages.

### Official channel (deprecated)

::::{danger}
:::{deprecated} 3.0.0
The official channel has been deprecated since 3.0.0, due to the challenging work of building dependencies for [multiple backends](../backend.md).
Old packages will still be available at https://conda.deepmodeling.com.
Maintainers will build packages in the conda-forge organization together with other conda-forge members.
:::
::::

## Install with docker

A docker for installing the DeePMD-kit is available [here](https://github.com/deepmodeling/deepmd-kit/pkgs/container/deepmd-kit).

To pull the CPU version:

```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:2.2.8_cpu
```

To pull the GPU version:

```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:2.2.8_cuda12.0_gpu
```

## Install Python interface with pip

[Create a new environment](https://docs.deepmodeling.com/faq/conda.html#how-to-create-a-new-conda-pip-environment), and then execute the following command:

:::::::{tab-set}

::::::{tab-item} TensorFlow {{ tensorflow_icon }}

:::::{tab-set}

::::{tab-item} CUDA 12

```bash
pip install deepmd-kit[gpu,cu12]
```

`cu12` is required only when CUDA Toolkit and cuDNN were not installed.

::::

::::{tab-item} CUDA 11

```bash
pip install deepmd-kit-cu11[gpu,cu11]
```

::::

::::{tab-item} CPU

```bash
pip install deepmd-kit[cpu]
```

::::

:::::

::::::

::::::{tab-item} PyTorch {{ pytorch_icon }}

:::::{tab-set}

::::{tab-item} CUDA 12

```bash
pip install deepmd-kit[torch]
```

::::

::::{tab-item} CUDA 11.8

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install deepmd-kit-cu11
```

::::

::::{tab-item} CPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install deepmd-kit
```

::::

:::::

::::::

::::::{tab-item} JAX {{ jax_icon }}

:::::{tab-set}

::::{tab-item} CUDA 12

```bash
pip install deepmd-kit[jax] jax[cuda12]
```

::::

::::{tab-item} CPU

```bash
pip install deepmd-kit[jax]
```

::::

:::::

To generate a SavedModel and use [the LAMMPS module](../third-party/lammps-command.md) and [the i-PI driver](../third-party/ipi.md),
you need to install the TensorFlow.
Switch to the TensorFlow {{ tensorflow_icon }} tab for more information.

::::::

::::::{tab-item} Paddle {{ paddle_icon }}

:::::{tab-set}

::::{tab-item} CUDA 12.6

```bash
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install deepmd-kit
```

::::

::::{tab-item} CUDA 11.8

```bash
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install deepmd-kit
```

::::

::::{tab-item} CPU

```bash
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install deepmd-kit
```

::::

:::::

::::::

:::::::

The supported platform includes Linux x86-64 and aarch64 with GNU C Library 2.28 or above, macOS x86-64 and arm64, and Windows x86-64.

:::{Warning}
If your platform is not supported, or you want to build against the installed backends, or you want to enable ROCM support, please [build from source](install-from-source.md).
:::

[The LAMMPS module](../third-party/lammps-command.md) and [the i-PI driver](../third-party/ipi.md) are provided on Linux and macOS for the TensorFlow, PyTorch, and JAX backend. It requires both TensorFlow and PyTorch. To install LAMMPS and/or i-PI, add `lmp` and/or `ipi` to extras:

```bash
pip install deepmd-kit[gpu,cu12,lmp,ipi]
```

MPICH is required for parallel running.
