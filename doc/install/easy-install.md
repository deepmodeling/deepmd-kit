# Easy install

There are various easy methods to install DeePMD-kit. Choose one that you prefer. If you want to build by yourself, jump to the next two sections.

After your easy installation, DeePMD-kit (`dp`) and LAMMPS (`lmp`) will be available to execute. You can try `dp -h` and `lmp -h` to see the help. `mpirun` is also available considering you may want to train models or run LAMMPS in parallel.

:::{note}
Note: The off-line packages and conda packages require the [GNU C Library](https://www.gnu.org/software/libc/) 2.17 or above. The GPU version requires [compatible NVIDIA driver](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility) to be installed in advance. It is possible to force conda to [override detection](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#overriding-detected-packages) when installation, but these requirements are still necessary during runtime.
:::

- [Install off-line packages](#install-off-line-packages)
- [Install with conda](#install-with-conda)
- [Install with docker](#install-with-docker)
- [Install Python interface with pip](#install-python-interface-with-pip)


## Install off-line packages
Both CPU and GPU version offline packages are available in [the Releases page](https://github.com/deepmodeling/deepmd-kit/releases).

Some packages are splited into two files due to size limit of GitHub. One may merge them into one after downloading:
```bash
cat deepmd-kit-2.1.1-cuda11.6_gpu-Linux-x86_64.sh.0 deepmd-kit-2.1.1-cuda11.6_gpu-Linux-x86_64.sh.1 > deepmd-kit-2.1.1-cuda11.6_gpu-Linux-x86_64.sh
```

One may enable the environment using
```bash
conda activate /path/to/deepmd-kit
```

## Install with conda
DeePMD-kit is available with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

### Official channel

One may create an environment that contains the CPU version of DeePMD-kit and LAMMPS:
```bash
conda create -n deepmd deepmd-kit=*=*cpu libdeepmd=*=*cpu lammps -c https://conda.deepmodeling.com -c defaults
```

Or one may want to create a GPU environment containing [CUDA Toolkit](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver):
```bash
conda create -n deepmd deepmd-kit=*=*gpu libdeepmd=*=*gpu lammps cudatoolkit=11.6 horovod -c https://conda.deepmodeling.com -c defaults
```
One could change the CUDA Toolkit version from `10.2` or `11.6`.

One may specify the DeePMD-kit version such as `2.1.1` using
```bash
conda create -n deepmd deepmd-kit=2.1.1=*cpu libdeepmd=2.1.1=*cpu lammps horovod -c https://conda.deepmodeling.com -c defaults
```

One may enable the environment using
```bash
conda activate deepmd
```

### conda-forge channel

DeePMD-kit is also available on the [conda-forge](https://conda-forge.org/) channel:

```bash
conda create -n deepmd deepmd-kit lammps -c conda-forge
```

The supported platform includes Linux x86-64, macOS x86-64, and macOS arm64.
Read [conda-forge FAQ](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch) to learn how to install CUDA-enabled packages.

## Install with docker
A docker for installing the DeePMD-kit is available [here](https://github.com/orgs/deepmodeling/packages/container/package/deepmd-kit).

To pull the CPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:2.1.1_cpu
```

To pull the GPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:2.1.1_cuda11.6_gpu
```

To pull the ROCm version:
```bash
docker pull deepmodeling/dpmdkit-rocm:dp2.0.3-rocm4.5.2-tf2.6-lmp29Sep2021
```

## Install Python interface with pip

If you have no existing TensorFlow installed, you can use `pip` to install the pre-built package of the Python interface with CUDA 11 supported:

```bash
pip install deepmd-kit[gpu,cu11]
```

`cu11` is required only when CUDA Toolkit and cuDNN were not installed.

Or install the CPU version without CUDA supported:
```bash
pip install deepmd-kit[cpu]
```

[The LAMMPS module](../third-party/lammps-command.md) and [the i-Pi driver](../third-party/ipi.md) are only provided on Linux and macOS. To install LAMMPS and/or i-Pi, add `lmp` and/or `ipi` to extras:
```bash
pip install deepmd-kit[gpu,cu11,lmp,ipi]
```
MPICH is required for parallel running.

It is suggested to install the package into an isolated environment.
The supported platform includes Linux x86-64 and aarch64 with GNU C Library 2.28 or above, macOS x86-64, and Windows x86-64.
A specific version of TensorFlow which is compatible with DeePMD-kit will be also installed.

:::{Warning}
If your platform is not supported, or want to build against the installed TensorFlow, or want to enable ROCM support, please [build from source](install-from-source.md).
:::
