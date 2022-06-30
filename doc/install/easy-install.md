# Easy install

There are various easy methods to install DeePMD-kit. Choose one that you prefer. If you want to build by yourself, jump to the next two sections.

After your easy installation, DeePMD-kit (`dp`) and LAMMPS (`lmp`) will be available to execute. You can try `dp -h` and `lmp -h` to see the help. `mpirun` is also available considering you may want to train models or run LAMMPS in parallel.

Note: The off-line packages and conda packages require the [GNU C Library](https://www.gnu.org/software/libc/) 2.17 or above. The GPU version requires [compatible NVIDIA driver](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#minor-version-compatibility) to be installed in advance. It is possible to force conda to [override detection](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-virtual.html#overriding-detected-packages) when installation, but these requirements are still necessary during runtime.

- [Install off-line packages](#install-off-line-packages)
- [Install with conda](#install-with-conda)
- [Install with docker](#install-with-docker)


## Install off-line packages
Both CPU and GPU version offline packages are available in [the Releases page](https://github.com/deepmodeling/deepmd-kit/releases).

Some packages are splited into two files due to size limit of GitHub. One may merge them into one after downloading:
```bash
cat deepmd-kit-2.1.1-cuda11.6_gpu-Linux-x86_64.sh.0 deepmd-kit-2.1.1-cuda11.6_gpu-Linux-x86_64.sh.1 > deepmd-kit-2.1.1-cuda11.6_gpu-Linux-x86_64.sh
```

## Install with conda
DeePMD-kit is avaiable with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

One may create an environment that contains the CPU version of DeePMD-kit and LAMMPS:
```bash
conda create -n deepmd deepmd-kit=*=*cpu libdeepmd=*=*cpu lammps -c https://conda.deepmodeling.com
```

Or one may want to create a GPU environment containing [CUDA Toolkit](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver):
```bash
conda create -n deepmd deepmd-kit=*=*gpu libdeepmd=*=*gpu lammps cudatoolkit=11.6 horovod -c https://conda.deepmodeling.com
```
One could change the CUDA Toolkit version from `10.2` or `11.6`.

One may speficy the DeePMD-kit version such as `2.1.1` using
```bash
conda create -n deepmd deepmd-kit=2.1.1=*cpu libdeepmd=2.1.1=*cpu lammps horovod -c https://conda.deepmodeling.com
```

One may enable the environment using
```bash
conda activate deepmd
```

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
