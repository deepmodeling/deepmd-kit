# Easy installation methods

There various easy methods to install DeePMD-kit. Choose one that you prefer. If you want to build by yourself, jump to the next two sections.

After your easy installation, DeePMD-kit (`dp`) and LAMMPS (`lmp`) will be available to execute. You can try `dp -h` and `lmp -h` to see the help. `mpirun` is also available considering you may want to run LAMMPS in parallel.

- [Install off-line packages](#Install-off-line-packages)
- [Install with conda](#Install-with-conda)
- [Install with docker](#Install-with-docker)


## Install off-line packages
Both CPU and GPU version offline packages are avaiable in [the Releases page](https://github.com/deepmodeling/deepmd-kit/releases).

## Install with conda
DeePMD-kit is avaiable with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

To install the CPU version:
```bash
conda install deepmd-kit=*=*cpu lammps-dp=*=*cpu -c deepmodeling
```

To install the GPU version containing [CUDA 10.1](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver):
```bash
conda install deepmd-kit=*=*gpu lammps-dp=*=*gpu -c deepmodeling
```

## Install with docker
A docker for installing the DeePMD-kit is available [here](https://github.com/orgs/deepmodeling/packages/container/package/deepmd-kit).

To pull the CPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:1.3.1_cpu
```

To pull the GPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:1.3.1_cuda10.1_gpu
```
