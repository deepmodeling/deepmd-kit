# Easy install

There various easy methods to install DeePMD-kit. Choose one that you prefer. If you want to build by yourself, jump to the next two sections.

After your easy installation, DeePMD-kit (`dp`) and LAMMPS (`lmp`) will be available to execute. You can try `dp -h` and `lmp -h` to see the help. `mpirun` is also available considering you may want to train models or run LAMMPS in parallel.

- [Install off-line packages](#install-off-line-packages)
- [Install with conda](#install-with-conda)
- [Install with docker](#install-with-docker)


## Install off-line packages
Both CPU and GPU version offline packages are available in [the Releases page](https://github.com/deepmodeling/deepmd-kit/releases).

Some packages are splited into two files due to size limit of GitHub. One may merge them into one after downloading:
```bash
cat deepmd-kit-2.0.0-cuda11.3_gpu-Linux-x86_64.sh.0 deepmd-kit-2.0.0-cuda11.3_gpu-Linux-x86_64.sh.1 > deepmd-kit-2.0.0-cuda11.3_gpu-Linux-x86_64.sh
```

## Install with conda
DeePMD-kit is avaiable with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

One may create an environment that contains the CPU version of DeePMD-kit and LAMMPS:
```bash
conda create -n deepmd deepmd-kit=*=*cpu libdeepmd=*=*cpu lammps-dp -c https://conda.deepmodeling.org
```

Or one may want to create a GPU environment containing [CUDA Toolkit](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver):
```bash
conda create -n deepmd deepmd-kit=*=*gpu libdeepmd=*=*gpu lammps-dp cudatoolkit=11.3 horovod -c https://conda.deepmodeling.org
```
One could change the CUDA Toolkit version from `10.1` or `11.3`.

One may speficy the DeePMD-kit version such as `2.0.0` using
```bash
conda create -n deepmd deepmd-kit=2.0.0=*cpu libdeepmd=2.0.0=*cpu lammps-dp=2.0.0 horovod -c https://conda.deepmodeling.org
```

One may enable the environment using
```bash
conda activate deepmd
```

## Install with docker
A docker for installing the DeePMD-kit is available [here](https://github.com/orgs/deepmodeling/packages/container/package/deepmd-kit).

To pull the CPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:2.0.0_cpu
```

To pull the GPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:2.0.0_cuda10.1_gpu
```
