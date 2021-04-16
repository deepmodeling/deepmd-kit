# With conda
DeePMD-kit is avaiable with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

To install the CPU version:
```bash
conda install deepmd-kit=*=*cpu lammps-dp=*=*cpu -c deepmodeling
```

To install the GPU version containing [CUDA 10.1](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver):
```bash
conda install deepmd-kit=*=*gpu lammps-dp=*=*gpu -c deepmodeling
```
