- [Easy installation methods](#easy-installation-methods)
    - [Offline packages](#offline-packages)
    - [With Docker](#with-docker)
    - [With conda](#with-conda)
- [From source code](#from-source-code)
    - [Install the python interaction](#install-the-python-interface)
      - [Install the Tensorflow's python interface](#install-the-tensorflows-python-interface)
      - [Install the DeePMD-kit's python interface](#install-the-deepmd-kits-python-interface)
    - [Install the C++ interface](#install-the-c-interface)
      - [Install the Tensorflow's C++ interface](#install-the-tensorflows-c-interface)
      - [Install the DeePMD-kit's C++ interface](#install-the-deepmd-kits-c-interface)
      - [Install LAMMPS's DeePMD-kit module](#install-lammpss-deepmd-kit-module)
    - [Hardware platforms](#hardware-platforms)


# Easy installation methods
There various easy methods to install DeePMD-kit. Choose one that you prefer. If you want to build by yourself, jump to the next two sections.

After your easy installation, DeePMD-kit (`dp`) and LAMMPS (`lmp`) will be available to execute. You can try `dp -h` and `lmp -h` to see the help. `mpirun` is also available considering you may want to run LAMMPS in parallel.

## Offline packages
Both CPU and GPU version offline packages are avaiable in [the Releases page](https://github.com/deepmodeling/deepmd-kit/releases).

## With conda
DeePMD-kit is avaiable with [conda](https://github.com/conda/conda). Install [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

To install the CPU version:
```bash
conda install deepmd-kit=*=*cpu lammps-dp=*=*cpu -c deepmodeling
```

To install the GPU version containing [CUDA 10.1](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver):
```bash
conda install deepmd-kit=*=*gpu lammps-dp=*=*gpu -c deepmodeling
```

## With Docker
A docker for installing the DeePMD-kit is available [here](https://github.com/orgs/deepmodeling/packages/container/package/deepmd-kit).

To pull the CPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:1.3.1_cpu
```

To pull the GPU version:
```bash
docker pull ghcr.io/deepmodeling/deepmd-kit:1.3.1_cuda10.1_gpu
```


# From source code

Please follow our [github](https://github.com/deepmodeling/deepmd-kit) webpage to download the [latest released version](https://github.com/deepmodeling/deepmd-kit/tree/master) and [development version](https://github.com/deepmodeling/deepmd-kit/tree/devel).

Or get the DeePMD-kit source code by `git clone`
```bash
cd /some/workspace
git clone --recursive https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
```
The `--recursive` option clones all [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) needed by DeePMD-kit.

For convenience, you may want to record the location of source to a variable, saying `deepmd_source_dir` by
```bash
cd deepmd-kit
deepmd_source_dir=`pwd`
```


## Install the python interface 
### Install the Tensorflow's python interface
First, check the python version on your machine 
```bash
python --version
```

We follow the virtual environment approach to install the tensorflow's Python interface. The full instruction can be found on [the tensorflow's official website](https://www.tensorflow.org/install/pip). Now we assume that the Python interface will be installed to virtual environment directory `$tensorflow_venv`
```bash
virtualenv -p python3 $tensorflow_venv
source $tensorflow_venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow==2.3.0
```
It is notice that everytime a new shell is started and one wants to use `DeePMD-kit`, the virtual environment should be activated by 
```bash
source $tensorflow_venv/bin/activate
```
if one wants to skip out of the virtual environment, he/she can do
```bash
deactivate
```
If one has multiple python interpreters named like python3.x, it can be specified by, for example
```bash
virtualenv -p python3.7 $tensorflow_venv
```
If one does not need the GPU support of deepmd-kit and is concerned about package size, the CPU-only version of tensorflow should be installed by	
```bash	
pip install --upgrade tensorflow-cpu==2.3.0	
```
To verify the installation, run
```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
One should remember to activate the virtual environment every time he/she uses deepmd-kit.

### Install the DeePMD-kit's python interface

Execute
```bash
cd $deepmd_source_dir
pip install .
```
To test the installation, one should firstly jump out of the source directory
```
cd /some/other/workspace
```
then execute
```bash
dp -h
```
It will print the help information like
```text
usage: dp [-h] {train,freeze,test} ...

DeePMD-kit: A deep learning package for many-body potential energy
representation and molecular dynamics

optional arguments:
  -h, --help           show this help message and exit

Valid subcommands:
  {train,freeze,test}
    train              train a model
    freeze             freeze the model
    test               test the model
```

## Install the C++ interface 

If one does not need to use DeePMD-kit with Lammps or I-Pi, then the python interface installed in the previous section does everything and he/she can safely skip this section. 

### Install the Tensorflow's C++ interface

Check the compiler version on your machine

```
gcc --version
```

The C++ interface of DeePMD-kit was tested with compiler gcc >= 4.8. It is noticed that the I-Pi support is only compiled with gcc >= 4.9.

First the C++ interface of Tensorflow should be installed. It is noted that the version of Tensorflow should be in consistent with the python interface. You may follow [the instruction](install-tf.2.3.md) to install the corresponding C++ interface.

### Install the DeePMD-kit's C++ interface

Now goto the source code directory of DeePMD-kit and make a build place.
```bash
cd $deepmd_source_dir/source
mkdir build 
cd build
```
I assume you want to install DeePMD-kit into path `$deepmd_root`, then execute cmake
```bash
cmake -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```
where the variable `tensorflow_root` stores the location where the tensorflow's C++ interface is installed. The DeePMD-kit will automatically detect if a CUDA tool-kit is available on your machine and build the GPU support accordingly. If you want to force the cmake to find CUDA tool-kit, you can speicify the key `USE_CUDA_TOOLKIT`, 
```bash
cmake -DUSE_CUDA_TOOLKIT=true -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```
and you may further asked to provide `CUDA_TOOLKIT_ROOT_DIR`. If the cmake has executed successfully, then 
```bash
make
make install
```
If everything works fine, you will have the following executable and libraries installed in `$deepmd_root/bin` and `$deepmd_root/lib`
```bash
$ ls $deepmd_root/bin
dp_ipi
$ ls $deepmd_root/lib
libdeepmd_ipi.so  libdeepmd_op.so  libdeepmd.so
```

### Install LAMMPS's DeePMD-kit module
DeePMD-kit provide module for running MD simulation with LAMMPS. Now make the DeePMD-kit module for LAMMPS.
```bash
cd $deepmd_source_dir/source/build
make lammps
```
DeePMD-kit will generate a module called `USER-DEEPMD` in the `build` directory. Now download the LAMMPS code (`29Oct2020` or later), and uncompress it:
```bash
cd /some/workspace
wget https://github.com/lammps/lammps/archive/stable_29Oct2020.tar.gz
tar xf stable_29Oct2020.tar.gz
```
The source code of LAMMPS is stored in directory `lammps-stable_29Oct2020`. Now go into the LAMMPS code and copy the DeePMD-kit module like this
```bash
cd lammps-stable_29Oct2020/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
```
Now build LAMMPS
```bash
make yes-kspace
make yes-user-deepmd
make mpi -j4
```
The option `-j4` means using 4 processes in parallel. You may want to use a different number according to your hardware. 

If everything works fine, you will end up with an executable `lmp_mpi`.
```bash
./lmp_mpi -h
```

The DeePMD-kit module can be removed from LAMMPS source code by 
```bash
make no-user-deepmd
```

## Hardware platforms


