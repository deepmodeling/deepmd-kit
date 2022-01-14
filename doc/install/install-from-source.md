# Install from source code

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
pip install --upgrade tensorflow
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
pip install --upgrade tensorflow-cpu
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

One may set the following environment variables before executing `pip`:

| Environment variables | Allowed value          | Default value | Usage                      |
| --------------------- | ---------------------- | ------------- | -------------------------- |
| DP_VARIANT            | `cpu`, `cuda`, `rocm`  | `cpu`         | Build CPU variant or GPU variant with CUDA or ROCM support. |
| CUDA_TOOLKIT_ROOT_DIR | Path                   | Detected automatically | The path to the CUDA toolkit directory. |
| ROCM_ROOT             | Path                   | Detected automatically | The path to the ROCM toolkit directory. |

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

### Install horovod and mpi4py

[Horovod](https://github.com/horovod/horovod) and [mpi4py](https://github.com/mpi4py/mpi4py) is used for parallel training. For better performance on GPU, please follow tuning steps in [Horovod on GPU](https://github.com/horovod/horovod/blob/master/docs/gpus.rst).
```bash
# With GPU, prefer NCCL as communicator.
HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install horovod mpi4py
```

If your work in CPU environment, please prepare runtime as below:
```bash
# By default, MPI is used as communicator.
HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod mpi4py
```

To ensure Horovod has been built with proper framework support enabled, one can invoke the `horovodrun --check-build` command, e.g.,

```bash
$ horovodrun --check-build

Horovod v0.22.1:

Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo
```

From version 2.0.1, Horovod and mpi4py with MPICH support is shipped with the installer.

If you don't install horovod, DeePMD-kit will fallback to serial mode.

## Install the C++ interface 

If one does not need to use DeePMD-kit with Lammps or I-Pi, then the python interface installed in the previous section does everything and he/she can safely skip this section. 

### Install the Tensorflow's C++ interface

Check the compiler version on your machine

```
gcc --version
```

The C++ interface of DeePMD-kit was tested with compiler gcc >= 4.8. It is noticed that the I-Pi support is only compiled with gcc >= 4.8.

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
where the variable `tensorflow_root` stores the location where the TensorFlow's C++ interface is installed. 

One may add the following arguments to `cmake`:

| CMake Aurgements         | Allowed value       | Default value | Usage                   |
| ------------------------ | ------------------- | ------------- | ------------------------|
| -DTENSORFLOW_ROOT=&lt;value&gt;  | Path              | -             | The Path to TensorFlow's C++ interface. |
| -DCMAKE_INSTALL_PREFIX=&lt;value&gt; | Path          | -             | The Path where DeePMD-kit will be installed. |
| -DUSE_CUDA_TOOLKIT=&lt;value&gt; | `TRUE` or `FALSE` | `FALSE`       | If `TRUE`, Build GPU support with CUDA toolkit. |
| -DCUDA_TOOLKIT_ROOT_DIR=&lt;value&gt; | Path         | Detected automatically | The path to the CUDA toolkit directory. |
| -DUSE_ROCM_TOOLKIT=&lt;value&gt; | `TRUE` or `FALSE` | `FALSE`       | If `TRUE`, Build GPU support with ROCM toolkit. |
| -DROCM_ROOT=&lt;value&gt; | Path         | Detected automatically | The path to the ROCM toolkit directory. |
| -DLAMMPS_VERSION_NUMBER=&lt;value&gt; | Number         | `20210929` | Only neccessary for LAMMPS built-in mode. The version number of LAMMPS (yyyymmdd). |
| -DLAMMPS_SOURCE_ROOT=&lt;value&gt; | Path         | - | Only neccessary for LAMMPS plugin mode. The path to the LAMMPS source code (later than 8Apr2021). If not assigned, the plugin mode will not be enabled. |

If the cmake has executed successfully, then 
```bash
make -j4
make install
```
The option `-j4` means using 4 processes in parallel. You may want to use a different number according to your hardware. 

If everything works fine, you will have the following executable and libraries installed in `$deepmd_root/bin` and `$deepmd_root/lib`
```bash
$ ls $deepmd_root/bin
dp_ipi      dp_ipi_low
$ ls $deepmd_root/lib
libdeepmd_cc_low.so  libdeepmd_ipi_low.so  libdeepmd_lmp_low.so  libdeepmd_low.so          libdeepmd_op_cuda.so  libdeepmd_op.so
libdeepmd_cc.so      libdeepmd_ipi.so      libdeepmd_lmp.so      libdeepmd_op_cuda_low.so  libdeepmd_op_low.so   libdeepmd.so
```
