# Install from source code

Please follow our [GitHub](https://github.com/deepmodeling/deepmd-kit) webpage to download the [latest released version](https://github.com/deepmodeling/deepmd-kit/tree/master) and [development version](https://github.com/deepmodeling/deepmd-kit/tree/devel).

Or get the DeePMD-kit source code by `git clone`
```bash
cd /some/workspace
git clone --recursive https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
```
The `--recursive` option clones all [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) needed by DeePMD-kit.

For convenience, you may want to record the location of the source to a variable, saying `deepmd_source_dir` by
```bash
cd deepmd-kit
deepmd_source_dir=`pwd`
```

## Install the python interface
### Install Tensorflow's python interface
First, check the python version on your machine
```bash
python --version
```

We follow the virtual environment approach to install TensorFlow's Python interface. The full instruction can be found on the official [TensorFlow website](https://www.tensorflow.org/install/pip). TensorFlow 1.8 or later is supported. Now we assume that the Python interface will be installed to the virtual environment directory `$tensorflow_venv`
```bash
virtualenv -p python3 $tensorflow_venv
source $tensorflow_venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
```
It is important that every time a new shell is started and one wants to use `DeePMD-kit`, the virtual environment should be activated by
```bash
source $tensorflow_venv/bin/activate
```
if one wants to skip out of the virtual environment, he/she can do
```bash
deactivate
```
If one has multiple python interpreters named something like python3.x, it can be specified by, for example
```bash
virtualenv -p python3.8 $tensorflow_venv
```
If one does not need the GPU support of DeePMD-kit and is concerned about package size, the CPU-only version of TensorFlow should be installed by
```bash
pip install --upgrade tensorflow-cpu
```
To verify the installation, run
```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
One should remember to activate the virtual environment every time he/she uses DeePMD-kit.

One can also [build the TensorFlow Python interface from source](https://www.tensorflow.org/install/source) for custom hardware optimization, such as CUDA, ROCM, or OneDNN support.

### Install the DeePMD-kit's python interface

Check the compiler version on your machine

```
gcc --version
```

The compiler GCC 4.8 or later is supported in the DeePMD-kit. Note that TensorFlow may have specific requirements for the compiler version. It is recommended to use the same compiler version as TensorFlow, which can be printed by `python -c "import tensorflow;print(tensorflow.version.COMPILER_VERSION)"`.

Execute
```bash
cd $deepmd_source_dir
pip install .
```

One may set the following environment variables before executing `pip`:

| Environment variables | Allowed value          | Default value | Usage                      |
| --------------------- | ---------------------- | ------------- | -------------------------- |
| DP_VARIANT            | `cpu`, `cuda`, `rocm`  | `cpu`         | Build CPU variant or GPU variant with CUDA or ROCM support. |
| CUDA_TOOLKIT_ROOT_DIR | Path                   | Detected automatically | The path to the CUDA toolkit directory. CUDA 7.0 or later is supported. NVCC is required. |
| ROCM_ROOT             | Path                   | Detected automatically | The path to the ROCM toolkit directory. |
| TENSORFLOW_ROOT       | Path                   | Detected automatically | The path to TensorFlow Python library. By default the installer only finds TensorFlow under user site-package directory (`site.getusersitepackages()`) or system site-package directory (`sysconfig.get_path("purelib")`) due to limitation of [PEP-517](https://peps.python.org/pep-0517/). If not found, the latest TensorFlow (or the environment variable `TENSORFLOW_VERSION` if given) from PyPI will be built against.|
| DP_ENABLE_NATIVE_OPTIMIZATION | 0, 1           | 0             | Enable compilation optimization for the native machine's CPU type. Do not enable it if generated code will run on different CPUs. |

To test the installation, one should first jump out of the source directory
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

[Horovod](https://github.com/horovod/horovod) and [mpi4py](https://github.com/mpi4py/mpi4py) are used for parallel training. For better performance on GPU, please follow the tuning steps in [Horovod on GPU](https://github.com/horovod/horovod/blob/master/docs/gpus.rst).
```bash
# With GPU, prefer NCCL as a communicator.
HOROVOD_WITHOUT_GLOO=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_HOME=/path/to/nccl pip install horovod mpi4py
```

If your work in a CPU environment, please prepare runtime as below:
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

Since version 2.0.1, Horovod and mpi4py with MPICH support are shipped with the installer.

If you don't install Horovod, DeePMD-kit will fall back to serial mode.

## Install the C++ interface

If one does not need to use DeePMD-kit with Lammps or I-Pi, then the python interface installed in the previous section does everything and he/she can safely skip this section.

### Install Tensorflow's C++ interface (optional)

Since TensorFlow 2.12, TensorFlow C++ library (`libtensorflow_cc`) is packaged inside the Python library. Thus, you can skip building TensorFlow C++ library manually. If that does not work for you, you can still build it manually.

The C++ interface of DeePMD-kit was tested with compiler GCC >= 4.8. It is noticed that the I-Pi support is only compiled with GCC >= 4.8. Note that TensorFlow may have specific requirements for the compiler version.

First, the C++ interface of Tensorflow should be installed. It is noted that the version of Tensorflow should be consistent with the python interface. You may follow [the instruction](install-tf.2.8.md) or run the script `$deepmd_source_dir/source/install/build_tf.py` to install the corresponding C++ interface.

### Install DeePMD-kit's C++ interface

Now go to the source code directory of DeePMD-kit and make a building place.
```bash
cd $deepmd_source_dir/source
mkdir build
cd build
```
I assume you have activated the TensorFlow Python environment and want to install DeePMD-kit into path `$deepmd_root`, then execute CMake
```bash
cmake -DUSE_TF_PYTHON_LIBS=TRUE -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```

If you specify `-DUSE_TF_PYTHON_LIBS=FALSE`, you need to give the location where TensorFlow's C++ interface is installed to `-DTENSORFLOW_ROOT=${tensorflow_root}`.

One may add the following arguments to `cmake`:

| CMake Aurgements         | Allowed value       | Default value | Usage                   |
| ------------------------ | ------------------- | ------------- | ------------------------|
| -DTENSORFLOW_ROOT=&lt;value&gt;  | Path              | -             | The Path to TensorFlow's C++ interface. |
| -DCMAKE_INSTALL_PREFIX=&lt;value&gt; | Path          | -             | The Path where DeePMD-kit will be installed. |
| -DUSE_CUDA_TOOLKIT=&lt;value&gt; | `TRUE` or `FALSE` | `FALSE`       | If `TRUE`, Build GPU support with CUDA toolkit. |
| -DCUDA_TOOLKIT_ROOT_DIR=&lt;value&gt; | Path         | Detected automatically | The path to the CUDA toolkit directory. CUDA 7.0 or later is supported. NVCC is required. |
| -DUSE_ROCM_TOOLKIT=&lt;value&gt; | `TRUE` or `FALSE` | `FALSE`       | If `TRUE`, Build GPU support with ROCM toolkit. |
| -DCMAKE_HIP_COMPILER_ROCM_ROOT=&lt;value&gt; | Path         | Detected automatically | The path to the ROCM toolkit directory. |
| -DLAMMPS_SOURCE_ROOT=&lt;value&gt; | Path         | - | Only neccessary for LAMMPS plugin mode. The path to the [LAMMPS source code](install-lammps.md). LAMMPS 8Apr2021 or later is supported. If not assigned, the plugin mode will not be enabled. |
| -DUSE_TF_PYTHON_LIBS=&lt;value&gt; | `TRUE` or `FALSE` | `FALSE`       | If `TRUE`, Build C++ interface with TensorFlow's Python libraries(TensorFlow's Python Interface is required). And there's no need for building TensorFlow's C++ interface.|
| -DENABLE_NATIVE_OPTIMIZATION       | `TRUE` or `FALSE` | `FALSE`       | Enable compilation optimization for the native machine's CPU type. Do not enable it if generated code will run on different CPUs. |

If the CMake has been executed successfully, then run the following make commands to build the package:
```bash
make -j4
make install
```
Option `-j4` means using 4 processes in parallel. You may want to use a different number according to your hardware.

If everything works fine, you will have the executable and libraries installed in `$deepmd_root/bin` and `$deepmd_root/lib`
```bash
$ ls $deepmd_root/bin
$ ls $deepmd_root/lib
```
