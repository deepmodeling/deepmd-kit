# Install from source code

Please follow our [GitHub](https://github.com/deepmodeling/deepmd-kit) webpage to download the [latest released version](https://github.com/deepmodeling/deepmd-kit/tree/master) and [development version](https://github.com/deepmodeling/deepmd-kit/tree/devel).

Or get the DeePMD-kit source code by `git clone`

```bash
cd /some/workspace
git clone https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
```

For convenience, you may want to record the location of the source to a variable, saying `deepmd_source_dir` by

```bash
cd deepmd-kit
deepmd_source_dir=`pwd`
```

## Install the Python interface

### Install Backend's Python interface

First, check the Python version on your machine.
Python 3.8 or above is required.

```bash
python --version
```

We follow the virtual environment approach to install the backend's Python interface.
Now we assume that the Python interface will be installed in the virtual environment directory `$deepmd_venv`:

```bash
virtualenv -p python3 $deepmd_venv
source $deepmd_venv/bin/activate
pip install --upgrade pip
```

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

The full instruction to install TensorFlow can be found on the official [TensorFlow website](https://www.tensorflow.org/install/pip). TensorFlow 2.7 or later is supported.

```bash
pip install --upgrade tensorflow
```

If one does not need the GPU support of DeePMD-kit and is concerned about package size, the CPU-only version of TensorFlow should be installed by

```bash
pip install --upgrade tensorflow-cpu
```

One can also [use conda](https://docs.deepmodeling.org/faq/conda.html) to install TensorFlow from [conda-forge](https://conda-forge.org).

To verify the installation, run

```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

One can also [build the TensorFlow Python interface from source](https://www.tensorflow.org/install/source) for customized hardware optimization, such as CUDA, ROCM, or OneDNN support.

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

To install PyTorch, run

```sh
pip install torch
```

Follow [PyTorch documentation](https://pytorch.org/get-started/locally/) to install PyTorch built against different CUDA versions or without CUDA.

One can also [use conda](https://docs.deepmodeling.org/faq/conda.html) to install PyTorch from [conda-forge](https://conda-forge.org).

:::

::::

It is important that every time a new shell is started and one wants to use `DeePMD-kit`, the virtual environment should be activated by

```bash
source $deepmd_venv/bin/activate
```

if one wants to skip out of the virtual environment, he/she can do

```bash
deactivate
```

If one has multiple python interpreters named something like python3.x, it can be specified by, for example

```bash
virtualenv -p python3.8 $deepmd_venv
```

One should remember to activate the virtual environment every time he/she uses DeePMD-kit.

### Install the DeePMD-kit's python interface

Check the compiler version on your machine

```
gcc --version
```

The compiler GCC 4.8 or later is supported in the DeePMD-kit.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

Note that TensorFlow may have specific requirements for the compiler version to support the C++ standard version and [`_GLIBCXX_USE_CXX11_ABI`](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) used by TensorFlow. It is recommended to use [the same compiler version as TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations), which can be printed by `python -c "import tensorflow;print(tensorflow.version.COMPILER_VERSION)"`.

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

You can set the environment variable `export DP_ENABLE_PYTORCH=1` to enable customized C++ OPs in the PyTorch backend.
Note that PyTorch may have specific requirements for the compiler version to support the C++ standard version and [`_GLIBCXX_USE_CXX11_ABI`](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html) used by PyTorch.

:::

::::

Execute

```bash
cd $deepmd_source_dir
pip install .
```

One may set the following environment variables before executing `pip`:

| Environment variables                               | Allowed value         | Default value          | Usage                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------- | --------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DP_VARIANT                                          | `cpu`, `cuda`, `rocm` | `cpu`                  | Build CPU variant or GPU variant with CUDA or ROCM support.                                                                                                                                                                                                                                                                                                                                                                                         |
| CUDAToolkit_ROOT                                    | Path                  | Detected automatically | The path to the CUDA toolkit directory. CUDA 9.0 or later is supported. NVCC is required.                                                                                                                                                                                                                                                                                                                                                           |
| ROCM_ROOT                                           | Path                  | Detected automatically | The path to the ROCM toolkit directory.                                                                                                                                                                                                                                                                                                                                                                                                             |
| DP_ENABLE_TENSORFLOW                                | 0, 1                  | 1                      | {{ tensorflow_icon }} Enable the TensorFlow backend.                                                                                                                                                                                                                                                                                                                                                                                                |
| DP_ENABLE_PYTORCH                                   | 0, 1                  | 0                      | {{ pytorch_icon }} Enable customized C++ OPs for the PyTorch backend. PyTorch can still run without customized C++ OPs, but features will be limited.                                                                                                                                                                                                                                                                                               |
| TENSORFLOW_ROOT                                     | Path                  | Detected automatically | {{ tensorflow_icon }} The path to TensorFlow Python library. By default the installer only finds TensorFlow under user site-package directory (`site.getusersitepackages()`) or system site-package directory (`sysconfig.get_path("purelib")`) due to limitation of [PEP-517](https://peps.python.org/pep-0517/). If not found, the latest TensorFlow (or the environment variable `TENSORFLOW_VERSION` if given) from PyPI will be built against. |
| PYTORCH_ROOT                                        | Path                  | Detected automatically | {{ pytorch_icon }} The path to PyTorch Python library. By default, the installer only finds PyTorch under the user site-package directory (`site.getusersitepackages()`) or the system site-package directory (`sysconfig.get_path("purelib")`) due to the limitation of [PEP-517](https://peps.python.org/pep-0517/). If not found, the latest PyTorch (or the environment variable `PYTORCH_VERSION` if given) from PyPI will be built against.   |
| DP_ENABLE_NATIVE_OPTIMIZATION                       | 0, 1                  | 0                      | Enable compilation optimization for the native machine's CPU type. Do not enable it if generated code will run on different CPUs.                                                                                                                                                                                                                                                                                                                   |
| CMAKE_ARGS                                          | str                   | -                      | Additional CMake arguments                                                                                                                                                                                                                                                                                                                                                                                                                          |
| &lt;LANG&gt;FLAGS (`<LANG>`=`CXX`, `CUDA` or `HIP`) | str                   | -                      | Default compilation flags to be used when compiling `<LANG>` files. See [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html).                                                                                                                                                                                                                                                                                  |

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

### Install horovod and mpi4py {{ tensorflow_icon }}

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

If one does not need to use DeePMD-kit with LAMMPS or i-PI, then the python interface installed in the previous section does everything and he/she can safely skip this section.

### Install Backends' C++ interface (optional)

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

Since TensorFlow 2.12, TensorFlow C++ library (`libtensorflow_cc`) is packaged inside the Python library. Thus, you can skip building TensorFlow C++ library manually. If that does not work for you, you can still build it manually.

The C++ interface of DeePMD-kit was tested with compiler GCC >= 4.8. It is noticed that the i-PI support is only compiled with GCC >= 4.8. Note that TensorFlow may have specific requirements for the compiler version.

First, the C++ interface of TensorFlow should be installed. It is noted that the version of TensorFlow should be consistent with the python interface. You may follow [the instruction](install-tf.2.12.md) or run the script `$deepmd_source_dir/source/install/build_tf.py` to install the corresponding C++ interface.

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

If you have installed PyTorch using pip, you can use libtorch inside the PyTorch Python package.
You can also download libtorch prebuilt library from the [PyTorch website](https://pytorch.org/get-started/locally/).

:::

::::

### Install DeePMD-kit's C++ interface

Now go to the source code directory of DeePMD-kit and make a building place.

```bash
cd $deepmd_source_dir/source
mkdir build
cd build
```

The installation requires CMake 3.16 or later for the CPU version, CMake 3.23 or later for the CUDA support, and CMake 3.21 or later for the ROCM support. One can install CMake via `pip` if it is not installed or the installed version does not satisfy the requirement:

```sh
pip install -U cmake
```

You must enable at least one backend.
If you enable two or more backends, these backend libraries must be built in a compatible way, e.g. using the same `_GLIBCXX_USE_CXX11_ABI` flag.
We recommend using [conda pacakges](https://docs.deepmodeling.org/faq/conda.html) from [conda-forge](https://conda-forge.org), which are usually compatible to each other.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

I assume you have activated the TensorFlow Python environment and want to install DeePMD-kit into path `$deepmd_root`, then execute CMake

```bash
cmake -DENABLE_TENSORFLOW=TRUE -DUSE_TF_PYTHON_LIBS=TRUE -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```

If you specify `-DUSE_TF_PYTHON_LIBS=FALSE`, you need to give the location where TensorFlow's C++ interface is installed to `-DTENSORFLOW_ROOT=${tensorflow_root}`.

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

I assume you have installed the PyTorch (either Python or C++ interface) to `$torch_root`, then execute CMake

```bash
cmake -DENABLE_PYTORCH=TRUE -DCMAKE_PREFIX_PATH=$torch_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```

You can specify `-DUSE_PT_PYTHON_LIBS=TRUE` to use libtorch from the Python installation,
but you need to be careful that [PyTorch PyPI packages are still built using `_GLIBCXX_USE_CXX11_ABI=0`](https://github.com/pytorch/pytorch/issues/51039), which may be not compatible with other libraries.

```bash
cmake -DENABLE_PYTORCH=TRUE -DUSE_PT_PYTHON_LIBS=TRUE -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```

:::

::::

One may add the following arguments to `cmake`:

| CMake Aurgements                                                             | Allowed value     | Default value          | Usage                                                                                                                                                                                             |
| ---------------------------------------------------------------------------- | ----------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| -DENABLE_TENSORFLOW=&lt;value&gt;                                            | `TRUE` or `FALSE` | `FALSE`                | {{ tensorflow_icon }} Whether building the TensorFlow backend.                                                                                                                                    |
| -DENABLE_PYTORCH=&lt;value&gt;                                               | `TRUE` or `FALSE` | `FALSE`                | {{ pytorch_icon }} Whether building the PyTorch backend.                                                                                                                                          |
| -DTENSORFLOW_ROOT=&lt;value&gt;                                              | Path              | -                      | {{ tensorflow_icon }} The Path to TensorFlow's C++ interface.                                                                                                                                     |
| -DCMAKE_INSTALL_PREFIX=&lt;value&gt;                                         | Path              | -                      | The Path where DeePMD-kit will be installed.                                                                                                                                                      |
| -DUSE_CUDA_TOOLKIT=&lt;value&gt;                                             | `TRUE` or `FALSE` | `FALSE`                | If `TRUE`, Build GPU support with CUDA toolkit.                                                                                                                                                   |
| -DCUDAToolkit_ROOT=&lt;value&gt;                                             | Path              | Detected automatically | The path to the CUDA toolkit directory. CUDA 9.0 or later is supported. NVCC is required.                                                                                                         |
| -DUSE_ROCM_TOOLKIT=&lt;value&gt;                                             | `TRUE` or `FALSE` | `FALSE`                | If `TRUE`, Build GPU support with ROCM toolkit.                                                                                                                                                   |
| -DCMAKE_HIP_COMPILER_ROCM_ROOT=&lt;value&gt;                                 | Path              | Detected automatically | The path to the ROCM toolkit directory.                                                                                                                                                           |
| -DLAMMPS_SOURCE_ROOT=&lt;value&gt;                                           | Path              | -                      | Only neccessary for LAMMPS plugin mode. The path to the [LAMMPS source code](install-lammps.md). LAMMPS 8Apr2021 or later is supported. If not assigned, the plugin mode will not be enabled.     |
| -DUSE_TF_PYTHON_LIBS=&lt;value&gt;                                           | `TRUE` or `FALSE` | `FALSE`                | {{ tensorflow_icon }} If `TRUE`, Build C++ interface with TensorFlow's Python libraries (TensorFlow's Python Interface is required). And there's no need for building TensorFlow's C++ interface. |
| -DUSE_PT_PYTHON_LIBS=&lt;value&gt;                                           | `TRUE` or `FALSE` | `FALSE`                | {{ pytorch_icon }} If `TRUE`, Build C++ interface with PyTorch's Python libraries (PyTorch's Python Interface is required). And there's no need for downloading PyTorch's C++ libraries.          |
| -DENABLE_NATIVE_OPTIMIZATION=&lt;value&gt;                                   | `TRUE` or `FALSE` | `FALSE`                | Enable compilation optimization for the native machine's CPU type. Do not enable it if generated code will run on different CPUs.                                                                 |
| -DCMAKE\_&lt;LANG&gt;\_FLAGS=&lt;value&gt; (`<LANG>`=`CXX`, `CUDA` or `HIP`) | str               | -                      | Default compilation flags to be used when compiling `<LANG>` files. See [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html).                                |

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
