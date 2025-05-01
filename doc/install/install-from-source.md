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
Python 3.9 or above is required.

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

:::{tab-item} JAX {{ jax_icon }}

To install [JAX AI Stack](https://github.com/jax-ml/jax-ai-stack), run

```sh
pip install jax-ai-stack
```

One can also install packages in JAX AI Stack manually.
Follow [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html) to install JAX built against different CUDA versions or without CUDA.

One can also [use conda](https://docs.deepmodeling.org/faq/conda.html) to install JAX from [conda-forge](https://conda-forge.org).

:::

:::{tab-item} Paddle {{ paddle_icon }}

To install Paddle, run

```sh
# cu126
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# cu118
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
# cpu
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

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
virtualenv -p python3.9 $deepmd_venv
```

One should remember to activate the virtual environment every time he/she uses DeePMD-kit.

### Install the DeePMD-kit's python interface

Check the compiler version on your machine

```bash
gcc --version
```

By default, DeePMD-kit uses C++ 14, so the compiler needs to support C++ 14 (GCC 5 or later).
The backend package may use a higher C++ standard version, and thus require a higher compiler version (for example, GCC 7 for C++ 17).

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

:::{envvar} DP_VARIANT

**Choices**: `cpu`, `cuda`, `rocm`; **Default**: `cpu`

Build CPU variant or GPU variant with CUDA or ROCM support.
:::

:::{envvar} CUDAToolkit_ROOT

**Type**: Path; **Default**: Detected automatically

The path to the CUDA toolkit directory. CUDA 9.0 or later is supported. NVCC is required.
:::

:::{envvar} ROCM_ROOT

**Type**: Path; **Default**: Detected automatically

The path to the ROCM toolkit directory. If `ROCM_ROOT` is not set, it will look for `ROCM_PATH`; if `ROCM_PATH` is also not set, it will be detected using `hipconfig --rocmpath`.

:::

:::{envvar} DP_ENABLE_TENSORFLOW

**Choices**: `0`, `1`; **Default**: `1`

{{ tensorflow_icon }} Enable the TensorFlow backend.
:::

:::{envvar} DP_ENABLE_PYTORCH

**Choices**: `0`, `1`; **Default**: `0`

{{ pytorch_icon }} Enable customized C++ OPs for the PyTorch backend. PyTorch can still run without customized C++ OPs, but features will be limited.
:::

:::{envvar} TENSORFLOW_ROOT

**Type**: Path; **Default**: Detected automatically

{{ tensorflow_icon }} The path to TensorFlow Python library. If not given, by default the installer only finds TensorFlow under user site-package directory (`site.getusersitepackages()`) or system site-package directory (`sysconfig.get_path("purelib")`) due to limitation of [PEP-517](https://peps.python.org/pep-0517/). If not found, the latest TensorFlow (or the environment variable `TENSORFLOW_VERSION` if given) from PyPI will be built against.
:::

:::{envvar} PYTORCH_ROOT

**Type**: Path; **Default**: Detected automatically

{{ pytorch_icon }} The path to PyTorch Python library. If not given, by default, the installer only finds PyTorch under the user site-package directory (`site.getusersitepackages()`) or the system site-package directory (`sysconfig.get_path("purelib")`) due to the limitation of [PEP-517](https://peps.python.org/pep-0517/). If not found, the latest PyTorch (or the environment variable `PYTORCH_VERSION` if given) from PyPI will be built against.
:::

:::{envvar} DP_ENABLE_NATIVE_OPTIMIZATION

**Choices**: `0`, `1`; **Default**: `0`

Enable compilation optimization for the native machine's CPU type. Do not enable it if generated code will run on different CPUs.
:::

:::{envvar} CMAKE_ARGS

**Type**: string

Additional CMake arguments.
:::

:::{envvar} <LANG>FLAGS

`<LANG>`=`CXX`, `CUDA` or `HIP`

**Type**: string

Default compilation flags to be used when compiling `<LANG>` files. See [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html) for details.
:::

Other [CMake environment variables](https://cmake.org/cmake/help/latest/manual/cmake-env-variables.7.html) may also be critical.

To test the installation, one should first jump out of the source directory

```bash
cd /some/other/workspace
```

then execute

```bash
dp -h
```

It will print the help information like

```{program-output} dp -h

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

:::{tab-item} TensorFlow {{ tensorflow_icon }} / JAX {{ jax_icon }}

The C++ interfaces of both TensorFlow and JAX backends are based on the TensorFlow C++ library.

Since TensorFlow 2.12, TensorFlow C++ library (`libtensorflow_cc`) is packaged inside the Python library. Thus, you can skip building TensorFlow C++ library manually. If that does not work for you, you can still build it manually.

The C++ interface of DeePMD-kit was tested with compiler GCC >= 4.8. It is noticed that the i-PI support is only compiled with GCC >= 4.8. Note that TensorFlow may have specific requirements for the compiler version.

First, the C++ interface of TensorFlow should be installed. It is noted that the version of TensorFlow should be consistent with the python interface. You may follow [the instruction](install-tf.2.12.md) or run the script `$deepmd_source_dir/source/install/build_tf.py` to install the corresponding C++ interface.

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

If you have installed PyTorch using pip, you can use libtorch inside the PyTorch Python package.
You can also download libtorch prebuilt library from the [PyTorch website](https://pytorch.org/get-started/locally/).

:::

:::{tab-item} JAX {{ jax_icon }}

The JAX backend only depends on the TensorFlow C API, which is included in both TensorFlow C++ library and [TensorFlow C library](https://www.tensorflow.org/install/lang_c).
If you want to use the TensorFlow C++ library, just enable the TensorFlow backend (which depends on the TensorFlow C++ library) and nothing else needs to do.
If you want to use the TensorFlow C library and disable the TensorFlow backend,
download the TensorFlow C library from [this page](https://www.tensorflow.org/install/lang_c#download_and_extract).

:::

:::{tab-item} Paddle {{ paddle_icon }}

If you want to use C++ interface of Paddle, you need to compile the Paddle inference library(C++ interface) manually from the [linux-compile-by-make](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/compile/linux-compile-by-make.html), then use the `.so` and `.a` files in `Paddle/build/paddle_inference_install_dir/`.

We also provide a weekly-build Paddle C++ inference library for Linux x86_64 with CUDA 11.8/12.3/CPU below:

CUDA 11.8: [Cuda118_cudnn860_Trt8531_D1/latest/paddle_inference.tgz](https://paddle-qa.bj.bcebos.com/paddle-pipeline/GITHUB_Docker_Compile_Test_Cuda118_cudnn860_Trt8531_D1/latest/paddle_inference.tgz)

CUDA 12.3: [Cuda123_cudnn900_Trt8616_D1/latest/paddle_inference.tgz](https://paddle-qa.bj.bcebos.com/paddle-pipeline/GITHUB_Docker_Compile_Test_Cuda123_cudnn900_Trt8616_D1/latest/paddle_inference.tgz)

CPU: [GITHUB_Docker_Compile_Test_Cpu_Mkl_Avx_D1/latest/paddle_inference.tgz](https://paddle-qa.bj.bcebos.com/paddle-pipeline/GITHUB_Docker_Compile_Test_Cpu_Mkl_Avx_D1/latest/paddle_inference.tgz)

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
We recommend using [conda packages](https://docs.deepmodeling.org/faq/conda.html) from [conda-forge](https://conda-forge.org), which are usually compatible to each other.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }} / JAX {{ jax_icon }}

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

You can specify `-DUSE_PT_PYTHON_LIBS=TRUE` to use libtorch from the Python installation.

```bash
cmake -DENABLE_PYTORCH=TRUE -DUSE_PT_PYTHON_LIBS=TRUE -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```

:::

:::{tab-item} JAX {{ jax_icon }}

If you want to use the TensorFlow C++ library, just enable the TensorFlow backend and nothing else needs to do.
If you want to use the TensorFlow C library and disable the TensorFlow backend, set {cmake:variable}`ENABLE_JAX` to `ON` and `CMAKE_PREFIX_PATH` to the root directory of the [TensorFlow C library](https://www.tensorflow.org/install/lang_c).

```bash
cmake -DENABLE_JAX=ON -D CMAKE_PREFIX_PATH=${tensorflow_c_root} ..
```

:::

:::{tab-item} Paddle {{ paddle_icon }}

I assume you have get the Paddle inference library(C++ interface) to `$PADDLE_INFERENCE_DIR`, then execute CMake

```bash
cmake -DENABLE_PADDLE=ON -DPADDLE_INFERENCE_DIR=$PADDLE_INFERENCE_DIR -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```

:::

::::

One may add the following CMake variables to `cmake` using the [`-D <var>=<value>` option](https://cmake.org/cmake/help/latest/manual/cmake.1.html#cmdoption-cmake-D):

:::{cmake:variable} ENABLE_TENSORFLOW

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

{{ tensorflow_icon }} {{ jax_icon }} Whether building the TensorFlow backend and the JAX backend.
Setting this option to `ON` will also set {cmake:variable}`ENABLE_JAX` to `ON`.

:::

:::{cmake:variable} ENABLE_PYTORCH

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

{{ pytorch_icon }} Whether building the PyTorch backend.

:::

:::{cmake:variable} ENABLE_JAX

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

{{ jax_icon }} Build the JAX backend.
If {cmake:variable}`ENABLE_TENSORFLOW` is `ON`, the TensorFlow C++ library is used to build the JAX backend;
If {cmake:variable}`ENABLE_TENSORFLOW` is `OFF`, the TensorFlow C library is used to build the JAX backend.

:::

:::{cmake:variable} ENABLE_PADDLE

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

{{ paddle_icon }} Whether building the Paddle backend.

:::

:::{cmake:variable} TENSORFLOW_ROOT

**Type**: `PATH`

{{ tensorflow_icon }} {{ jax_icon }} The Path to TensorFlow's C++ interface.

:::

:::{cmake:variable} PADDLE_INFERENCE_DIR

**Type**: `PATH`

{{ paddle_icon }} The Path to Paddle's C++ inference directory, such as `/path/to/paddle_inference_install_dir` or `/path/to/paddle_inference`.

:::

:::{cmake:variable} CMAKE_INSTALL_PREFIX

**Type**: `PATH`

The Path where DeePMD-kit will be installed.
See also [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html).

:::

:::{cmake:variable} USE_CUDA_TOOLKIT

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

If `TRUE`, Build GPU support with CUDA toolkit.

:::

:::{cmake:variable} CUDAToolkit_ROOT

**Type**: `PATH`, **Default**: [Search automatically](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html)

The path to the CUDA toolkit directory. CUDA 9.0 or later is supported. NVCC is required.
See also [CMake documentation](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html).

:::

:::{cmake:variable} USE_ROCM_TOOLKIT

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

If `TRUE`, Build GPU support with ROCM toolkit.

:::

:::{cmake:variable} CMAKE_HIP_COMPILER_ROCM_ROOT

**Type**: `PATH`, **Default**: [Search automatically](https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html)

The path to the ROCM toolkit directory.
See also [ROCm documentation](https://rocm.docs.amd.com/en/latest/conceptual/cmake-packages.html).

:::

:::{cmake:variable} LAMMPS_SOURCE_ROOT

**Type**: `PATH`

Only necessary for using [LAMMPS plugin mode](./install-lammps.md#install-lammps-plugin-mode).
The path to the [LAMMPS source code](install-lammps.md).
LAMMPS 8Apr2021 or later is supported.
If not assigned, the plugin mode will not be enabled.

:::

:::{cmake:variable} USE_TF_PYTHON_LIBS

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

{{ tensorflow_icon }} If `TRUE`, Build C++ interface with TensorFlow's Python libraries (TensorFlow's Python Interface is required).
There's no need for building TensorFlow's C++ interface.

:::

:::{cmake:variable} USE_PT_PYTHON_LIBS

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

{{ pytorch_icon }} If `TRUE`, Build C++ interface with PyTorch's Python libraries (PyTorch's Python Interface is required).
There's no need for downloading PyTorch's C++ libraries.

:::

:::{cmake:variable} ENABLE_NATIVE_OPTIMIZATION

**Type**: `BOOL` (`ON`/`OFF`), Default: `OFF`

Enable compilation optimization for the native machine's CPU type.
Do not enable it if generated code will run on different CPUs.

:::

<!-- prettier-ignore -->
:::{cmake:variable} CMAKE_<LANG>_FLAGS

(`<LANG>`=`CXX`, `CUDA` or `HIP`)

**Type**: `STRING`

Default compilation flags to be used when compiling `<LANG>` files.
See also [CMake documentation](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html).

:::

---

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
