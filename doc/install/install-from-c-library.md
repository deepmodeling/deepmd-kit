# Install from pre-compiled C library {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }}

> [!NOTE]
> **Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}

DeePMD-kit provides pre-compiled C library package (`libdeepmd_c.tar.gz`) in each [release](https://github.com/deepmodeling/deepmd-kit/releases). It can be used to build the [LAMMPS plugin](./install-lammps.md) and the [i-PI driver](./install-ipi.md), as well as many [third-party software packages](../third-party/out-of-deepmd-kit.md), without building TensorFlow and DeePMD-kit on one's own.
It can be downloaded via the shell command:

```sh
wget https://github.com/deepmodeling/deepmd-kit/releases/latest/download/libdeepmd_c.tar.gz
tar xzf libdeepmd_c.tar.gz
```

The library is built in Linux (GLIBC 2.28) with CUDA 12.9 (`libdeepmd_c.tar.gz`). It's noted that this package does not contain CUDA Toolkit, cuDNN, or PyTorch runtime libraries.
To use the PyTorch C/C++ backend on Linux, install a libtorch runtime that exactly matches the PyTorch version used to build the package.
The PyTorch version must match exactly, while the CUDA variant may be omitted only when the target runtime is compatible.
Make the libtorch `lib` directory discoverable by the dynamic linker, for example by adding it to `LD_LIBRARY_PATH`.
The C library package includes `download_libtorch.sh`, which downloads and unpacks the matching libtorch runtime and writes `libtorch_env.sh`:

```sh
cd libdeepmd_c
./download_libtorch.sh
. ./libtorch_env.sh
```

## Use Pre-compiled C Library to build the LAMMPS plugin and i-PI driver

When one [installs DeePMD-kit's C++ interface](./install-from-source.md#install-deepmd-kits-c-interface), one can use the CMake argument {cmake:variable}`DEEPMD_C_ROOT` to the path `libdeepmd_c`.

```bash
cd $deepmd_source_dir/source
mkdir build
cd build
cmake -DDEEPMD_C_ROOT=/path/to/libdeepmd_c -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j8
make install
```

Then the i-PI driver `dp_ipi` will be built and installed.
One can also follow the manual [Install LAMMPS](./install-lammps.md). For historical GROMACS context, see the [deprecation notice](./install-gromacs.md).

:::{cmake:variable} DEEPMD_C_ROOT

**Type**: `Path`

Prefix to the pre-compiled C library.
:::
