# Install from pre-compiled C library {{ tensorflow_icon }} {{ jax_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, JAX {{ jax_icon }}
:::

DeePMD-kit provides pre-compiled C library package (`libdeepmd_c.tar.gz`) in each [release](https://github.com/deepmodeling/deepmd-kit/releases). It can be used to build the [LAMMPS plugin](./install-lammps.md) and [GROMACS patch](./install-gromacs.md), as well as many [third-party software packages](../third-party/out-of-deepmd-kit.md), without building TensorFlow and DeePMD-kit on one's own.
It can be downloaded via the shell command:

```sh
wget https://github.com/deepmodeling/deepmd-kit/releases/latest/download/libdeepmd_c.tar.gz
tar xzf libdeepmd_c.tar.gz
```

The library is built in Linux (GLIBC 2.17) with CUDA 12.2 (`libdeepmd_c.tar.gz`) or 11.8 (`libdeepmd_c_cu11.tar.gz`). It's noted that this package does not contain CUDA Toolkit and cuDNN, so one needs to download them from the NVIDIA website.

## Use Pre-compiled C Library to build the LAMMPS plugin, i-PI driver, and GROMACS patch

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
One can also follow the manual [Install LAMMPS](./install-lammps.md) and/or [Install GROMACS](./install-gromacs.md).

:::{cmake:variable} DEEPMD_C_ROOT

**Type**: `Path`

Prefix to the pre-compiled C library.

:::
