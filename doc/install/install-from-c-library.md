# Install from pre-compiled C library

DeePMD-kit provides pre-compiled C library package (`libdeepmd_c.tar.gz`) in each [release](https://github.com/deepmodeling/deepmd-kit/releases). It can be used to build the [LAMMPS plugin](./install-lammps.md) and [GROMACS patch](./install-gromacs.md), as well as many [third-party software packages](../third-party/out-of-deepmd-kit.md), without building TensorFlow and DeePMD-kit on one's own.

The library is built in Linux (GLIBC 2.17) with CUDA 11.8. It's noted that this package does not contain CUDA Toolkit and cuDNN, so one needs to download them from the NVIDIA website.

## Use Pre-compiled C Library to build the LAMMPS plugin and GROMACS patch

When one [installs DeePMD-kit's C++ interface](./install-from-source.md#install-deepmd-kits-c-interface), one can use the CMake argument `DEEPMD_C_ROOT` to the path `libdeepmd_c`.

```bash
cd $deepmd_source_dir/source
mkdir build
cd build
cmake -DDEEPMD_C_ROOT=/path/to/libdeepmd_c -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
make -j8
make install
```

Then one can follow the manual [Install LAMMPS](./install-lammps.md) and/or [Install GROMACS](./install-gromacs.md).
