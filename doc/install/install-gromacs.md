# Install GROMACS with DeepMD
## Patch source code of GROMACS 
Download source code of a supported gromacs version (2020.2) from https://manual.gromacs.org/2020.2/download.html. Run the following command:
```bash
export PATH=$PATH:$deepmd_kit_root/bin
dp_gmx_patch -d $gromacs_root -v $version -p
```
where `deepmd_kit_root` is the directory where the latest version of deepmd-kit is installed, and `gromacs_root` refers to source code directory of gromacs. And `version` represents the version of gromacs, **only support 2020.2 now**. You may patch another version of gromacs but still setting `version` to `2020.2`. However, we cannot ensure that it works.

<!-- ## Install C++ api of deepmd-kit and tensorflow
The C++ interface of `deepmd-kit 2.x` and `tensorflow 2.x` are required. -->
<!-- + Tips: C++ api of deepmd and tensorflow could be easily installed from the deepmd-kit offline packages. But before using tensorflow, you need to manually change the protobuf package to [version 3.9.2](https://github.com/protocolbuffers/protobuf/releases/tag/v3.9.2) in `$deepmd_env_dir/include/google/protobuf` (the offline package will install a version of 3.14, which will cause incompability). Here `deepmd_env_dir` refers to the directory of conda environment created by the deepmd-kit offline packages.  -->

## Compile GROMACS with deepmd-kit
The C++ interface of `deepmd-kit 2.x` and `tensorflow 2.x` are required. And be aware that only deepmd-kit with **high precision** is supported now, since we cannot ensure single precision is enough for a GROMACS simulation. Here is a sample compile scipt:
```bash
#!/bin/bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CMAKE_PREFIX_PATH="/path/to/fftw-3.3.9" # fftw libraries
mkdir build
cd build

cmake3 .. -DCMAKE_CXX_STANDARD=14 \ # not required, but c++14 seems to be more compatible with higher version of tensorflow
          -DGMX_MPI=ON \
          -DGMX_GPU=CUDA \ # Gromacs on ROCm has not been fully developed yet
          -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
          -DCMAKE_INSTALL_PREFIX=/path/to/gromacs-2020.2-deepmd
make -j
make install
```