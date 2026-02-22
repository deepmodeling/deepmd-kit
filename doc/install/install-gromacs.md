# Install GROMACS with DeePMD-kit

Before following this section, [DeePMD-kit C++ interface](install-from-source.md) should have be installed.

## Patch source code of GROMACS

Download the source code of a supported GROMACS version (2020.2) from https://manual.gromacs.org/2020.2/download.html. Run the following command:

```bash
export PATH=$PATH:$deepmd_kit_root/bin
dp_gmx_patch -d $gromacs_root -v $version -p
```

where `deepmd_kit_root` is the directory where the latest version of DeePMD-kit is installed, and `gromacs_root` refers to the source code directory of GROMACS. And `version` represents the version of GROMACS, where **only 2020.2 is supported now**. If attempting to patch another version of GROMACS you will still need to set `version` to `2020.2` as this is the only supported version, we cannot guarantee that patching other versions of GROMACS will work.

<!-- ## Install C++ api of deepmd-kit and tensorflow
The C++ interface of `deepmd-kit 2.x` and `tensorflow 2.x` are required. -->
<!-- + Tips: C++ api of deepmd and TensorFlow could be easily installed from the deepmd-kit offline packages. But before using tensorflow, you need to manually change the protobuf package to [version 3.9.2](https://github.com/protocolbuffers/protobuf/releases/tag/v3.9.2) in `$deepmd_env_dir/include/google/protobuf` (the offline package will install a version of 3.14, which will cause incompatibility). Here `deepmd_env_dir` refers to the directory of conda environment created by the deepmd-kit offline packages.  -->

## Compile GROMACS with deepmd-kit

The C++ interface of `Deepmd-kit 2.x` and `TensorFlow 2.x` are required. And be aware that only DeePMD-kit with **high precision** is supported now since we cannot ensure single precision is enough for a GROMACS simulation. Here is a sample compile script:

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
          -DCUDAToolkit_ROOT=/path/to/cuda \
          -DCMAKE_INSTALL_PREFIX=/path/to/gromacs-2020.2-deepmd
make -j
make install
```
