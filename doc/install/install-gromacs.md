# Install GROMACS with DeepMD
## 1. Patch source code of GROMACS 
Download source code of a supported gromacs version from https://manual.gromacs.org. Run the following command:
```bash
cd $deepmd_root/source/gmx && chmod +x patch.sh
./patch.sh -d $gromacs_root -v $version -p
```
where `deepmd_root` and `gromacs_root` refer to source code directory of deepmd-kit and gromacs respectively. And `version` represents the version of gromacs, **only support 2020.2 now**. You may patch another version of gromacs but still setting `version` to `2020.2`. However, we cannot ensure that it works.
## 2. Install C++ api of deepmd-kit and tensorflow
The C++ api of `deepmd-kit 2.x` and `tensorflow 2.x` are required.
+ Tips: C++ api of deepmd and tensorflow could be easily installed from the deepmd-kit offline packages. But before using tensorflow, you need to manually change the protobuf package to [version 3.9.2](https://github.com/protocolbuffers/protobuf/releases?after=v3.11.2) in `$deepmd_env_dir/include/google/protobuf` (the offline package will install a version of 3.14, which will cause incompability). 

## 3. Compile GROMACS with deepmd-kit
Specify the installation path of tensorflow and deepmd using cmake options: `-DGMX_TENSORFLOW_ROOT` and `-DGMX_DEEPMD_ROOT`. Here is a sample compile scipt:
```bash
#!/bin/bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CMAKE_PREFIX_PATH="/path/to/fftw-3.3.9" # fftw libraries
mkdir build
cd build
TENSORFLOW_ROOT="/path/to/tensorflow"
DEEPMD_ROOT="/path/to/tensorflow"

cmake3 .. -DCMAKE_CXX_STANDARD=14 \ # not required, but c++14 seems to be more compatible with higher version of tensorflow
          -DGMX_TENSORFLOW_ROOT=${TENSORFLOW_ROOT} \
          -DGMX_DEEPMD_ROOT=${DEEPMD_ROOT} \
          -DGMX_MPI=ON -DGMX_GPU=CUDA \
          -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda \
          -DCMAKE_INSTALL_PREFIX=/path/to/gromacs-2020.2-deepmd-v3
make -j
make install
```
