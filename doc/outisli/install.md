鉴于大家可能觉得从源码安装`DeePMD-kit`门槛较高，而极少使用。然而从源码安装的灵活性最高，为进一步推广，并减少可能的坑，笔者在此根据自己的安装流程结合官方文档给出一个适用性较广的安装教程，各位可自行尝试。

本教程适用于 Linux(with NVIDIA GPU) 及 Mac(with Apple Silicon)

Since some users may find installing `DeePMD-kit` from source to be challenging and rarely attempt it, this guide aims to make the process more accessible. Installing from source offers the highest flexibility. To promote this method and reduce potential pitfalls, I have compiled a broadly applicable installation tutorial based on my own experience and the official documentation. You are encouraged to try it out.

This tutorial is applicable to Linux (with NVIDIA GPU) and Mac (with Apple Silicon).

> 注：
>
> 1. 安装过程不强制要求`sudo`权限
> 2. 若在有`sudo`权限的电脑上，可自行安装 CUDA Toolkit 以及 mpi（可选）
> 3. 在 HPC 集群上可通过`source\module`方式加载 CUDA Toolkit 以及 mpi 环境
> 4. 默认安装在用户 home 目录 Software 目录下，若需要修改路径，请修改教程中涉及路径的命令
> 5. 本教程需有一定计算机（linux）操作常识，若遇到问题，可以评论沟通或询问 AI

> Notes:
>
> 1. The installation process does not strictly require `sudo` privileges.
> 2. If you have `sudo` privileges, you may install CUDA Toolkit and MPI (optional) yourself.
> 3. On HPC clusters, you can load CUDA Toolkit and MPI environments using `source` or `module` commands.
> 4. By default, the installation path is set to the user's home directory under the Software folder. If you wish to change the path, please modify the relevant commands in the tutorial.
> 5. This tutorial assumes some basic knowledge of computer (Linux) operations. If you encounter any issues, feel free to comment or ask AI for help.

# 0. Preparation (Optional)

## 0.1 CUDA Toolkit

```shell
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
# for Ubuntu 24.04 LTS
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
# for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y
sudo apt install cuda-toolkit-12-8 -y

#config cuda
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64
```

You can also use CUDA 12.6, 12.9 as well.

## 0.2 Intel® oneAPI Toolkit

```shell
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/e6ff8e9c-ee28-47fb-abd7-5c524c983e1c/l_BaseKit_p_2024.2.1.100_offline.sh
sudo sh ./l_BaseKit_p_2024.2.1.100_offline.sh -a --silent --cli --eula accept

wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d461a695-6481-426f-a22f-b5644cd1fa8b/l_HPCKit_p_2024.2.1.79_offline.sh
sudo sh ./l_HPCKit_p_2024.2.1.79_offline.sh -a --silent --cli --eula accept

# load intel oneapi
source /opt/intel/oneapi/setvars.sh --force > /dev/null
```

# 1. Install Backend’s Python interface

## 1.1 Use Miniforge (Conda/mamba)

```shell
# 0. no need for HPC
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y

# 1. Preparation
# 1.1 Get source code
# or git clone https://github.com/deepmodeling/deepmd-kit.git && cd deepmd-kit && git checkout devel
# 下述链接为笔者自己的fork，时不时增加一些小改进，欢迎star
git clone git@github.com:OutisLi/deepmd-kit.git && cd deepmd-kit && git checkout outisli

# 1.2 Create virtual environment
# optional if you installed miniforge: alias conda="mamba"
conda update -n base -c conda-forge conda -y ; conda update -n base -c conda-forge mamba -y
conda deactivate && conda env remove -n dpmd -y ; rm -rf build ; git clean -xdf ; conda create -n dpmd gcc=13 gxx=13 cmake python=3.13 -c conda-forge -y && conda activate dpmd && pip install --upgrade pip && pip install uv

# 1.3 (Optional) install openmpi if you do not have mpi
conda install openmpi -c conda-forge

# 2.1 Install pytorch
uv pip install -U torch --index-url https://download.pytorch.org/whl/cu129

# 2.2 (Optional) Install tensorflow
uv pip install -U tensorflow

# 2.3 (Optional) Install jax
uv pip install -U jax-ai-stack "jax[cuda]"

# 3. Install deepmd-kit
export DP_VARIANT="cuda" DP_ENABLE_PYTORCH=1 DP_ENABLE_TENSORFLOW=1 DP_ENABLE_PADDLE=0 DP_ENABLE_NATIVE_OPTIMIZATION=1 CUDAToolkit_ROOT="/usr/local/cuda" CUDA_HOME="/usr/local/cuda" CUDA_PATH="/usr/local/cuda" CUDA_VERSION=12.9 && pip install -e . -v

# 4. Install other useful packages
uv pip install -U dpdata pymatgen freud-analysis tensorboard torch-tb-profiler seaborn ipykernel nglview git+https://gitlab.com/1041176461/ase-abacus.git
```

## 1.2 For Mac

```shell
# 1. Preparation
# 1.1 Get source code
# or git clone https://github.com/deepmodeling/deepmd-kit.git
git clone git@github.com:OutisLi/deepmd-kit.git && cd deepmd-kit && git checkout outisli
# 1.2 Create virtual environment
conda update -n base -c conda-forge conda -y ; conda update -n base -c conda-forge mamba -y
conda deactivate && conda env remove -n dpmd -y ; rm -rf build ; git clean -xdf ; mamba create -n dpmd compilers llvm-openmp python=3.13 -c conda-forge -y && mamba activate dpmd && pip install --upgrade pip && pip install uv

# 2. Install pytorch
uv pip install -U torch

# 3. Install deepmd-kit
export DP_ENABLE_PYTORCH=1 DP_ENABLE_PADDLE=0 DP_ENABLE_TENSORFLOW=0 DP_ENABLE_NATIVE_OPTIMIZATION=1 && uv pip install -e . -v

# 4. Install other useful packages
uv pip install -U dpdata pymatgen freud-analysis tensorboard torch-tb-profiler seaborn ipykernel nglview git+https://gitlab.com/1041176461/ase-abacus.git
```

# 2. Install the C++ interface

If one does not need to use DeePMD-kit with LAMMPS or i-PI, then the python interface installed in the previous section does everything and he/she can safely skip this section.

```shell
# 0. (Optional) for reinstall
export software="$HOME/Software"
rm -rfv $software/deepmd-kit_cpp $software/deepmd-kit/source/build

# 1. Environment Variables
export deepmd_source_dir=$(pwd) && mkdir -p ../deepmd-kit_cpp && cd ../deepmd-kit_cpp && export deepmd_root=$(pwd) && cd ../deepmd-kit && cd source && mkdir -p build && cd build
# export deepmd_source_dir="$software/deepmd-kit"
# export deepmd_root="$software/deepmd-kit_cpp"

# 2. CMake (Choice either one)
# 2.1 Option 1: use pytorch only
cmake -DENABLE_PYTORCH=ON -DUSE_PT_PYTHON_LIBS=ON -DUSE_CUDA_TOOLKIT=ON -DENABLE_NATIVE_OPTIMIZATION=ON -DCMAKE_INSTALL_PREFIX=$deepmd_root -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..

# 2.2 Option 2: use pytorch & tensorflow & jax
cmake -DENABLE_TENSORFLOW=ON -DUSE_TF_PYTHON_LIBS=ON -DENABLE_PYTORCH=ON -DUSE_PT_PYTHON_LIBS=ON -DENABLE_JAX=ON -DUSE_CUDA_TOOLKIT=ON -DENABLE_NATIVE_OPTIMIZATION=ON -DCMAKE_INSTALL_PREFIX=$deepmd_root -DCMAKE_PREFIX_PATH=$CONDA_PREFIX ..

# 3. Install
make -j && make install
```

# 3. Install LAMMPS’s DeePMD-kit module (built-in mode)

_Before following this section, [DeePMD-kit C++ interface](https://docs.deepmodeling.com/projects/deepmd/en/master/install/install-from-source.html) should have be installed_ (see 3.3)

```shell
# 0.
export software="$HOME/Software"
make lammps && rm -rf $software/lammps

# 1. Install requirements
# Or conda install
mamba install jpeg libpng zlib -c conda-forge -y

# 2. Download LAMMPS
cd $software && mkdir -p lammps && cd lammps && wget https://gh-proxy.com/github.com/lammps/lammps/archive/stable_22Jul2025.tar.gz && tar xzf stable_22Jul2025.tar.gz && cd lammps-stable_22Jul2025 && mkdir -p build && cd build
# wget https://github.com/lammps/lammps/archive/stable_22Jul2025.tar.gz

# 3. Compile
echo "include($deepmd_source_dir/source/lmp/builtin.cmake)" >> ../cmake/CMakeLists.txt && export TORCH_CMAKE_DIR=$(python -c "import torch; print(torch.utils.cmake_prefix_path)") && export TF_LIB_PATH=$(find $CONDA_PREFIX -name "libtensorflow_framework.so.2" | xargs dirname)
cmake -DLAMMPS_INSTALL_RPATH=ON -DBUILD_SHARED_LIBS=yes -DCMAKE_INSTALL_PREFIX=$deepmd_root -DCMAKE_PREFIX_PATH="$deepmd_root;$CONDA_PREFIX;$TORCH_CMAKE_DIR" -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$TF_LIB_PATH" ../cmake
make -j && make install

# test
$deepmd_root/bin/lmp -h
```

# 4. DPGEN2

```shell
# alias conda="mamba"
export software="$HOME/Software"
cd $software
git clone git@github.com:OutisLi/dpgen2.git
cd dpgen2 && conda activate dpmd && pip install uv dpdispatcher && uv pip install -e . -v
```
