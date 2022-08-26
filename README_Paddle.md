# 1.Introduction
This repo is based on the PaddlePaddle deep learning framework including training and inference parts，DeepMD-kit package，LAMMPS software. The target is, basing on PaddlePaddle framework, to accomplish molecular dynamics simulation with deep learning method.
- PaddlePaddle (PArallel Distributed Deep LEarning) is a simple, efficient and extensible deep learning framework.
- DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD).
- LAMMPS is a classical molecular dynamics code with a focus on materials modeling. It's an acronym for Large-scale Atomic/Molecular Massively Parallel Simulator.

# 2.Progress&Features
- Based on Intel CPU, the pipline of training and inference runs smoothly
- Support traditional molecular dynamics software LAMMPS
- Support se_a desciptor model

# 3.Compiling&Building&Installation
- prepare docker and python environment
```
docker pull paddlepaddle/paddle:latest-dev-cuda11.0-cudnn8-gcc82 
docker run -it --name {name} -v 绝对路径开发目录:绝对路径开发目录 -v /root/.cache:/root/.cache -v /root/.ccache:/root/.ccache {image_id} bash 
rm -f /usr/bin/python3
ln -s /usr/bin/python3.8 /usr/bin/python3
wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz && tar -xf cmake-3.21.0-linux-x86_64.tar.gz
add ~/.bashrc：export PATH=/home/cmake-3.21.0-linux-x86_64/bin:$PATH
```

- compile Paddle
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle  
git reset --hard eca6638c599591c69fe40aa196f5fd42db7efbe2  
rm -rf build && mkdir build && cd build  
cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=OFF -DWITH_AVX=ON -DON_INFER=ON -DCMAKE_BUILD_TYPE=Release  
# cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=On -DWITH_AVX=ON -DON_INFER=ON -DCUDA_ARCH_NAME=Auto -DCMAKE_BUILD_TYPE=Release
make -j 32  
make -j 32 inference_lib_dist  
python3 -m pip install python/dist/paddlepaddle-0.0.0-cp38-cp38-linux_x86_64.whl --no-cache-dir
PADDLE_ROOT=/home/Paddle/build/paddle_inference_install_dir(or add in bashrc with export)
```
- compile Paddle_DeepMD-kit --training part 
```
cd /home
git clone https://github.com/X4Science/paddle-deepmd.git
cd /home/paddle-deepmd
python3 -m pip install tensorflow-cpu==2.5.0
# python3 -m pip install tensorflow-gpu==2.5.0
python3 -m pip install scikit-build
python3 setup.py install
find the package name of deepmd-kit in the location of installation and add in bashrc
        export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/**{deepmd_name}**/deepmd/op:$LD_LIBRARY_PATH
        export LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/**{deepmd_name}**/deepmd/op:$LIBRARY_PATH
        export DEEP_MD_PATH=/usr/local/lib/python3.8/dist-packages/**{deepmd_name}**/deepmd/op
source ~/.bashrc
cd deepmd && python3 load_paddle_op.py install
```

- compile Paddle_DeepMD-kit --inference part 
```
rm -rf /home/deepmdroot/ && mkdir /home/deepmdroot && DEEPMD_ROOT=/home/deepmdroot(or add in bashrc with export)
cd /home/paddle-deepmd/source && rm -rf build && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$DEEPMD_ROOT -DPADDLE_ROOT=$paddle_root -DUSE_CUDA_TOOLKIT=FALSE -DFLOAT_PREC=low ..
make -j 4 && make install
make lammps
```
- compile LAMMPS
```
cd /home
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7
./configure
make all install
add bashrc: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
#apt install libc-dev
cd /home
wget https://github.com/lammps/lammps/archive/stable_29Oct2020.tar.gz
rm -rf lammps-stable_29Oct2020/
tar -xzvf stable_29Oct2020.tar.gz
cd lammps-stable_29Oct2020/src/
cp -r /home/paddle-deepmd/source/build/USER-DEEPMD .
make yes-kspace yes-user-deepmd
#make serial -j 20
make mpi -j 20
add in bashrc by
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/paddle/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle_inference_install_dir/third_party/install/mklml/lib:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/Paddle/build/paddle/fluid/pybind/:$LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=/home/deepmd-kit/source/build:$LD_LIBRARY_PATH
source ~/.bashrc
```
# 4.Using Guide
example: water
- training
```
cd /paddle_deepmd-kit_PATH/example/water/train/
dp train water_se_a.json
cp ./model.ckpt/model.pd* ../lmp/ -r
cd ../lmp
```
- inference
```
mpirun -np 10 lmp_mpi -in in.lammps
```


# 5.Performance
- The performance of inference based on the LAMMPS with PaddlePaddle framework，comparing with TensorFlow framework, about single core and multi-threads
![截屏2022-05-25 23 08 11](https://user-images.githubusercontent.com/50223303/170295703-32e18058-aff9-4368-93cd-38a1ed787e8a.png)
- single thread performance
```
TF_INTRA_OP_PARALLELISM_THREADS=8 TF_INTER_OP_PARALLELISM_THREADS=1 numactl -c 0 -m 0 lmp_serial -in in.lammps
```
- multithreads performance
```
OMP_NUM_THREADS=1 TF_INTRA_OP_PARALLELISM_THREADS=1 TF_INTER_OP_PARALLELISM_THREADS=1  mpirun --allow-run-as-root -np 4 lmp_mpi -in in.lammps
```  
# 6.Future Plans
- fix training precision
- support Gromacs
- support more descriptor and fitting net model
- support GPU trainning

# 7.Cooperation
Welcome to join us to develop this program together.  
Please contact us from [X4Science](https://github.com/X4Science) [PaddlePaddle](https://www.paddlepaddle.org.cn) [PPSIG](https://www.paddlepaddle.org.cn/sig) [PaddleAIforScience](https://www.paddlepaddle.org.cn/science) [PaddleScience](https://github.com/PaddlePaddle/PaddleScience).
