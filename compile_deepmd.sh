rm -rf /home/deepmdroot/ && mkdir /home/deepmdroot && deepmd_root=/home/deepmdroot
cd /home/paddle-deepmd/source && rm -rf build && mkdir build && cd build
#cmake -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DPADDLE_ROOT=/home/Paddle/build/paddle_inference_install_dir -DUSE_CUDA_TOOLKIT=FALSE ..
cmake -DPADDLE_ROOT=/home/Paddle/build/paddle_inference_install_dir -DUSE_CUDA_TOOLKIT=FALSE ..
make -j 4 && make install
make lammps

