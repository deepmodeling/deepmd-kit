rm -rf /home/deepmdroot/ && mkdir /home/deepmdroot && deepmd_root=/home/deepmdroot
cd /home/deepmd-kit/source && rm -rf build && mkdir build && cd build
cmake -DPADDLE_ROOT=/home/Paddle/build/paddle_inference_install_dir -DUSE_CUDA_TOOLKIT=FALSE ..
make -j 4 && make install
make lammps

