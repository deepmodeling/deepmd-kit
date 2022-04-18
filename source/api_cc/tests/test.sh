rm -rf build && mkdir build && cd build
cmake .. -DHIHG_PREC=OFF -DPADDLE_ROOT=/home/Paddle/build/paddle_inference_install_dir/
make -j 12
