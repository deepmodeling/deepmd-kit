cd Paddle
git reset --hard eca6638c599591c69fe40aa196f5fd42db7efbe2
rm -rf build && mkdir build && cd build
#cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=OFF -DWITH_AVX=ON -DON_INFER=ON -DWITH_MKLDNN=ON -DCMAKE_BUILD_TYPE=Release
cmake .. -DPY_VERSION=3.8 -DPYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") -DPYTHON_LIBRARY=$(python3 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -DWITH_GPU=OFF -DWITH_AVX=ON -DON_INFER=ON -DCMAKE_BUILD_TYPE=Release

export PADDLEPADDLE_TP_CACHE="/home/Paddle/tp_cache"

cp $PADDLEPADDLE_TP_CACHE/boost* third_party/boost/src/ || true
cp $PADDLEPADDLE_TP_CACHE/csrmm_mklml_lnx* third_party/mklml/src/ || true
cp $PADDLEPADDLE_TP_CACHE/lapack_lnx* third_party/lapack/src/ || true

make -j 32

make -j 32 inference_lib_dist

