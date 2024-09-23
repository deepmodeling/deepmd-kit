set -e

if [ "$DP_VARIANT" = "cuda" ]; then
	CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]; then
	CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]; then
	INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

# LAMMPS_DIR 设置为 LAMMPS 的安装目录
export LAMMPS_DIR="/workspace/hesensen/deepmd_backend/deepmd_paddle_new/source/build_lammps/lammps-stable_29Aug2024/"
export LAMMPS_SOURCE_ROOT="/workspace/hesensen/deepmd_backend/deepmd_paddle_new/source/build_lammps/lammps-stable_29Aug2024/"

# 设置推理时的 GPU 卡号
export CUDA_VISIBLE_DEVICES=3
# export FLAGS_benchmark=1
# export GLOG_v=6

# PADDLE_DIR 设置为第二步 clone下来的 Paddle 目录
export PADDLE_DIR="/workspace/hesensen/PaddleScience_enn_debug/Paddle/"

# DEEPMD_DIR 设置为本项目的根目录
export DEEPMD_DIR="/workspace/hesensen/deepmd_backend/deepmd_paddle_new/"

# PADDLE_INFERENCE_DIR 设置为第二步编译得到的 Paddle 推理库目录
export PADDLE_INFERENCE_DIR="/workspace/hesensen/PaddleScience_enn_debug/Paddle/build/paddle_inference_install_dir/"

# TENSORFLOW_DIR 设置为 tensorflow 的安装目录，可用 pip show tensorflow 确定
# export TENSORFLOW_DIR="/path/to/tensorflow"

export LD_LIBRARY_PATH=${PADDLE_DIR}/paddle/fluid/pybind/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${DEEPMD_DIR}/deepmd/op:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PADDLE_INFERENCE_DIR}/paddle/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PADDLE_INFERENCE_DIR}/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${DEEPMD_DIR}/source/build:$LD_LIBRARY_PATH
export LIBRARY_PATH=${DEEPMD_DIR}/deepmd/op:$LIBRARY_PATH
# export FLAGS_check_nan_inf=1
# cd ${DEEPMD_DIR}/source
# rm -rf build # 若改动CMakeLists.txt，则需要打开该注释
# mkdir build
# cd -

# DEEPMD_INSTALL_DIR 设置为 deepmd-lammps 的目标安装目录，可自行设置任意路径
# export DEEPMD_INSTALL_DIR="path/to/deepmd_root"

# 开始编译
# cmake -DCMAKE_INSTALL_PREFIX=${DEEPMD_INSTALL_DIR} \
#     -DUSE_CUDA_TOOLKIT=TRUE \
#     -DTENSORFLOW_ROOT=${TENSORFLOW_DIR} \
#     -DPADDLE_LIB=${PADDLE_INFERENCE_DIR} \
#     -DFLOAT_PREC=low ..
# make -j4 && make install
# make lammps

# cd ${LAMMPS_DIR}/src/
# \cp -r ${DEEPMD_DIR}/source/build/USER-DEEPMD .
# make yes-kspace
# make yes-extra-fix
# make yes-user-deepmd
# make serial -j
# export PATH=${LAMMPS_DIR}/src:$PATH

# cd ${DEEPMD_DIR}/examples/water/lmp

# lmp_serial -in in.lammps

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DCMAKE_PREFIX_PATH=/workspace/hesensen/PaddleScience_enn_debug/Paddle/build/paddle_inference_install_dir/paddle \
	-D ENABLE_TENSORFLOW=OFF \
	-D ENABLE_PYTORCH=OFF \
	-D ENABLE_PADDLE=ON \
	-D PADDLE_LIB=${PADDLE_INFERENCE_DIR} \
	-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-D USE_TF_PYTHON_LIBS=TRUE \
	-D LAMMPS_SOURCE_ROOT=${LAMMPS_SOURCE_ROOT} \
	-D ENABLE_IPI=OFF \
	-D PADDLE_LIBRARIES=/workspace/hesensen/PaddleScience_enn_debug/Paddle/build/paddle_inference_install_dir/paddle/lib/libpaddle_inference.so \
	${CUDA_ARGS} \
	-D LAMMPS_VERSION=stable_29Aug2024 \
	..
cmake --build . -j${NPROC}
cmake --install .

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"

cd ${DEEPMD_DIR}/source
cd build
make lammps
cd ${LAMMPS_DIR}/src/
\cp -r ${DEEPMD_DIR}/source/build/USER-DEEPMD .
make no-kspace
make yes-kspace
make no-extra-fix
make yes-extra-fix
make no-user-deepmd
make yes-user-deepmd
# make serial -j
make mpi -j 20
export PATH=${LAMMPS_DIR}/src:$PATH

cd ${DEEPMD_DIR}/examples/water/lmp

echo "START INFERENCE..."
# lmp_serial -in paddle_in.lammps 2>&1 | tee paddle_infer.log
mpirun -np 1 lmp_mpi -in paddle_in.lammps 2>&1 | tee paddle_infer.log
