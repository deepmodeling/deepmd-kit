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
export CUDA_VISIBLE_DEVICES=1

# deepmd_root 设置为本项目的根目录
export deepmd_root="/workspace/hesensen/deepmd_backend/deepmd_paddle_new/"

# PADDLE_INFERENCE_DIR 设置为第二步编译得到的 Paddle 推理库目录
export PADDLE_INFERENCE_DIR="/workspace/hesensen/PaddleScience_enn_debug/Paddle/build/paddle_inference_install_dir/"

export LD_LIBRARY_PATH=${deepmd_root}/deepmd/op:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PADDLE_INFERENCE_DIR}/paddle/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PADDLE_INFERENCE_DIR}/third_party/install/mkldnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${deepmd_root}/source/build:$LD_LIBRARY_PATH

cd ${deepmd_root}/source
rm -rf build # 若改动CMakeLists.txt，则需要打开该注释
mkdir build
cd -

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -D ENABLE_PADDLE=ON \
	-D PADDLE_INFERENCE_DIR=${PADDLE_INFERENCE_DIR} \
	-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-D USE_TF_PYTHON_LIBS=TRUE \
	-D LAMMPS_SOURCE_ROOT=${LAMMPS_SOURCE_ROOT} \
	${CUDA_ARGS} \
	-D LAMMPS_VERSION=stable_29Aug2024 \
	..
cmake --build . -j${NPROC}
cmake --install .

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"

cd ${deepmd_root}/source
cd build
make lammps
cd ${LAMMPS_DIR}/src/
\cp -r ${deepmd_root}/source/build/USER-DEEPMD .
make no-kspace
make yes-kspace
make no-extra-fix
make yes-extra-fix
make no-user-deepmd
make yes-user-deepmd
# make serial -j
make mpi -j 10
export PATH=${LAMMPS_DIR}/src:$PATH

cd ${deepmd_root}/examples/water/lmp

echo "START INFERENCE..."
# lmp_serial -in paddle_in.lammps 2>&1 | tee paddle_infer.log
mpirun -np 2 lmp_mpi -in paddle_in.lammps 2>&1 | tee paddle_infer.log
