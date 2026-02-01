#!/bin/bash
set -ex

if [ "$DP_VARIANT" = "cuda" ]; then
	CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]; then
	CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

echo "try to find tensorflow in the Python environment"
INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
PADDLE_INFERENCE_DIR=${BUILD_TMP_DIR}/paddle_inference_install_dir
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake \
	-D ENABLE_TENSORFLOW=${ENABLE_TENSORFLOW:-TRUE} \
	-D ENABLE_PYTORCH=${ENABLE_PYTORCH:-TRUE} \
	-D ENABLE_PADDLE=${ENABLE_PADDLE:-TRUE} \
	-D INSTALL_TENSORFLOW=FALSE \
	-D USE_TF_PYTHON_LIBS=${ENABLE_TENSORFLOW:-TRUE} \
	-D USE_PT_PYTHON_LIBS=${ENABLE_PYTORCH:-TRUE} \
	-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-D BUILD_TESTING:BOOL=TRUE \
	-D LAMMPS_VERSION=stable_22Jul2025_update2 \
	${CUDA_ARGS} ..
cmake --build . -j${NPROC}
cmake --install .
if [ "${ENABLE_PADDLE:-TRUE}" == "TRUE" ]; then
	PADDLE_INFERENCE_DIR=${BUILD_TMP_DIR}/paddle_inference_install_dir
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib:${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib
fi
ctest --output-on-failure
