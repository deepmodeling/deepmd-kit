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

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DUSE_TF_PYTHON_LIBS=TRUE ${CUDA_ARGS} -DLAMMPS_VERSION=stable_2Aug2023_update1 ..
cmake --build . -j${NPROC}
cmake --install .

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"
