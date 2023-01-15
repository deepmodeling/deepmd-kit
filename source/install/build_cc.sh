set -e

if [ "$DP_VARIANT" = "cuda" ]
then
  CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]
then
  CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]
then
  INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
if [ -z "$DOWNLOAD_TENSORFLOW" ]
then
  DOWNLOAD_TENSORFLOW=TRUE
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DINSTALL_TENSORFLOW=${DOWNLOAD_TENSORFLOW} ${CUDA_ARGS} -DLAMMPS_VERSION=stable_23Jun2022_update2 ..
make -j${NPROC}
make install

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"
