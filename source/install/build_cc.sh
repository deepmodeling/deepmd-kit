set -e

if [ "$DP_VARIANT" == "cuda" ]
then
  CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
fi
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]
then
  INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DINSTALL_TENSORFLOW=TRUE ${CUDA_ARGS} -DLAMMPS_VERSION=stable_29Sep2021 -DUSE_TTM=TRUE ..
make -j${NPROC}
make install

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"
