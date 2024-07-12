set -e

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
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DDEEPMD_C_ROOT=${DEEPMD_C_ROOT} -DLAMMPS_VERSION=stable_2Aug2023_update3 ..
cmake --build . -j${NPROC}
cmake --install .
cmake --build . --target=lammps

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"
