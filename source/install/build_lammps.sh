set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]; then
	INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing LAMMPS to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_lammps
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
# download LAMMMPS
LAMMPS_VERSION=stable_2Aug2023_update3
if [ ! -d "lammps-${LAMMPS_VERSION}" ]; then
	curl -L -o lammps.tar.gz https://github.com/lammps/lammps/archive/refs/tags/${LAMMPS_VERSION}.tar.gz
	tar vxzf lammps.tar.gz
fi

cd ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}
mkdir -p ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/build
cd ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/build
cmake -C ../cmake/presets/all_off.cmake -D PKG_PLUGIN=ON -D PKG_MOLECULE=ON -DLAMMPS_EXCEPTIONS=yes -D BUILD_SHARED_LIBS=yes -D LAMMPS_INSTALL_RPATH=ON -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -D CMAKE_INSTALL_LIBDIR=lib -D CMAKE_INSTALL_FULL_LIBDIR=${INSTALL_PREFIX}/lib ../cmake

make -j${NPROC}
make install
make install-python

#------------------
echo "Congratulations! LAMMPS has been installed at ${INSTALL_PREFIX}"
