set -e

# You need to first run ./build_cc.sh

if [ -z "$FLOAT_PREC" ]
then
  FLOAT_PREC=high
fi
#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]
then
  INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing LAMMPS to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------
# copy lammps plugin
BUILD_TMP_DIR2=${SCRIPT_PATH}/../build
cd ${BUILD_TMP_DIR2}
make lammps

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_lammps
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
# download LAMMMPS
LAMMPS_VERSION=stable_29Oct2020
if [ ! -d "lammps-${LAMMPS_VERSION}" ]
then
	curl -L -o lammps.tar.gz https://github.com/lammps/lammps/archive/refs/tags/${LAMMPS_VERSION}.tar.gz
	tar vxzf lammps.tar.gz
fi
curl -L -o lammps.patch https://github.com/deepmd-kit-recipes/lammps-dp-feedstock/raw/fdd954a1af4fadabe5c0dd2f3bed260a484175a4/recipe/deepmd.patch
cd ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}
patch -f -p1 < ../lammps.patch || true 
mkdir -p ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/src/USER-DEEPMD
cp -r ${BUILD_TMP_DIR2}/USER-DEEPMD/* ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/src/USER-DEEPMD

mkdir -p ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/build
cd ${BUILD_TMP_DIR}/lammps-${LAMMPS_VERSION}/build
if [ ${FLOAT_PREC} == "high" ]; then
    export PREC_DEF="-DHIGH_PREC"
fi
cmake -C ../cmake/presets/all_off.cmake -D PKG_USER-DEEPMD=ON -D PKG_KSPACE=ON -D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -D CMAKE_CXX_FLAGS="${PREC_DEF} -I${INSTALL_PREFIX}/include -L${INSTALL_PREFIX}/lib -Wl,--no-as-needed -lrt -ldeepmd_op -ldeepmd -ldeepmd_cc -ltensorflow_cc -ltensorflow_framework -Wl,-rpath=${INSTALL_PREFIX}/lib" ../cmake

make -j${NPROC}
make install

#------------------
echo "Congratulations! LAMMPS has been installed at ${INSTALL_PREFIX}"

