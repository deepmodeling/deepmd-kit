set -e

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

echo "try to find tensorflow in ${tensorflow_root}"
INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DINSTALL_TENSORFLOW=FALSE -DTENSORFLOW_ROOT=${tensorflow_root} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DBUILD_TESTING:BOOL=TRUE -DLAMMPS_VERSION=stable_23Jun2022_update2 ..
make -j${NPROC}
make install

#------------------
# go to a subdirectory...
# TODO: detect directory of graph files
mkdir -p ${BUILD_TMP_DIR}/exec_tests
cd ${BUILD_TMP_DIR}/exec_tests

${INSTALL_PREFIX}/bin/runUnitTests_lib
${INSTALL_PREFIX}/bin/runUnitTests_cc
${INSTALL_PREFIX}/bin/runUnitTests_c
