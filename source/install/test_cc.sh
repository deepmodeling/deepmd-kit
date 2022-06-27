set -e

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake ../lib/tests -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
make -j${NPROC}
make install

#------------------
${INSTALL_PREFIX}/bin/runUnitTests


#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_cc_tests
INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test_cc
mkdir -p ${BUILD_TMP_DIR}
mkdir -p ${INSTALL_PREFIX}
cd ${BUILD_TMP_DIR}
cmake -DINSTALL_TENSORFLOW=TRUE -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ../api_cc/tests
make -j${NPROC}
make install

#------------------
cd ${SCRIPT_PATH}/../api_cc/tests
${INSTALL_PREFIX}/bin/runUnitTests


