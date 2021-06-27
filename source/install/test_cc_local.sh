set -e

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake ../lib/tests
make -j${NPROC}

#------------------
${BUILD_TMP_DIR}/runUnitTests


#------------------

echo "try to find tensorflow in ${tensorflow_root}"
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_cc_tests
INSTALL_PREFIX=${SCRIPT_PATH}/../../dp
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DINSTALL_TENSORFLOW=FALSE -DTENSORFLOW_ROOT=${tensorflow_root} ../api_cc/tests
make -j${NPROC}

#------------------
cd ${SCRIPT_PATH}/../api_cc/tests
${BUILD_TMP_DIR}/runUnitTests
