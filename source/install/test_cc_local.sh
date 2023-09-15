set -e

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

echo "try to find tensorflow in the Python environment"
INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DINSTALL_TENSORFLOW=FALSE -DUSE_TF_PYTHON_LIBS=TRUE -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DBUILD_TESTING:BOOL=TRUE -DLAMMPS_VERSION=stable_2Aug2023 ..
cmake --build . -j${NPROC}
cmake --install .
cmake --build . --target test
