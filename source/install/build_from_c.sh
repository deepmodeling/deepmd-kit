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
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DDEEPMD_C_ROOT=${DEEPMD_C_ROOT} -DLAMMPS_VERSION=stable_22Jul2025_update2 ..
cmake --build . -j${NPROC}
cmake --install .

CONSUMER_TMP_DIR=${BUILD_TMP_DIR}/cmake-consumer
rm -rf ${CONSUMER_TMP_DIR}
mkdir -p ${CONSUMER_TMP_DIR}
cat >${CONSUMER_TMP_DIR}/CMakeLists.txt <<'EOF'
cmake_minimum_required(VERSION 3.25)
project(deepmd_c_consumer LANGUAGES CXX)
find_package(DeePMD REQUIRED CONFIG)
if(NOT TARGET DeePMD::deepmd_c)
  message(FATAL_ERROR "DeePMD::deepmd_c target is missing")
endif()
add_executable(deepmd_c_consumer main.cc)
target_link_libraries(deepmd_c_consumer PRIVATE DeePMD::deepmd_c)
EOF
cat >${CONSUMER_TMP_DIR}/main.cc <<'EOF'
#include <deepmd/c_api.h>

int main() { return 0; }
EOF
cmake -S ${CONSUMER_TMP_DIR} -B ${CONSUMER_TMP_DIR}/build -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}
cmake --build ${CONSUMER_TMP_DIR}/build -j${NPROC}

cmake --build . --target=lammps

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"
