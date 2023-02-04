# package C library into a tarball

set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ -z "$INSTALL_PREFIX" ]; then
	INSTALL_PREFIX=$(realpath -s ${SCRIPT_PATH}/../../dp_c)
fi
mkdir -p ${INSTALL_PREFIX}
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_c
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-DUSE_CUDA_TOOLKIT=TRUE \
	-DOP_CXX_ABI=0 \
	-DPACKAGE_C=TRUE \
	..
make -j${NPROC}
make install

#------------------

tar vczf ${SCRIPT_PATH}/../../libdeepmd_c.tar.gz -C ${BUILD_TMP_DIR} libdeepmd_c
