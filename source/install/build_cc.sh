set -e

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
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DFLOAT_PREC=${FLOAT_PREC} -DINSTALL_TENSORFLOW=TRUE ..
make -j${NPROC}
make install

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"

