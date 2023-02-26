set -e

SCRIPT_PATH=$(dirname $(realpath -s $0))
if [ ! $# -eq 2 ]; then
	echo "${SCRIPT_PATH}: Params error, installation of tensorflow libraries failed!"
	exit 1
fi

PYTHON_SITE_PACKAGE_PATH=$(realpath -s $1)
TENSORFLOW_ROOT=$(realpath -s $2)
TF_INSTALL_PATH=${PYTHON_SITE_PACKAGE_PATH}/tensorflow

if [ ! -d ${TF_INSTALL_PATH} ]; then
	echo "${SCRIPT_PATH}: ${TF_INSTALL_PATH}, TensorFlow not found!"
	exit 1
fi

#----------------------------------------
# check if the installation folders exist
#----------------------------------------
if [ ! -d ${TENSORFLOW_ROOT} ]; then
	mkdir ${TENSORFLOW_ROOT}
fi
if [ ! -d ${TENSORFLOW_ROOT}/include ]; then
	mkdir ${TENSORFLOW_ROOT}/include
fi
if [ ! -d ${TENSORFLOW_ROOT}/lib ]; then
	mkdir ${TENSORFLOW_ROOT}/lib
fi

#----------------------------------------
# install the TF libraries
#----------------------------------------
cp -r ${TF_INSTALL_PATH}/include ${TENSORFLOW_ROOT}
cp ${TF_INSTALL_PATH}/libtensorflow_framework.so* ${TENSORFLOW_ROOT}/lib
cp ${TF_INSTALL_PATH}/python/_pywrap_tensorflow_internal.so ${TENSORFLOW_ROOT}/lib
ln -s ${TENSORFLOW_ROOT}/lib/libtensorflow_framework.so* ${TENSORFLOW_ROOT}/lib/libtensorflow_framework.so
ln -s ${TENSORFLOW_ROOT}/lib/_pywrap_tensorflow_internal.so ${TENSORFLOW_ROOT}/lib/libtensorflow_cc.so
