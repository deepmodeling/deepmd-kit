set -e

if [ "$DP_VARIANT" = "cuda" ]; then
	CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]; then
	CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi
# These selectors are mutually exclusive, but CMake caches them between
# builds. Remove the stale selector, and keep an explicit source path as one
# argument so valid paths containing spaces are not split by the shell.
if [ -n "${LAMMPS_SOURCE_ROOT:-}" ]; then
	set -- -ULAMMPS_VERSION "-DLAMMPS_SOURCE_ROOT=${LAMMPS_SOURCE_ROOT}"
else
	set -- -ULAMMPS_SOURCE_ROOT "-DLAMMPS_VERSION=stable_22Jul2025_update2"
fi
#------------------

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
cmake -D ENABLE_TENSORFLOW=ON \
	-D ENABLE_PYTORCH=ON \
	-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-D USE_TF_PYTHON_LIBS=TRUE \
	-D USE_PT_PYTHON_LIBS=TRUE \
	${CUDA_ARGS} \
	"$@" \
	${CMAKE_ARGS:-} \
	..
cmake --build . -j${NPROC}
cmake --install .

#------------------
echo "Congratulations! DeePMD-kit has been installed at ${INSTALL_PREFIX}"
