#!/bin/bash
set -ex

if [ "$DP_VARIANT" = "cuda" ]; then
	CUDA_ARGS="-DUSE_CUDA_TOOLKIT=TRUE"
elif [ "$DP_VARIANT" = "rocm" ]; then
	CUDA_ARGS="-DUSE_ROCM_TOOLKIT=TRUE"
fi

#------------------

SCRIPT_PATH=$(dirname $(realpath -s $0))
NPROC=$(nproc --all)

#------------------

echo "try to find tensorflow in the Python environment"
INSTALL_PREFIX=${SCRIPT_PATH}/../../dp_test
BUILD_TMP_DIR=${SCRIPT_PATH}/../build_tests
PADDLE_INFERENCE_DIR=${BUILD_TMP_DIR}/paddle_inference_install_dir
mkdir -p ${BUILD_TMP_DIR}
cd ${BUILD_TMP_DIR}
cmake \
	-D ENABLE_TENSORFLOW=${ENABLE_TENSORFLOW:-TRUE} \
	-D ENABLE_PYTORCH=${ENABLE_PYTORCH:-TRUE} \
	-D ENABLE_PADDLE=${ENABLE_PADDLE:-TRUE} \
	-D INSTALL_TENSORFLOW=FALSE \
	-D USE_TF_PYTHON_LIBS=${ENABLE_TENSORFLOW:-TRUE} \
	-D USE_PT_PYTHON_LIBS=${ENABLE_PYTORCH:-TRUE} \
	-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
	-D BUILD_TESTING:BOOL=TRUE \
	-D LAMMPS_VERSION=stable_22Jul2025_update2 \
	${CUDA_ARGS} ..
cmake --build . -j${NPROC}
cmake --install .
# Generate PT/PT2 model files for C++ tests.
# Must run after cmake --build so that libdeepmd_op_pt.so (custom ops) is available.
if [ "${ENABLE_PYTORCH:-TRUE}" == "TRUE" ]; then
	# Install the custom op .so to SHARED_LIB_DIR so that `import deepmd.pt`
	# loads it via cxx_op.py.  The .so depends on libdeepmd.so (compute
	# kernels) in the install prefix, so add that to LD_LIBRARY_PATH too.
	export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
	python -c '
import shutil, sys
from pathlib import Path
from deepmd.env import SHARED_LIB_DIR
so = Path("'"${BUILD_TMP_DIR}"'") / "op" / "pt" / "libdeepmd_op_pt.so"
dst = SHARED_LIB_DIR / so.name
if not so.exists():
    print(f"WARNING: {so} not found, custom ops will not be available", file=sys.stderr)
elif dst.exists() and dst.resolve() == so.resolve():
    print(f"Already linked: {dst} -> {so}")
else:
    SHARED_LIB_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(so), str(dst))
    print(f"Installed {so} -> {dst}")
'
	# When the build uses -fsanitize=leak, the custom op .so requires the LSAN
	# runtime to be preloaded (otherwise dlopen fails).  We disable leak detection
	# in the gen scripts to avoid false reports from torch/paddle internals.
	INFER_SCRIPT_PATH=${SCRIPT_PATH}/../tests/infer
	_GEN_ENV=""
	if echo "${CXXFLAGS:-}" | grep -q fsanitize=leak; then
		_LSAN_LIB=$(gcc -print-file-name=liblsan.so 2>/dev/null || true)
		if [ -n "${_LSAN_LIB}" ] && [ -f "${_LSAN_LIB}" ]; then
			_GEN_ENV="LD_PRELOAD=${_LSAN_LIB} LSAN_OPTIONS=detect_leaks=0"
		fi
	fi
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_sea.py
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa1.py
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa2.py
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa3.py
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_fparam_aparam.py
fi
if [ "${ENABLE_PADDLE:-TRUE}" == "TRUE" ]; then
	PADDLE_INFERENCE_DIR=${BUILD_TMP_DIR}/paddle_inference_install_dir
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib:${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib
fi
ctest --output-on-failure
