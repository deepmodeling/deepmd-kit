#!/bin/bash
set -ex

# Phase gating (default: run everything, so existing callers like test_cc.yml
# are unaffected). The CUDA workflow splits the CPU-bound build+fixture-gen
# from the GPU-bound ctest so the former can overlap the Python GPU tests:
#   DP_CC_SKIP_CTEST=1 -> configure + build + install + gen fixtures, no ctest
#   DP_CC_SKIP_BUILD=1 -> skip build/gen, run only ctest on the built tree
DP_CC_SKIP_BUILD=${DP_CC_SKIP_BUILD:-0}
DP_CC_SKIP_CTEST=${DP_CC_SKIP_CTEST:-0}

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

# LD_LIBRARY_PATH additions needed by BOTH the gen scripts (which import
# deepmd.pt and dlopen the custom op .so that depends on libdeepmd.so in the
# install prefix) AND ctest. Set once up front so either phase works alone.
# The install prefix may not exist yet during the build; a missing dir in
# LD_LIBRARY_PATH is harmless.
export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
if [ "${ENABLE_PADDLE:-TRUE}" == "TRUE" ]; then
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib:${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib
fi

if [ "${DP_CC_SKIP_BUILD}" != "1" ]; then
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
		# loads it via cxx_op.py.
		#
		# The install MUST be atomic (copy to a temp file in the same dir, then
		# os.replace). shutil.copy2 overwrites the destination inode IN PLACE,
		# which corrupts the mmap'd code pages of any process that already
		# dlopen'd this .so -- e.g. the concurrent Python test lane in the CUDA
		# CI overlap -- and SIGSEGVs it (see #5882). os.replace swaps the
		# directory entry to a NEW inode instead: live mappings keep the old
		# (refcounted) inode intact, and new dlopens (the gen_*.py scripts) pick
		# up the new file.
		python -c '
import os, shutil, sys
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
    tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
    shutil.copy2(str(so), str(tmp))
    os.replace(str(tmp), str(dst))
    print(f"Installed {so} -> {dst}")
'
		# When the build uses -fsanitize=leak, the custom op .so requires the LSAN
		# runtime to be preloaded (otherwise dlopen fails).  We disable leak detection
		# in the gen scripts to avoid false reports from torch/paddle internals.
		INFER_SCRIPT_PATH=${SCRIPT_PATH}/../tests/infer
		# Remove stale generated model files so they can't be accidentally reused
		# if gen scripts change format or the code version changes.
		rm -f ${INFER_SCRIPT_PATH}/*.pt2 ${INFER_SCRIPT_PATH}/*.pte
		_GEN_ENV=""
		if echo "${CXXFLAGS:-}" | grep -q fsanitize=leak; then
			_LSAN_LIB=$(gcc -print-file-name=liblsan.so 2>/dev/null || true)
			if [ -n "${_LSAN_LIB}" ] && [ -f "${_LSAN_LIB}" ]; then
				# DP_GEN_UNDER_SANITIZER: explicit signal for gen scripts that need
				# to skip sanitizer-incompatible sections (e.g. gen_dpa2.py's
				# AOTInductor graph .pt2 eval, which can SEGV under the LSAN
				# runtime). Sniffing LD_PRELOAD inside the gen script is NOT
				# reliable: the sanitizer runtime removes its own entry from the
				# process environment during startup.
				_GEN_ENV="LD_PRELOAD=${_LSAN_LIB} LSAN_OPTIONS=detect_leaks=0 DP_GEN_UNDER_SANITIZER=lsan"
			fi
		fi
		# Run gen scripts in parallel for faster model generation.
		# Wait on each PID separately so any failure is caught by set -e.
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_sea.py &
		PID1=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa1.py &
		PID2=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa2.py &
		PID3=$!
		wait $PID1
		wait $PID2
		wait $PID3

		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa3.py &
		PID4=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_fparam_aparam.py &
		PID5=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_model_devi.py &
		PID6=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_chg_spin.py &
		PID9=$!
		wait $PID4
		wait $PID5
		wait $PID6
		wait $PID9

		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4.py &
		PID9=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa1_pairexcl.py &
		PID10=$!
		wait $PID9
		wait $PID10

		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_spin.py &
		PID7=$!
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_spin_model_devi.py &
		PID8=$!
		wait $PID7
		wait $PID8
	fi
fi

if [ "${DP_CC_SKIP_CTEST}" != "1" ]; then
	ctest --output-on-failure
fi
