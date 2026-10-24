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
	# Copies of the multi-rank DPA3 archives of gen_dpa3.py and gen_spin.py
	# whose nested with-comm artifact is corrupt; without them the load-failure
	# cases of the pt_expt backends (test_with_comm_load_failure_ptexpt.cc)
	# skip.
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_corrupt_with_comm.py

	# Native-spin DPA4 graph archives (baseline + model-level pair
	# exclusion). Without this the whole native-spin graph C++ suite
	# GTEST_SKIPs on the missing fixture, which is how a dead
	# applyPairExclusion seam in DeepSpinPTExpt went unnoticed.
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4_spin.py &
	PID11=$!
	# DPA4 + analytical ZBL bridging: a linear COMPOSITION on the graph
	# lower, which no other C++ fixture covers.
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4_zbl.py &
	PID12=$!
	# Native-spin DPA4 + charge-spin FiLM: the ONLY fixture with both
	# is_spin=true and dim_chg_spin>0, i.e. the only one on which the
	# runtime charge_spin argument of DeepSpin::compute is not inert.
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4_spin_chgspin.py &
	PID13=$!
	# Native-spin DPA4 + ZBL bridging COMBINED: spin reaching a linear
	# composition, and the only fixture pinning that a bridged model
	# exports NO with-comm artifact (single-rank only by construction).
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4_spin_zbl.py &
	PID14=$!
	# Minimal native-spin DPA4C for LAMMPS pair_style dpa4spin (no
	# fparam/aparam/chg_spin). CPU-safe; used by the magnetic-force
	# convention behavioral tests.
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4c_spin_lmp.py &
	PID14B=$!
	# Native-spin DPA4C: the archives above all ship a with-comm artifact
	# and are therefore served by DeepSpinPTExpt, which leaves
	# NativeSpinPTExpt -- the class the dispatch selects for a native-spin
	# archive without one -- uncovered. DPA4C keeps its messaging local and
	# so reaches it. The compiled graph calls CUDA-only operators, so the
	# fixture is generated only where the C++ suite has a GPU; the tests
	# built on it skip when it is absent.
	if [ "${DP_VARIANT}" = "cuda" ]; then
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4c_spin.py &
		PID15=$!
		# Compressed DPA4C, an energy and a native-spin model, on the compact
		# canonical lower: the only fixtures that reach the device-resident
		# entry points LAMMPS + Kokkos calls, with the replay of repeated
		# evaluations and a runtime charge state.
		env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4c_canonical.py &
		PID16=$!
	fi
	# DPA4C with a latent-charge long-range module: the uncompressed archive
	# takes the all-pairs topology of the graph lower; for the CUDA variant the
	# script also writes the compressed archive, whose compact canonical lower
	# reads the node positions.
	env ${_GEN_ENV} python ${INFER_SCRIPT_PATH}/gen_dpa4c_long_range.py &
	PID16=$!
	wait $PID11
	wait $PID12
	wait $PID13
	wait $PID14
	wait $PID14B
	if [ "${DP_VARIANT}" = "cuda" ]; then
		wait $PID15
		wait $PID16
	fi
	wait $PID16
fi
if [ "${ENABLE_PADDLE:-TRUE}" == "TRUE" ]; then
	PADDLE_INFERENCE_DIR=${BUILD_TMP_DIR}/paddle_inference_install_dir
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib:${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib
fi
ctest --output-on-failure
