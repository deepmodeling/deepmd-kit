#!/bin/bash
set -ex

# With no arguments, build, generate every fixture, and run CTest as before.
# The independent Kokkos job needs the installation and its compressed fixtures.
MODE=${1:-all}
if [ "$#" -gt 1 ]; then
	echo "Usage: $0 [--build-only | --models-only=kokkos]" >&2
	exit 2
fi
case "$MODE" in
all | --build-only | --models-only=kokkos) ;;
*)
	echo "Usage: $0 [--build-only | --models-only=kokkos]" >&2
	exit 2
	;;
esac
if [ "$MODE" = "--models-only=kokkos" ]; then
	if [ "${DP_VARIANT:-}" != "cuda" ] || [ "${ENABLE_PYTORCH:-TRUE}" != "TRUE" ]; then
		echo "Kokkos fixtures require DP_VARIANT=cuda and ENABLE_PYTORCH=TRUE." >&2
		exit 1
	fi
fi

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
if [ "$MODE" != "--models-only=kokkos" ]; then
	BUILD_TESTING=TRUE
	if [ "$MODE" = "--build-only" ]; then
		BUILD_TESTING=FALSE
	fi
	cmake \
		-D ENABLE_TENSORFLOW=${ENABLE_TENSORFLOW:-TRUE} \
		-D ENABLE_PYTORCH=${ENABLE_PYTORCH:-TRUE} \
		-D ENABLE_PADDLE=${ENABLE_PADDLE:-TRUE} \
		-D INSTALL_TENSORFLOW=FALSE \
		-D USE_TF_PYTHON_LIBS=${ENABLE_TENSORFLOW:-TRUE} \
		-D USE_PT_PYTHON_LIBS=${ENABLE_PYTORCH:-TRUE} \
		-D CMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
		-D BUILD_TESTING:BOOL=${BUILD_TESTING} \
		-D LAMMPS_VERSION="${LAMMPS_VERSION:-stable_22Jul2025_update2}" \
		${CUDA_ARGS} ..
	cmake --build . -j${NPROC}
	cmake --install .
else
	# Never let a missing installation fall back to an unrelated Python build.
	test -f "${INSTALL_PREFIX}/lib/libdeepmd_c.so"
fi
# Generate PT/PT2 model files for C++ tests.
# Must run after cmake --build so that libdeepmd_op_pt.so (custom ops) is available.
if [ "${ENABLE_PYTORCH:-TRUE}" == "TRUE" ]; then
	# Install the custom op .so to SHARED_LIB_DIR so that `import deepmd.pt`
	# loads it via cxx_op.py.  The .so depends on libdeepmd.so (compute
	# kernels) in the install prefix, so add that to LD_LIBRARY_PATH too.
	export LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH}
	if [ "$MODE" != "all" ]; then
		test -f "${BUILD_TMP_DIR}/op/pt/libdeepmd_op_pt.so"
	fi
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
fi
if [ "$MODE" = "--build-only" ]; then
	exit 0
fi
if [ "${ENABLE_PYTORCH:-TRUE}" == "TRUE" ]; then
	# Keep one timing record per generator, including successful runs. Each
	# invocation owns its directory and each parallel script owns its file.
	mkdir -p "${BUILD_TMP_DIR}/generator-reports"
	GENERATOR_REPORT_DIR=$(mktemp -d "${BUILD_TMP_DIR}/generator-reports/run.XXXXXX")
	run_generator_named() {
		local name=$1
		shift
		local start=$SECONDS status=0
		if env ${_GEN_ENV} python -u "$@"; then
			status=0
		else
			status=$?
		fi
		printf 'generator\tseconds\texit_status\n%s\t%s\t%s\n' \
			"$name" "$((SECONDS - start))" "$status" >"${GENERATOR_REPORT_DIR}/${name}.tsv"
		return "$status"
	}
	run_generator() {
		run_generator_named "$(basename "${1%.py}")" "$@"
	}
	# When the build uses -fsanitize=leak, the custom op .so requires the LSAN
	# runtime to be preloaded (otherwise dlopen fails).  We disable leak detection
	# in the gen scripts to avoid false reports from torch/paddle internals.
	INFER_SCRIPT_PATH=${SCRIPT_PATH}/../tests/infer
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
	if [ "$MODE" = "--models-only=kokkos" ]; then
		# Targeted regeneration invalidates the previous full fixture manifest.
		rm -f "${INFER_SCRIPT_PATH}/.pt-fixture-cache-manifest"
		# Leave the other C++ fixtures alone when reusing an existing build.
		rm -f ${INFER_SCRIPT_PATH}/deeppot_dpa4c_long_range*.pt2 \
			${INFER_SCRIPT_PATH}/deeppot_dpa4c_long_range*.pte \
			${INFER_SCRIPT_PATH}/deeppot_dpa4c_long_range*.expected
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4c_long_range.py --compressed-only
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4c_canonical.py
		exit 0
	fi
	_FIXTURE_MANIFEST="${INFER_SCRIPT_PATH}/.pt-fixture-cache-manifest"
	# Reuse only the complete, content-addressed fixture set restored by CI.
	if [ "${DP_REUSE_GENERATED_PT_MODELS:-0}" = "1" ]; then
		if [ ! -s "${_FIXTURE_MANIFEST}" ]; then
			echo "ERROR: PT fixture cache hit without a non-empty manifest" >&2
			exit 1
		fi
		while IFS= read -r fixture || [ -n "${fixture}" ]; do
			[ -n "${fixture}" ] || continue
			if [ ! -f "${INFER_SCRIPT_PATH}/${fixture}" ]; then
				echo "ERROR: cached PT fixture is missing: ${fixture}" >&2
				exit 1
			fi
		done <"${_FIXTURE_MANIFEST}"
		echo "Reusing $(wc -l <"${_FIXTURE_MANIFEST}") cached PT fixture files"
	else
		# Remove stale generated model files so they can't be accidentally reused
		# if gen scripts change format or the code version changes.
		rm -f ${INFER_SCRIPT_PATH}/*.pt2 ${INFER_SCRIPT_PATH}/*.pte \
			${INFER_SCRIPT_PATH}/*.expected "${_FIXTURE_MANIFEST}"
		# Refill the first three generator slots instead of leaving two idle
		# behind the long DPA2 export. DPA3 and DPA4 have no input dependency
		# on these generators and write distinct fixtures. Keep at most three
		# processes in this phase; do not add both on top of the initial batch.
		# Wait on each PID separately so any failure is caught by set -e.
		run_generator ${INFER_SCRIPT_PATH}/gen_sea.py &
		PID1=$!
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa1.py &
		PID2=$!
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa2.py &
		PID3=$!
		wait $PID1
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4.py &
		PID9=$!
		wait $PID2
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa3.py &
		PID4=$!
		wait $PID3
		wait $PID9
		wait $PID4

		run_generator ${INFER_SCRIPT_PATH}/gen_fparam_aparam.py &
		PID5=$!
		run_generator ${INFER_SCRIPT_PATH}/gen_model_devi.py &
		PID6=$!
		run_generator ${INFER_SCRIPT_PATH}/gen_chg_spin.py &
		PID9=$!
		# DPA1 completed above; its archive-reuse consumer needs no other
		# producer. Fill the fourth slot freed by moving DPA3 into the first phase.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa1_pairexcl.py --reuse-dpa1 &
		PID10=$!
		wait $PID5
		wait $PID6
		wait $PID9
		wait $PID10

		# Keep both producers parallel, but compile the shared DPA1 spin model in
		# model-devi's shorter branch, rather than repeating it on gen_spin's
		# critical path. Finalize only after both producers succeed. Distinct
		# report names retain both phases without overwriting their timings.
		run_generator_named gen_spin_dpa3 ${INFER_SCRIPT_PATH}/gen_spin.py --dpa3-only &
		PID7=$!
		run_generator ${INFER_SCRIPT_PATH}/gen_spin_model_devi.py &
		PID8=$!
		wait $PID7
		wait $PID8
		run_generator_named gen_spin_reuse ${INFER_SCRIPT_PATH}/gen_spin.py --reuse-model-devi
		# Copies of the multi-rank DPA3 archives of gen_dpa3.py and gen_spin.py
		# whose nested with-comm artifact is corrupt; without them the load-failure
		# cases of the pt_expt backends (test_with_comm_load_failure_ptexpt.cc)
		# skip.
		run_generator ${INFER_SCRIPT_PATH}/gen_corrupt_with_comm.py

		# Native-spin DPA4 graph archives (baseline + model-level pair
		# exclusion). Without this the whole native-spin graph C++ suite
		# GTEST_SKIPs on the missing fixture, which is how a dead
		# applyPairExclusion seam in DeepSpinPTExpt went unnoticed.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4_spin.py &
		PID11=$!
		# DPA4 + analytical ZBL bridging: a linear COMPOSITION on the graph
		# lower, which no other C++ fixture covers.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4_zbl.py &
		PID12=$!
		# Native-spin DPA4 + charge-spin FiLM: the ONLY fixture with both
		# is_spin=true and dim_chg_spin>0, i.e. the only one on which the
		# runtime charge_spin argument of DeepSpin::compute is not inert.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4_spin_chgspin.py &
		PID13=$!
		# Native-spin DPA4 + ZBL bridging COMBINED: spin reaching a linear
		# composition, including the source-freeze gate's cross-rank exchange
		# through the nested with-comm artifact.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4_spin_zbl.py &
		PID14=$!
		# Minimal native-spin DPA4C for LAMMPS pair_style dpa4spin (no
		# fparam/aparam/chg_spin). CPU-safe; used by the magnetic-force
		# convention behavioral tests.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4c_spin_lmp.py &
		PID14B=$!
		# Native-spin DPA4C: the archives above all ship a with-comm artifact
		# and are therefore served by DeepSpinPTExpt, which leaves
		# NativeSpinPTExpt -- the class the dispatch selects for a native-spin
		# archive without one -- uncovered. DPA4C keeps its messaging local and
		# so reaches it. The compiled graph calls CUDA-only operators, so the
		# fixture is generated only where the C++ suite has a GPU; the tests
		# built on it skip when it is absent.
		if [ "${DP_VARIANT}" = "cuda" ]; then
			run_generator ${INFER_SCRIPT_PATH}/gen_dpa4c_spin.py &
			PID15=$!
			# Compressed DPA4C, an energy and a native-spin model, on the compact
			# canonical lower: the only fixtures that reach the device-resident
			# entry points LAMMPS + Kokkos calls, with the replay of repeated
			# evaluations and a runtime charge state.
			run_generator ${INFER_SCRIPT_PATH}/gen_dpa4c_canonical.py &
			PID16=$!
		fi
		# DPA4C with a latent-charge long-range module: the uncompressed archive
		# takes the all-pairs topology of the graph lower; for the CUDA variant the
		# script also writes the compressed archive, whose compact canonical lower
		# reads the node positions.
		run_generator ${INFER_SCRIPT_PATH}/gen_dpa4c_long_range.py &
		PID17=$!
		wait $PID11
		wait $PID12
		wait $PID13
		wait $PID14
		wait $PID14B
		if [ "${DP_VARIANT}" = "cuda" ]; then
			wait $PID15
			wait $PID16
		fi
		wait $PID17
		# Only publish the cache manifest after successful model generation.
		for fixture in "${INFER_SCRIPT_PATH}"/*.pt2 "${INFER_SCRIPT_PATH}"/*.pte "${INFER_SCRIPT_PATH}"/*.expected; do
			[ -f "${fixture}" ] && basename "${fixture}"
		done | LC_ALL=C sort >"${_FIXTURE_MANIFEST}"
		if [ ! -s "${_FIXTURE_MANIFEST}" ]; then
			echo "ERROR: PT fixture generation produced no cacheable files" >&2
			exit 1
		fi
	fi
fi
if [ "${ENABLE_PADDLE:-TRUE}" == "TRUE" ]; then
	PADDLE_INFERENCE_DIR=${BUILD_TMP_DIR}/paddle_inference_install_dir
	export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PADDLE_INFERENCE_DIR}/third_party/install/onednn/lib:${PADDLE_INFERENCE_DIR}/third_party/install/mklml/lib
fi
# CTest otherwise hides every successful gtest's per-case time. Passing a
# directory makes GoogleTest use each executable's name (lib, cc, c), so even
# a parallel CTest run cannot overwrite another executable's JSON report.
mkdir -p "${BUILD_TMP_DIR}/gtest-reports"
GTEST_REPORT_DIR=$(mktemp -d "${BUILD_TMP_DIR}/gtest-reports/run.XXXXXX")
export GTEST_OUTPUT=${GTEST_OUTPUT:-json:${GTEST_REPORT_DIR}/}
ctest --output-on-failure --output-junit "${GTEST_REPORT_DIR}/ctest.xml"
