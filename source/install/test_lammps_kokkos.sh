#!/bin/bash
set -euxo pipefail

# Run after test_cc_local.sh: reuse its C API/backend installation and the
# matching LAMMPS checkout instead of rebuilding DeePMD or using the pip wheel.
SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
REPO_ROOT=$(realpath "${SCRIPT_PATH}/../..")
DEEPMD_C_ROOT=${DEEPMD_C_ROOT:-${REPO_ROOT}/dp_test}
LAMMPS_SOURCE_ROOT=${LAMMPS_SOURCE_ROOT:-${REPO_ROOT}/source/build_tests/_deps/lammps_download-src}
BUILD_ROOT=${REPO_ROOT}/source/build_lammps_kokkos
LAMMPS_BUILD_DIR=${BUILD_ROOT}/lammps
PLUGIN_BUILD_DIR=${BUILD_ROOT}/plugin
PLUGIN_DIR=${BUILD_ROOT}/autoload
NVCC_WRAPPER=${LAMMPS_SOURCE_ROOT}/lib/kokkos/bin/nvcc_wrapper
# CUDA compilation can use substantial memory per process.
BUILD_JOBS=${CMAKE_BUILD_PARALLEL_LEVEL:-4}

if [[ ${DP_VARIANT:-} != cuda ]]; then
	echo "This regression requires DP_VARIANT=cuda." >&2
	exit 1
fi
test -f "${DEEPMD_C_ROOT}/lib/libdeepmd_c.so"
test -f "${LAMMPS_SOURCE_ROOT}/cmake/CMakeLists.txt"
test -x "${NVCC_WRAPPER}"
command -v nvcc
python -c 'import torch; assert torch.cuda.is_available(), "CUDA device is required"'

# Use the same MPI implementation for LAMMPS and its plugin. In CUDA CI this
# is MPICH, also selected for the ordinary plugin by test_cc_local.sh.
MPI_SUFFIX=
if [[ ${DP_USE_MPICH2:-0} == 1 ]]; then
	MPI_SUFFIX=.mpich
	export MPI4PY_MPIABI=mpich
fi
MPI_C_COMPILER=$(command -v "mpicc${MPI_SUFFIX}")
MPI_CXX_COMPILER=$(command -v "mpicxx${MPI_SUFFIX}")
export DPA4SPIN_KOKKOS_MPIEXEC
DPA4SPIN_KOKKOS_MPIEXEC=$(command -v "mpirun${MPI_SUFFIX}")
MPI_ARGS=(
	"-DMPI_C_COMPILER=${MPI_C_COMPILER}"
	"-DMPI_CXX_COMPILER=${MPI_CXX_COMPILER}"
)

# Kokkos detects the GPU architecture on the runner. Both builds use its
# nvcc_wrapper and the plugin imports the exact Kokkos build configuration.
cmake -S "${LAMMPS_SOURCE_ROOT}/cmake" -B "${LAMMPS_BUILD_DIR}" \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
	-D CMAKE_CXX_STANDARD=17 \
	-D BUILD_SHARED_LIBS=ON \
	-D BUILD_MPI=ON \
	-D BUILD_OMP=OFF \
	-D PKG_KOKKOS=ON \
	-D Kokkos_ENABLE_CUDA=ON \
	-D Kokkos_ENABLE_SERIAL=ON \
	-D PKG_SPIN=ON \
	-D PKG_PLUGIN=ON \
	-D PKG_KSPACE=ON \
	-D PKG_EXTRA-FIX=ON \
	-D FFT=KISS \
	-D FFT_KOKKOS=KISS \
	"${MPI_ARGS[@]}"
cmake --build "${LAMMPS_BUILD_DIR}" --target lammps --parallel "${BUILD_JOBS}"

cmake -S "${REPO_ROOT}/source" -B "${PLUGIN_BUILD_DIR}" \
	-D CMAKE_BUILD_TYPE=Release \
	-D CMAKE_CXX_COMPILER="${NVCC_WRAPPER}" \
	-D CMAKE_CXX_STANDARD=17 \
	-D CMAKE_DISABLE_PRECOMPILE_HEADERS=ON \
	-D BUILD_TESTING=OFF \
	-D DEEPMD_C_ROOT="${DEEPMD_C_ROOT}" \
	-D DP_USING_C_API=ON \
	-D LAMMPS_SOURCE_ROOT="${LAMMPS_SOURCE_ROOT}" \
	-D DEEPMD_LAMMPS_KOKKOS=ON \
	-D Kokkos_DIR="${LAMMPS_BUILD_DIR}/lib/kokkos" \
	"${MPI_ARGS[@]}"
cmake --build "${PLUGIN_BUILD_DIR}" --target deepmd_lmp --parallel "${BUILD_JOBS}"
mkdir -p "${PLUGIN_DIR}"
ln -sfn "${PLUGIN_BUILD_DIR}/lmp/plugin/libdeepmd_lmp.so" "${PLUGIN_DIR}/dpplugin.so"

export LD_LIBRARY_PATH=${LAMMPS_BUILD_DIR}:${DEEPMD_C_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
python "${REPO_ROOT}/source/tests/infer/gen_dpa4c_spin_lmp_kokkos.py"

# The wheel's Python package loads its own liblammps by absolute path. Merely
# setting LD_LIBRARY_PATH would therefore still test the non-Kokkos wheel.
export PYTHONPATH=${LAMMPS_SOURCE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}
export LAMMPS_PLUGIN_PATH=${PLUGIN_DIR}
python - "${LAMMPS_SOURCE_ROOT}" "${LAMMPS_BUILD_DIR}" <<'PY'
import sys
from pathlib import Path

import lammps

source, build = map(Path, sys.argv[1:])
assert Path(lammps.__file__).resolve() == (
    source / "python/lammps/__init__.py"
).resolve(), lammps.__file__
with lammps.lammps(cmdargs=["-log", "none", "-screen", "none"]) as lmp:
    mapped_libraries = {
        Path(line.split()[-1]).resolve()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/liblammps" in line
    }
    assert mapped_libraries == {(build / "liblammps.so").resolve()}, mapped_libraries
    assert lmp.has_style("pair", "dpa4spin/kk"), "Kokkos plugin was not loaded"
    print(f"Verified matching LAMMPS Python and shared library: {mapped_libraries}")
PY

DPA4SPIN_KOKKOS_REQUIRED=1 python -m pytest -v -s \
	"${REPO_ROOT}/source/lmp/tests/test_dpa4spin_kokkos_magnetic_force.py"
