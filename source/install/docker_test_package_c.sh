# test libdeepmd_c.tar.gz works on a manylinux 2.28 runtime
set -e

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
MANYLINUX_IMAGE=${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64:latest}
PYTHON_BIN=${PYTHON_BIN:-/opt/python/cp311-cp311/bin/python}
PYTORCH_DEPENDENCY_GROUP=${PYTORCH_DEPENDENCY_GROUP:-pin_pytorch_cpu}
PYTORCH_TORCH_BACKEND=${PYTORCH_TORCH_BACKEND:-cpu}

# assume libdeepmd_c.tar.gz has been created

docker run --rm -v "${SCRIPT_PATH}/../..":/root/deepmd-kit -w /root/deepmd-kit \
	-e CHECK_PYTORCH_RUNTIME="${CHECK_PYTORCH_RUNTIME:-0}" \
	-e PYTHON_BIN="${PYTHON_BIN}" \
	-e PYTORCH_DEPENDENCY_GROUP="${PYTORCH_DEPENDENCY_GROUP}" \
	-e PYTORCH_TORCH_BACKEND="${PYTORCH_TORCH_BACKEND}" \
	"${MANYLINUX_IMAGE}" \
	/bin/bash -lc 'set -euo pipefail
            export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
            tar vxzf libdeepmd_c.tar.gz
            if [ -f libdeepmd_c/download_libtorch.sh ]; then
              sh -n libdeepmd_c/download_libtorch.sh
            fi
            cd examples/infer_water
            gcc convert_model.c -std=c99 -L ../../libdeepmd_c/lib -I ../../libdeepmd_c/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=../../libdeepmd_c/lib -o convert_model
            gcc infer_water.c -std=c99 -L ../../libdeepmd_c/lib -I ../../libdeepmd_c/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=../../libdeepmd_c/lib -o infer_water
            ./convert_model
            ./infer_water
            cd /root/deepmd-kit
            if [ "${CHECK_PYTORCH_RUNTIME}" = "1" ]; then
              "${PYTHON_BIN}" -m pip install uv
              source/install/uv_with_retry.sh pip install --python "${PYTHON_BIN}" --group "${PYTORCH_DEPENDENCY_GROUP}" --torch-backend "${PYTORCH_TORCH_BACKEND}"
              TORCH_RUNTIME_LIB_DIRS=$("${PYTHON_BIN}" - <<'"'"'PY'"'"'
import pathlib
import site
import sys

import torch

dirs = [pathlib.Path(torch.__file__).resolve().parent / "lib"]
roots = []
try:
    roots.extend(site.getsitepackages())
except AttributeError:
    pass
try:
    roots.append(site.getusersitepackages())
except AttributeError:
    pass
roots.extend(sys.path)
for root in roots:
    if not root:
        continue
    nvidia_dir = pathlib.Path(root) / "nvidia"
    if nvidia_dir.is_dir():
        dirs.extend(sorted(path for path in nvidia_dir.glob("*/lib") if path.is_dir()))
seen = set()
unique_dirs = []
for path in dirs:
    path = path.resolve()
    if path in seen:
        continue
    seen.add(path)
    unique_dirs.append(str(path))
print(":".join(unique_dirs))
PY
)
              RUNTIME_LIBRARY_PATH="${PWD}/libdeepmd_c/lib:${TORCH_RUNTIME_LIB_DIRS}"
              if [ -n "${LD_LIBRARY_PATH:-}" ]; then
                RUNTIME_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
              fi
              LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" ldd libdeepmd_c/lib/libdeepmd_backend_pt.so > /tmp/deepmd_pt_ldd.txt
              LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" ldd libdeepmd_c/lib/libdeepmd_backend_ptexpt.so >> /tmp/deepmd_pt_ldd.txt
              LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" ldd libdeepmd_c/lib/libdeepmd_op_pt.so >> /tmp/deepmd_pt_ldd.txt
              cat /tmp/deepmd_pt_ldd.txt
              if grep -q "not found" /tmp/deepmd_pt_ldd.txt; then
                exit 1
              fi
              LD_LIBRARY_PATH="${RUNTIME_LIBRARY_PATH}" "${PYTHON_BIN}" - <<'"'"'PY'"'"'
import ctypes
import os
import pathlib

import torch  # noqa: F401

libdir = pathlib.Path("libdeepmd_c/lib").resolve()
for name in [
    "libdeepmd_op_pt.so",
    "libdeepmd_backend_pt.so",
    "libdeepmd_backend_ptexpt.so",
]:
    ctypes.CDLL(str(libdir / name), mode=os.RTLD_NOW | os.RTLD_LOCAL)
PY
            fi'
