# package C library into a tarball

set -e

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
if [ -z "${INSTALL_PREFIX}" ]; then
	INSTALL_PREFIX=$(realpath -s "${SCRIPT_PATH}/../../dp_c")
fi
mkdir -p "${INSTALL_PREFIX}"
echo "Installing DeePMD-kit to ${INSTALL_PREFIX}"
NPROC=$(nproc --all)

#------------------

BUILD_TMP_DIR=${SCRIPT_PATH}/../build_c
mkdir -p "${BUILD_TMP_DIR}"
cd "${BUILD_TMP_DIR}"
CMAKE_EXTRA_ARGS=${CMAKE_EXTRA_ARGS:-}
if [ -n "${PYTHON_BIN:-}" ]; then
	PYTHON_INCLUDE_DIR=$(
		"${PYTHON_BIN}" - <<'PY'
import sysconfig

print(sysconfig.get_path("include") or "")
PY
	)
	PYTHON_LIBRARY=$(
		"${PYTHON_BIN}" - <<'PY'
import pathlib
import sysconfig

lib_names = (
    sysconfig.get_config_var("LDLIBRARY"),
    sysconfig.get_config_var("INSTSONAME"),
    sysconfig.get_config_var("LIBRARY"),
)
lib_dirs = (
    sysconfig.get_config_var("LIBDIR"),
    sysconfig.get_config_var("LIBPL"),
)
for lib_dir in filter(None, lib_dirs):
    for lib_name in filter(None, lib_names):
        candidate = pathlib.Path(lib_dir) / lib_name
        if candidate.exists():
            print(candidate)
            raise SystemExit
print("")
PY
	)
	PYTHON_CMAKE_ARGS="-DPython_EXECUTABLE=${PYTHON_BIN}"
	if [ -n "${PYTHON_INCLUDE_DIR}" ]; then
		PYTHON_CMAKE_ARGS="${PYTHON_CMAKE_ARGS} -DPython_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR}"
	fi
	if [ -n "${PYTHON_LIBRARY}" ]; then
		PYTHON_CMAKE_ARGS="${PYTHON_CMAKE_ARGS} -DPython_LIBRARY=${PYTHON_LIBRARY} -DPYTHON_LIBRARY=${PYTHON_LIBRARY}"
	fi
	if [ -z "${CUDA_NVRTC_LIBRARY:-}" ]; then
		CUDA_NVRTC_LIBRARY=$(
			"${PYTHON_BIN}" - <<'PY'
import pathlib
import site
import sys

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
seen = set()
for root in roots:
    if not root:
        continue
    root_path = pathlib.Path(root)
    if root_path in seen:
        continue
    seen.add(root_path)
    for pattern in (
        "nvidia/cuda_nvrtc/lib/libnvrtc.so",
        "nvidia/cuda_nvrtc/lib/libnvrtc.so.*",
    ):
        for candidate in sorted(root_path.glob(pattern)):
            if candidate.exists():
                print(candidate)
                raise SystemExit
print("")
PY
		)
	fi
fi
if [ -n "${CUDA_NVRTC_LIBRARY:-}" ]; then
	CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DCUDA_nvrtc_LIBRARY=${CUDA_NVRTC_LIBRARY}"
fi
cmake -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
	-DUSE_CUDA_TOOLKIT="${USE_CUDA_TOOLKIT:-TRUE}" \
	-DPACKAGE_C=TRUE \
	-DUSE_TF_PYTHON_LIBS="${USE_TF_PYTHON_LIBS:-TRUE}" \
	-DENABLE_PYTORCH="${ENABLE_PYTORCH:-FALSE}" \
	-DUSE_PT_PYTHON_LIBS="${USE_PT_PYTHON_LIBS:-${ENABLE_PYTORCH:-FALSE}}" \
	-DPACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES="${PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES:-}" \
	-DPACKAGE_C_RUNTIME_DEPENDENCY_POST_EXCLUDE_REGEXES="${PACKAGE_C_RUNTIME_DEPENDENCY_POST_EXCLUDE_REGEXES:-}" \
	${PYTHON_CMAKE_ARGS:-} \
	${CMAKE_EXTRA_ARGS:-} \
	..
cmake --build . -j"${NPROC}"
cmake --install .

#------------------

# fix runpath
for ii in "${BUILD_TMP_DIR}"/libdeepmd_c/lib/*.so*; do
	patchelf --set-rpath \$ORIGIN "$ii"
done

if [ -z "${PYTORCH_RUNTIME_DOWNLOAD_URL:-}" ] && [ "${ENABLE_PYTORCH:-FALSE}" = "TRUE" ] && [ -n "${PYTHON_BIN:-}" ]; then
	PYTORCH_RUNTIME_DOWNLOAD_URL=$(
		"${PYTHON_BIN}" - <<'PY'
from urllib.parse import quote

import torch

version = torch.__version__
if "+" in version:
    variant = version.split("+", 1)[1]
else:
    cuda_version = torch.version.cuda
    variant = "cu" + cuda_version.replace(".", "") if cuda_version else "cpu"
    version = f"{version}+{variant}"

if variant == "cpu" or variant.startswith("cu"):
    print(
        f"https://download.pytorch.org/libtorch/{variant}/"
        f"libtorch-shared-with-deps-{quote(version, safe='')}.zip"
    )
PY
	)
fi

if [ "${ENABLE_PYTORCH:-FALSE}" = "TRUE" ] && [ -z "${PYTORCH_RUNTIME_DOWNLOAD_URL:-}" ]; then
	if [ -z "${PYTHON_BIN:-}" ]; then
		echo "WARNING: PyTorch support is enabled, but PYTHON_BIN is not set; README/download_libtorch.sh will not be generated. Set PYTORCH_RUNTIME_DOWNLOAD_URL or PYTHON_BIN to emit the libtorch runtime helper." >&2
	else
		echo "WARNING: PyTorch support is enabled, but no supported libtorch runtime download URL could be derived from ${PYTHON_BIN}; README/download_libtorch.sh will not be generated. Set PYTORCH_RUNTIME_DOWNLOAD_URL explicitly to emit the libtorch runtime helper." >&2
	fi
fi

if [ -n "${PYTORCH_RUNTIME_DOWNLOAD_URL:-}" ]; then
	cat >"${BUILD_TMP_DIR}/libdeepmd_c/README.md" <<EOF
This DeePMD-kit C package was built with PyTorch support, but PyTorch runtime libraries are not bundled.

To use the PyTorch C/C++ backend, install a libtorch runtime that exactly matches the PyTorch version used at build time:

${PYTORCH_RUNTIME_DOWNLOAD_URL}

The PyTorch version must match exactly. The CUDA variant may be omitted only when the target runtime is compatible with the models and hardware you use.
Make the libtorch lib directory discoverable by the dynamic linker, for example by adding it to LD_LIBRARY_PATH.
Run ./download_libtorch.sh from this directory to download and unpack the matching libtorch runtime.
EOF
	cat >"${BUILD_TMP_DIR}/libdeepmd_c/download_libtorch.sh" <<'EOF'
#!/bin/sh

set -eu

LIBTORCH_DOWNLOAD_URL="__PYTORCH_RUNTIME_DOWNLOAD_URL__"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
DEST_DIR=${1:-"${SCRIPT_DIR}"}
ARCHIVE_PATH=${LIBTORCH_ARCHIVE:-"${SCRIPT_DIR}/libtorch.zip"}

mkdir -p "${DEST_DIR}"

if [ -d "${DEST_DIR}/libtorch" ]; then
	echo "libtorch already exists at ${DEST_DIR}/libtorch"
else
	echo "Downloading ${LIBTORCH_DOWNLOAD_URL}"
	if command -v curl >/dev/null 2>&1; then
		curl -L --fail --retry 3 -o "${ARCHIVE_PATH}" "${LIBTORCH_DOWNLOAD_URL}"
	elif command -v wget >/dev/null 2>&1; then
		wget -O "${ARCHIVE_PATH}" "${LIBTORCH_DOWNLOAD_URL}"
	else
		echo "curl or wget is required to download libtorch." >&2
		exit 1
	fi

	echo "Extracting ${ARCHIVE_PATH} to ${DEST_DIR}"
	if command -v unzip >/dev/null 2>&1; then
		unzip -q -o "${ARCHIVE_PATH}" -d "${DEST_DIR}"
	elif command -v python3 >/dev/null 2>&1; then
		ZIP_ARCHIVE="${ARCHIVE_PATH}" ZIP_DEST_DIR="${DEST_DIR}" python3 - <<'PY'
import os
import zipfile

with zipfile.ZipFile(os.environ["ZIP_ARCHIVE"]) as zip_file:
    zip_file.extractall(os.environ["ZIP_DEST_DIR"])
PY
	elif command -v python >/dev/null 2>&1; then
		ZIP_ARCHIVE="${ARCHIVE_PATH}" ZIP_DEST_DIR="${DEST_DIR}" python - <<'PY'
import os
import zipfile

with zipfile.ZipFile(os.environ["ZIP_ARCHIVE"]) as zip_file:
    zip_file.extractall(os.environ["ZIP_DEST_DIR"])
PY
	else
		echo "unzip or python is required to extract libtorch." >&2
		exit 1
	fi
fi

cat >"${SCRIPT_DIR}/libtorch_env.sh" <<EOF_ENV
export LD_LIBRARY_PATH="${DEST_DIR}/libtorch/lib:\${LD_LIBRARY_PATH:-}"
EOF_ENV

echo "libtorch is available at ${DEST_DIR}/libtorch"
echo "Run this before using the PyTorch C/C++ backend:"
echo "  . ${SCRIPT_DIR}/libtorch_env.sh"
EOF
	sed -i "s#__PYTORCH_RUNTIME_DOWNLOAD_URL__#${PYTORCH_RUNTIME_DOWNLOAD_URL}#g" "${BUILD_TMP_DIR}/libdeepmd_c/download_libtorch.sh"
	chmod +x "${BUILD_TMP_DIR}/libdeepmd_c/download_libtorch.sh"
fi

tar vczf "${SCRIPT_PATH}/../../libdeepmd_c.tar.gz" -C "${BUILD_TMP_DIR}" libdeepmd_c
