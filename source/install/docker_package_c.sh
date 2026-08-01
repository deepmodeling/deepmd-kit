set -e

SCRIPT_PATH=$(dirname "$(realpath -s "$0")")
MANYLINUX_CUDA_IMAGE=${MANYLINUX_CUDA_IMAGE:-quay.io/manylinux_cuda/manylinux_2_28_x86_64_cuda12_9:latest}
PYTHON_BIN=${PYTHON_BIN:-/opt/python/cp311-cp311/bin/python}
PYTORCH_DEPENDENCY_GROUP=${PYTORCH_DEPENDENCY_GROUP:-pin_pytorch_cpu}
PYTORCH_TORCH_BACKEND=${PYTORCH_TORCH_BACKEND:-cpu}
TENSORFLOW_DEPENDENCY_GROUP=${TENSORFLOW_DEPENDENCY_GROUP:-pin_tensorflow_cpu}

docker run --rm -v "${SCRIPT_PATH}/../..":/root/deepmd-kit -w /root/deepmd-kit \
	-e CIBUILDWHEEL="${CIBUILDWHEEL:-1}" \
	-e ENABLE_PYTORCH="${ENABLE_PYTORCH:-FALSE}" \
	-e MANYLINUX_CUDA_IMAGE="${MANYLINUX_CUDA_IMAGE}" \
	-e PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES="${PACKAGE_C_RUNTIME_DEPENDENCY_PRE_EXCLUDE_REGEXES:-}" \
	-e PACKAGE_C_RUNTIME_DEPENDENCY_POST_EXCLUDE_REGEXES="${PACKAGE_C_RUNTIME_DEPENDENCY_POST_EXCLUDE_REGEXES:-}" \
	-e PYTORCH_DEPENDENCY_GROUP="${PYTORCH_DEPENDENCY_GROUP}" \
	-e PYTHON_BIN="${PYTHON_BIN}" \
	-e PYTORCH_RUNTIME_DOWNLOAD_URL="${PYTORCH_RUNTIME_DOWNLOAD_URL:-}" \
	-e PYTORCH_TORCH_BACKEND="${PYTORCH_TORCH_BACKEND}" \
	-e TENSORFLOW_DEPENDENCY_GROUP="${TENSORFLOW_DEPENDENCY_GROUP}" \
	"${MANYLINUX_CUDA_IMAGE}" \
	/bin/bash -lc 'set -euo pipefail
            export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
            "${PYTHON_BIN}" -m pip install uv
            UV_INSTALL_ARGS=(
              pip install
              --python "${PYTHON_BIN}"
              cmake
              ninja
              patchelf
              --group "${TENSORFLOW_DEPENDENCY_GROUP}"
            )
            if [ "${ENABLE_PYTORCH}" = "TRUE" ]; then
              UV_INSTALL_ARGS+=(--group "${PYTORCH_DEPENDENCY_GROUP}" --torch-backend "${PYTORCH_TORCH_BACKEND}")
            fi
            source/install/uv_with_retry.sh "${UV_INSTALL_ARGS[@]}"
            git config --global --add safe.directory /root/deepmd-kit
            cd /root/deepmd-kit/source/install
            /bin/sh package_c.sh'
