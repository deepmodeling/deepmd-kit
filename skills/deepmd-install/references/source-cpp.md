# Source installation: C/C++ interface

Build the C/C++ interface only after the Python/backend gate passes. Use
`doc/install/install-from-source.md` in the selected checkout for the exact
CMake options supported by that ref. The published documentation is
<https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-from-source.html>.

## Contents

- [Build-directory gate](#build-directory-gate)
- [Backend configuration](#backend-configuration)
- [Configure and install](#configure-and-install)
- [C/C++ verification](#cc-verification)

## Build-directory gate

Require CMake 3.25.2 or newer and the exact compilers from the plan:

```bash
"<absolute-cmake>" --version
"<absolute-cc>" --version
"<absolute-cxx>" --version
```

The source, build, and install paths must be distinct. The install prefix must
be dedicated to this build. If the build directory contains a `CMakeCache.txt`
from another source commit, compiler, backend, or accelerator, select a new
build directory. Never delete or recursively replace the install prefix.

## Backend configuration

Start with all backends disabled, then enable the selected backend and its
documented C API dependency. Enabling TensorFlow also enables the JAX C API in
DeePMD-kit by design.

| Backend                 | Required CMake arguments                                                            |
| ----------------------- | ----------------------------------------------------------------------------------- |
| PyTorch                 | `-DENABLE_PYTORCH=ON -DUSE_PT_PYTHON_LIBS=ON` plus the PyTorch CMake prefix         |
| TensorFlow              | `-DENABLE_TENSORFLOW=ON -DUSE_TF_PYTHON_LIBS=ON` and the selected Python executable |
| JAX with TensorFlow C++ | `-DENABLE_TENSORFLOW=ON -DUSE_TF_PYTHON_LIBS=ON`                                    |
| JAX with TensorFlow C   | `-DENABLE_JAX=ON -DCMAKE_PREFIX_PATH=<tensorflow-c-root>`                           |
| Paddle                  | `-DENABLE_PADDLE=ON -DPADDLE_INFERENCE_DIR=<paddle-inference-dir>`                  |

For PyTorch, discover the prefix from the planned interpreter in the same shell
call that configures CMake:

```bash
"<absolute-python>" -c "import torch; print(torch.utils.cmake_prefix_path)"
```

When more than one backend is explicitly requested, confirm compatible
`_GLIBCXX_USE_CXX11_ABI` settings before compiling. Do not enable an incidental
backend merely because its Python package is importable.

Accelerator arguments:

| Build variant | Required CMake arguments                                                                                                           |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| CPU           | `-DUSE_CUDA_TOOLKIT=OFF -DUSE_ROCM_TOOLKIT=OFF`                                                                                    |
| CUDA          | `-DUSE_CUDA_TOOLKIT=ON -DCUDAToolkit_ROOT=<cuda-home> -DCMAKE_CUDA_COMPILER=<cuda-home>/bin/nvcc -DCMAKE_CUDA_HOST_COMPILER=<cxx>` |
| ROCm          | `-DUSE_ROCM_TOOLKIT=ON -DCMAKE_HIP_COMPILER_ROCM_ROOT=<rocm-root>`                                                                 |

## Configure and install

Render all plan values into one call. This PyTorch CUDA example shows the
complete shape:

```bash
torch_cmake_prefix=$("<absolute-python>" -c \
    "import torch; print(torch.utils.cmake_prefix_path)") &&
"<absolute-cmake>" \
    -S "<absolute-deepmd-source>/source" \
    -B "<absolute-cpp-build-directory>" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="<absolute-dedicated-prefix>" \
    -DCMAKE_C_COMPILER="<absolute-cc>" \
    -DCMAKE_CXX_COMPILER="<absolute-cxx>" \
    -DPython_EXECUTABLE="<absolute-python>" \
    -DPython3_EXECUTABLE="<absolute-python>" \
    -DBUILD_CPP_IF=ON \
    -DBUILD_PY_IF=OFF \
    -DENABLE_NATIVE_OPTIMIZATION=OFF \
    -DENABLE_PYTORCH=ON \
    -DUSE_PT_PYTHON_LIBS=ON \
    -DENABLE_TENSORFLOW=OFF \
    -DENABLE_JAX=OFF \
    -DENABLE_PADDLE=OFF \
    -DUSE_CUDA_TOOLKIT=ON \
    -DUSE_ROCM_TOOLKIT=OFF \
    -DCMAKE_PREFIX_PATH="$torch_cmake_prefix" \
    -DCUDAToolkit_ROOT="<absolute-cuda-home>" \
    -DCMAKE_CUDA_COMPILER="<absolute-cuda-home>/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="<absolute-cxx>"
```

Replace only the backend and accelerator arguments using the tables. Set
`ENABLE_NATIVE_OPTIMIZATION=ON` only when the installed libraries remain on the
same CPU model.

Build and install with the bounded job count from the plan:

```bash
"<absolute-cmake>" --build "<absolute-cpp-build-directory>" \
    --parallel "<jobs>"
"<absolute-cmake>" --install "<absolute-cpp-build-directory>"
```

If the prefix already contains an installation, choose a new versioned prefix
and switch consumers only after its verification passes.

## C/C++ verification

Check the cache against the plan. For a CUDA PyTorch build, for example:

```bash
grep -E \
    'BUILD_CPP_IF:|BUILD_PY_IF:|ENABLE_PYTORCH:|USE_CUDA_TOOLKIT:|CMAKE_CUDA_COMPILER:' \
    "<absolute-cpp-build-directory>/CMakeCache.txt"
```

Require the headers and libraries:

```bash
test -f "<absolute-prefix>/include/deepmd/deepmd.hpp"
test -f "<absolute-prefix>/include/deepmd/c_api.h"
test -f "<absolute-prefix>/lib/libdeepmd_cc.so"
test -f "<absolute-prefix>/lib/libdeepmd_c.so"
```

Use the platform library suffix on macOS or Windows. On Linux, fail if any
installed DeePMD library has an unresolved dynamic dependency:

```bash
for library in "<absolute-prefix>"/lib/libdeepmd*.so; do
    ldd "$library"
done
```

Finally compile and run a public C++ API probe inside the build directory:

```bash
probe_source="<absolute-cpp-build-directory>/verify_deepmd.cpp"
probe_binary="<absolute-cpp-build-directory>/verify_deepmd"
printf '%s\n' \
    '#include "deepmd/common.h"' \
    'int main() { deepmd::print_summary(""); return 0; }' \
    > "$probe_source"
"<absolute-cxx>" -std=c++14 "$probe_source" \
    -I"<absolute-prefix>/include" \
    -L"<absolute-prefix>/lib" \
    -Wl,-rpath,"<absolute-prefix>/lib" \
    -ldeepmd_cc \
    -o "$probe_binary"
"$probe_binary"
```

Do not proceed to LAMMPS when cache values, dynamic dependencies, or the public
API probe fail.
