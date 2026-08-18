# Source installation: Python interface

Build the Python package from the exact Git commit recorded by the workflow.
Use `doc/install/install-from-source.md` inside that checkout as the authority
for version-specific backend packages and build variables. The published
counterpart is
<https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-from-source.html>.

## Contents

- [Checkout gate](#checkout-gate)
- [Interpreter and compiler gate](#interpreter-and-compiler-gate)
- [Backend dependencies](#backend-dependencies)
- [Build](#build)
- [Verification](#verification)

## Checkout gate

For an absent target directory, clone without checking out a moving branch,
fetch the requested opaque ref, resolve it to a commit, and check out that
commit detached. Render all values literally in one shell call:

```bash
git clone --no-checkout "<git-remote>" "<absolute-source-directory>" &&
git -C "<absolute-source-directory>" fetch --tags "<git-remote>" "<git-ref>" &&
git -C "<absolute-source-directory>" checkout --detach \
    "$(git -C "<absolute-source-directory>" rev-parse 'FETCH_HEAD^{commit}')" &&
git -C "<absolute-source-directory>" rev-parse HEAD
```

Store the resolved SHA in `source.commit` and re-run `validate_plan.py` with
`--require-resolved-source` before installing backend packages or building
DeePMD-kit.

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/validate_plan.py" \
    "<absolute-plan-path>" --require-resolved-source
```

For an existing checkout, inspect it without changing its branch, remotes, or
working tree:

```bash
git -C "<absolute-source-directory>" status --short
git -C "<absolute-source-directory>" remote -v
git -C "<absolute-source-directory>" rev-parse HEAD
```

If the existing `HEAD` is not the requested ref, or tracked/untracked changes
overlap the build, use a separate clone or ask the user which tree to use. Do
not reset, clean, or repoint `origin`.

## Interpreter and compiler gate

Use the plan's absolute interpreter and confirm Python 3.10 or newer. Record
the resolved `sys.prefix` as `environment.prefix` and revalidate the plan if it
differs from the recorded value:

```bash
"<absolute-python>" -c \
    "import sys; print(sys.executable); print(sys.prefix); print(sys.version)"
"<absolute-cc>" --version
"<absolute-cxx>" --version
```

For a CUDA build, verify the selected toolkit and host compiler in the same
call. A CUDA major-version mismatch with the selected backend package is a hard
failure; handle a same-major minor mismatch through the actual configure/build
result rather than string equality alone.

```bash
"<absolute-cuda-home>/bin/nvcc" --version
"<absolute-cxx>" --version
```

Install build requirements into the same environment:

```bash
"<absolute-python>" -m pip install --upgrade \
    pip scikit-build-core packaging cmake ninja dependency_groups
```

## Backend dependencies

Install every entry in `package.backend_packages` before building DeePMD-kit.
Use the recorded backend index only when it is non-null.

### PyTorch

Select the PyTorch requirement and index from the official PyTorch installer or
the checkout documentation. Verify the wheel before compiling DeePMD-kit:

```bash
"<absolute-python>" -c \
    "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
```

Use the `torch` extra for the source package. It supplies `vesin[torch]` and
the platform-eligible nvalchemi integration.

### TensorFlow

Install the TensorFlow package selected by the checkout documentation, then
verify it with the same interpreter:

```bash
"<absolute-python>" -c \
    "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

A CUDA DeePMD-kit build requires a local CUDA toolkit visible to CMake. Keep
`DP_ENABLE_TENSORFLOW=1` and `DP_ENABLE_PYTORCH=0` explicit during the build.

### JAX

Install the JAX accelerator package selected by the official JAX instructions
and use the DeePMD-kit `jax` extra. JAX may provide the GPU runtime while the
DeePMD-kit compiled variant remains CPU; record `accelerator` and
`build.variant` separately in the plan.

```bash
"<absolute-python>" -c \
    "import jax; print(jax.__version__, jax.devices())"
```

### Paddle

Install the Paddle package and index selected by
`doc/install/easy-install.md` in the checkout. Build the Python package with
TensorFlow and PyTorch disabled unless either backend is also requested.

```bash
"<absolute-python>" -c \
    "import paddle; print(paddle.__version__, paddle.device.get_device())"
```

## Build

Select the source requirement by backend:

| Backend    | Local requirement | Build switches                                  |
| ---------- | ----------------- | ----------------------------------------------- |
| PyTorch    | `.[torch]`        | `DP_ENABLE_PYTORCH=1`, `DP_ENABLE_TENSORFLOW=0` |
| TensorFlow | `.`               | `DP_ENABLE_PYTORCH=0`, `DP_ENABLE_TENSORFLOW=1` |
| JAX        | `.[jax]`          | `DP_ENABLE_PYTORCH=0`, `DP_ENABLE_TENSORFLOW=0` |
| Paddle     | `.`               | `DP_ENABLE_PYTORCH=0`, `DP_ENABLE_TENSORFLOW=0` |

Render a single self-contained invocation. This CUDA PyTorch template shows the
required shape:

```bash
cd "<absolute-source-directory>" &&
env \
    CC="<absolute-cc>" \
    CXX="<absolute-cxx>" \
    CUDA_HOME="<absolute-cuda-home>" \
    CUDA_PATH="<absolute-cuda-home>" \
    CUDAToolkit_ROOT="<absolute-cuda-home>" \
    CUDACXX="<absolute-cuda-home>/bin/nvcc" \
    CUDAHOSTCXX="<absolute-cxx>" \
    DP_VARIANT=cuda \
    DP_ENABLE_PYTORCH=1 \
    DP_ENABLE_TENSORFLOW=0 \
    DP_ENABLE_NATIVE_OPTIMIZATION=0 \
    CMAKE_BUILD_PARALLEL_LEVEL="<jobs>" \
    "<absolute-python>" -m pip install \
    ".[torch]" --no-build-isolation --verbose
```

For CPU, omit CUDA variables and set `DP_VARIANT=cpu`. For ROCm, follow the
selected checkout documentation and set `DP_VARIANT=rocm` plus the planned
`ROCM_ROOT` in the same invocation.

For TensorFlow, JAX, and Paddle, replace the requirement and the two
`DP_ENABLE_*` values using the table. Do not add a backend the plan does not
request.

## Verification

Run outside the source directory so an import cannot succeed from the checkout
alone:

```bash
cd "<absolute-neutral-directory>" &&
"<absolute-python>" "<absolute-skill-root>/scripts/verify_python.py" \
    --backend "<pytorch|tensorflow|jax|paddle>" \
    --accelerator "<cpu|cuda|rocm>" \
    --expected-prefix "<absolute-environment-prefix>" \
    --expected-source-commit "<resolved-source-commit>" \
    --expected-build-variant "<cpu|cuda|rocm>"
```

Add `--expected-version "<deepmd-version>"` when the plan pins a release. Add
`--expect-custom-op` for a PyTorch source build. Add `--expect-nv` and
`--expect-vesin` only when their platform markers and the plan require them. Do
not continue to the C/C++ gate until every requested check passes.
