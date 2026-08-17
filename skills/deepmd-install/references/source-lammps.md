# Source installation: LAMMPS

Build LAMMPS only after the C/C++ prefix passes its public API and dynamic-link
checks. Use `doc/install/install-lammps.md` and the example README files inside
the selected DeePMD-kit checkout for version-specific commands. The published
documentation is
<https://docs.deepmodeling.com/projects/deepmd/en/latest/install/install-lammps.html>.

## Contents

- [Acquire LAMMPS](#acquire-lammps)
- [Add the built-in module](#add-the-built-in-module)
- [Select the runtime](#select-the-runtime)
- [Host build](#host-build)
- [Kokkos CUDA build](#kokkos-cuda-build)
- [Verification](#verification)
- [Smoke tests](#smoke-tests)

## Acquire LAMMPS

Use an existing source directory only when its version matches the plan and its
working tree is suitable for a build. Otherwise download the exact HTTPS URL
from the plan into a separate directory:

```bash
curl -fL "<lammps-url>" -o "<absolute-download-path>"
```

For a download, require `lammps.sha256` and verify it before extraction:

```bash
printf '%s  %s\n' "<sha256>" "<absolute-download-path>" | sha256sum --check -
```

Stop on a missing or mismatched checksum. A plan may omit the URL and checksum
only when `lammps.source_directory` already exists and its version has been
verified.

Extract into a directory whose resolved path equals
`lammps.source_directory`. Stop if the archive creates an unexpected directory
or contains paths outside its top-level directory.

## Add the built-in module

Use the bundled preparer instead of appending an unquoted CMake line:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/prepare_lammps.py" \
    --lammps-source "<absolute-lammps-source>" \
    --deepmd-source "<absolute-deepmd-source>"
```

The script maintains exactly one quoted include block and replaces a single
legacy include. It fails on ambiguous duplicate includes.

## Select the runtime

| Selection                | Required pair styles      |
| ------------------------ | ------------------------- |
| conventional host        | `deepmd`                  |
| conventional Kokkos CUDA | `deepmd`, `deepmd/kk`     |
| DPA4/SeZM host           | `deepmd`                  |
| DPA4/SeZM Kokkos CUDA    | `deepmd`, `deepmd/kk`     |
| DPA4C host               | `dpa4spin`                |
| DPA4C Kokkos CUDA        | `dpa4spin`, `dpa4spin/kk` |

`deepmd/kk` accepts a compatible edge-input or graph-input `.pt2` artifact.
It requires `atom_modify map yes` and a single model; model-deviation ensembles
use the host pair style. `dpa4spin/kk` accepts a DPA4C compact canonical graph
artifact and requires `atom_style spin`. Neither style is provided by LAMMPS
`PKG_GPU`.

For Kokkos, select the architecture from numeric compute capability, then
confirm that the chosen LAMMPS tree defines the option:

| Compute capability | Kokkos architecture |
| ------------------ | ------------------- |
| 7.0                | `VOLTA70`           |
| 7.5                | `TURING75`          |
| 8.0                | `AMPERE80`          |
| 8.6                | `AMPERE86`          |
| 8.9                | `ADA89`             |
| 9.0                | `HOPPER90`          |
| 10.0               | `BLACKWELL100`      |
| 12.0               | `BLACKWELL120`      |
| 12.1               | `BLACKWELL121`      |

```bash
grep -F "kokkos_arch_option(<KOKKOS_ARCH>" \
    "<absolute-lammps-source>/lib/kokkos/cmake/kokkos_arch.cmake"
```

If the numeric capability or the Kokkos option is unavailable, stop and select
a compatible toolkit/LAMMPS version. Use one architecture and a distinct build
directory/binary for each target GPU family.

## Host build

Configure the minimal host binary. Enable optional LAMMPS packages only when
the user's simulation requires them:

```bash
"<absolute-cmake>" \
    -S "<absolute-lammps-source>/cmake" \
    -B "<absolute-lammps-build>" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="<absolute-cc>" \
    -DCMAKE_CXX_COMPILER="<absolute-cxx>" \
    -DPython_EXECUTABLE="<absolute-python>" \
    -DPython3_EXECUTABLE="<absolute-python>" \
    -DCMAKE_INSTALL_PREFIX="<absolute-deepmd-prefix>" \
    -DBUILD_MPI="<ON|OFF>" \
    -DBUILD_SHARED_LIBS=ON \
    -DLAMMPS_INSTALL_RPATH=ON \
    -DCMAKE_PREFIX_PATH="<absolute-deepmd-prefix>"
```

Build and install with the planned job limit:

```bash
"<absolute-cmake>" --build "<absolute-lammps-build>" --parallel "<jobs>"
"<absolute-cmake>" --install "<absolute-lammps-build>"
```

The expected binary is `<deepmd-prefix>/bin/lmp` unless the plan explicitly
sets a LAMMPS machine suffix.

## Kokkos CUDA build

Configure one architecture. The same shell call discovers PyTorch and
namespace-package NCCL, constructs RPATH, and invokes CMake so no derived value
can disappear:

```bash
runtime_rpath=$("<absolute-python>" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

import torch

paths = [Path(torch.__file__).resolve().parent / "lib"]
spec = find_spec("nvidia.nccl") if find_spec("nvidia") is not None else None
if spec is not None:
    for root in spec.submodule_search_locations or ():
        candidate = Path(root) / "lib"
        if list(candidate.glob("libnccl.so*")):
            paths.append(candidate)
print(";".join(str(path) for path in paths if path.is_dir()))
PY
) &&
test -n "$runtime_rpath" &&
cuda_runtime_dir=$("<absolute-python>" - "<absolute-cuda-home>" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = [root / "lib64", *root.glob("targets/*-linux/lib")]
print(next((path.resolve() for path in candidates if path.is_dir()), ""))
PY
) &&
test -n "$cuda_runtime_dir" &&
torch_cmake_prefix=$("<absolute-python>" -c \
    "import torch; print(torch.utils.cmake_prefix_path)") &&
runtime_rpath="<absolute-deepmd-prefix>/lib;$cuda_runtime_dir;$runtime_rpath" &&
"<absolute-cmake>" \
    -S "<absolute-lammps-source>/cmake" \
    -B "<absolute-lammps-build>" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="<absolute-cc>" \
    -DCMAKE_CXX_COMPILER="<absolute-cxx>" \
    -DCMAKE_CUDA_COMPILER="<absolute-cuda-home>/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="<absolute-cxx>" \
    -DPython_EXECUTABLE="<absolute-python>" \
    -DPython3_EXECUTABLE="<absolute-python>" \
    -DCMAKE_INSTALL_PREFIX="<absolute-deepmd-prefix>" \
    -DLAMMPS_MACHINE="<machine>" \
    -DBUILD_MPI="<ON|OFF>" \
    -DBUILD_SHARED_LIBS=OFF \
    -DLAMMPS_INSTALL_RPATH=ON \
    -DPKG_KOKKOS=ON \
    -DKokkos_ENABLE_CUDA=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ARCH_<KOKKOS_ARCH>=ON \
    -DCMAKE_PREFIX_PATH="<absolute-deepmd-prefix>;$torch_cmake_prefix" \
    -DCUDAToolkit_ROOT="<absolute-cuda-home>" \
    -DCMAKE_BUILD_RPATH="$runtime_rpath" \
    -DCMAKE_INSTALL_RPATH="$runtime_rpath"
```

The rendered command must contain one real `Kokkos_ARCH_*` argument and no
`<KOKKOS_ARCH>` placeholder. Add linker flags only in response to a concrete
link error and only after locating the named library.

Build and install:

```bash
"<absolute-cmake>" --build "<absolute-lammps-build>" --parallel "<jobs>"
"<absolute-cmake>" --install "<absolute-lammps-build>"
```

The expected binary is `<deepmd-prefix>/bin/lmp_<machine>`.

## Verification

Run the exact model-family verifier. Use `--check-links` for a source binary:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/verify_lammps.py" \
    --binary "<absolute-lammps-binary>" \
    --model-family "<conventional|dpa4|dpa4c>" \
    --flavor "<host|kokkos-cuda>" \
    --check-links
```

Do not replace a failed exact style check with a broad `grep deepmd`.

## Smoke tests

Check the selected physical GPU immediately before a CUDA test:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv
```

Bind it explicitly so the process sees that physical card as logical device 0:

```bash
CUDA_VISIBLE_DEVICES="<physical-index>" \
    "<absolute-lammps-binary>" -k on g 1 -sf kk -in "<absolute-input-file>"
```

For DPA4/SeZM, follow
`examples/water/dpa4/lmp/README.md` from the same DeePMD-kit checkout. For
DPA4C, follow `examples/spin/dpa4c/lmp/README.md`; use its `--pt-expt`
compression/freeze sequence and spin input rather than a generic freeze
command. For conventional models, use a short user-selected example compatible
with the model format.

Success requires a completed run with `Loop time` in the LAMMPS log and no
artifact-schema, pair-style, device, or unresolved-library error.
