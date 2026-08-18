# Installation failure modes

Diagnose the current gate only. Preserve the plan and complete logs, change one
input at a time, and re-run the smallest failing command. Do not restart the
workflow, wipe caches, reset source trees, or remove install prefixes.

## Index

| Symptom                                         | Section                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------- |
| `python`, pip, and `dp` disagree                | [Wrong Python environment](#wrong-python-environment)                   |
| A plan value is empty or a placeholder remains  | [Invalid or stale plan](#invalid-or-stale-plan)                         |
| A bundled helper cannot be found                | [Wrong skill root](#wrong-skill-root)                                   |
| Backend imports but accelerator is unavailable  | [Backend accelerator failure](#backend-accelerator-failure)             |
| DeePMD-kit reports the wrong build variant      | [Wrong compiled variant](#wrong-compiled-variant)                       |
| `ENABLE_CUSTOMIZED_OP` is false                 | [PyTorch custom OP is unavailable](#pytorch-custom-op-is-unavailable)   |
| CUDA/toolkit/compiler error                     | [CUDA toolchain mismatch](#cuda-toolchain-mismatch)                     |
| nvalchemi or vesin is unavailable               | [Optional neighbor-list dependency](#optional-neighbor-list-dependency) |
| Source checkout is not the requested ref        | [Source identity mismatch](#source-identity-mismatch)                   |
| CMake keeps an old compiler/backend             | [Stale build directory](#stale-build-directory)                         |
| Torch, NCCL, CUPTI, or CUDA library is missing  | [Native library discovery](#native-library-discovery)                   |
| LAMMPS has no DeePMD styles                     | [Built-in module is absent](#built-in-module-is-absent)                 |
| `deepmd/kk` or `dpa4spin/kk` is absent          | [Wrong LAMMPS runtime](#wrong-lammps-runtime)                           |
| Kokkos reports an architecture error            | [Kokkos architecture mismatch](#kokkos-architecture-mismatch)           |
| LAMMPS starts but a shared library is not found | [Runtime link failure](#runtime-link-failure)                           |
| Build terminates from memory or disk pressure   | [Insufficient build resources](#insufficient-build-resources)           |
| Downloaded archive cannot be parsed or verified | [Artifact download failure](#artifact-download-failure)                 |

## Wrong Python environment

Run all checks with the absolute interpreter from the plan:

```bash
"<absolute-python>" -c "import sys; print(sys.executable); print(sys.prefix)"
"<absolute-python>" -m pip --version
command -v dp
head -n 1 "$(command -v dp)"
```

The interpreter, pip prefix, and `dp` shebang must identify the same
environment. Resolve `deepmd.__file__` as well; it must exist below the planned
prefix. If it points into a checkout or another environment, run the verifier
from a neutral directory without an injected `PYTHONPATH`. A previous
`conda activate` does not persist across independent agent shell calls.
If a neutral-directory import still points into the checkout, replace the
editable installation with the regular source installation documented in
[`source-python.md`](source-python.md). Re-render the failed gate with the
absolute interpreter.

## Invalid or stale plan

Re-run validation whenever a path, version, backend, accelerator, source ref,
or build directory changes:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/validate_plan.py" \
    "<absolute-plan-path>"
```

Never repair a failed command by guessing a missing value. Update the plan,
validate it, and render the gate again. A command containing an unassigned
plan variable, an empty required argument, or `<placeholder>` is not
executable.

For conda, compare every rendered `-c` argument with `package.channels`. An
empty list uses only the stable `conda-forge` default; a non-empty list replaces
that default and preserves the recorded order. For JAX C/C++, two null
TensorFlow roots select the Python-library route. A single root selects its
corresponding external library route; setting both is ambiguous and invalid.

## Wrong skill root

Helper scripts belong to the installed skill, not the DeePMD-kit checkout or
current directory:

```bash
test -f "<absolute-skill-root>/scripts/probe_env.py"
test -f "<absolute-skill-root>/scripts/verify_python.py"
test -f "<absolute-skill-root>/scripts/verify_lammps.py"
test -f "<absolute-skill-root>/scripts/verify_native.py"
```

Resolve the directory containing `SKILL.md`; do not search for a same-named
script elsewhere and do not skip the gate.

## Backend accelerator failure

Use the backend-aware verifier and retain its complete output:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/verify_python.py" \
    --backend "<backend>" \
    --accelerator "<accelerator>" \
    --expected-version "<deepmd-version>" \
    --expected-prefix "<absolute-environment-prefix>"
```

Omit `--expected-version` when the plan does not pin a release.

Check visibility masks from the probe before replacing packages. A false GPU
availability result may come from `CUDA_VISIBLE_DEVICES`,
`HIP_VISIBLE_DEVICES`, a driver/runtime mismatch, or a CPU backend package.
For JAX, inspect backend metadata rather than the device vendor label:

```bash
"<absolute-python>" - <<'PY'
import jax

for device in jax.devices():
    client = device.client
    print(device.platform, client.platform, client.platform_version)
PY
```

For a ROCm smoke test, inspect occupancy with `rocm-smi`, then bind the planned
physical index in the same command:

```bash
HIP_VISIBLE_DEVICES="<physical-index>" \
    ROCR_VISIBLE_DEVICES="<physical-index>" \
    "<absolute-python>" "<absolute-skill-root>/scripts/verify_python.py" \
    --backend "<backend>" \
    --accelerator rocm \
    --expected-prefix "<absolute-environment-prefix>"
```

## Wrong compiled variant

Framework GPU visibility does not prove that DeePMD-kit custom operations were
compiled for that accelerator. Inspect the recorded variant:

```bash
"<absolute-python>" - <<'PY'
from deepmd.env import GLOBAL_CONFIG

print(GLOBAL_CONFIG["dp_variant"])
PY
```

Return to the source Python gate when it differs from `build.variant`. Keep the
toolkit, compiler, `DP_VARIANT`, and backend switches in the same pip call.

## PyTorch custom OP is unavailable

Common causes are build isolation, a missing PyTorch installation during the
build, `DP_ENABLE_PYTORCH=0`, an ABI mismatch, or a runtime library that cannot
be loaded.

Confirm the intended PyTorch and toolchain, then re-run the source install with
`--no-build-isolation` and the complete inline environment from
[`source-python.md`](source-python.md). Do not use `--force-reinstall` until the
rendered command is known to contain the correct plan values.

## CUDA toolchain mismatch

Compare all three layers:

```bash
"<absolute-cuda-home>/bin/nvcc" --version
nvidia-smi
"<absolute-python>" -c \
    "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A CUDA major-version mismatch between toolkit and PyTorch is a hard failure.
A same-major minor mismatch requires the actual configure/compile result;
PyTorch treats it as a warning in extension builds. The NVIDIA driver must
support the selected runtime. If `nvcc` rejects the host compiler, select a
compiler supported by that toolkit and update the plan rather than using
`--allow-unsupported-compiler`.

## Optional neighbor-list dependency

`vesin[torch]` belongs to the PyTorch extra. nvalchemi package markers depend on
the operating system and Python version. Require these integrations only when
the selected checkout and platform declare them applicable:

```bash
"<absolute-python>" -m pip show vesin nvalchemi-toolkit-ops
```

Their absence on an ineligible platform is not a core DeePMD-kit installation
failure.

## Source identity mismatch

Inspect without modifying the checkout:

```bash
git -C "<absolute-source-directory>" status --short
git -C "<absolute-source-directory>" remote -v
git -C "<absolute-source-directory>" rev-parse HEAD
```

Use a separate clone when `HEAD`, remote, or local changes do not match the
plan. Store the resolved SHA in `source.commit`, revalidate with
`--require-resolved-source`, and pass the same SHA to
`verify_python.py --expected-source-commit`. Do not reset, clean, or rewrite
the existing checkout.

## Stale build directory

A CMake cache records absolute source paths, compilers, toolkit, backend, and
architecture. Compare it with the plan:

```bash
grep -E \
    'CMAKE_(C|CXX|CUDA)_COMPILER:|CMAKE_INSTALL_PREFIX:|ENABLE_|USE_.*TOOLKIT:|Kokkos_ARCH_' \
    "<absolute-build-directory>/CMakeCache.txt"
```

If any identity differs, create a new build directory. Preserve the old
directory for diagnosis until the replacement passes.

## Native library discovery

Locate PyTorch and namespace-package NCCL from the selected environment:

```bash
"<absolute-python>" - <<'PY'
from importlib.util import find_spec
from pathlib import Path

import torch

print(Path(torch.__file__).resolve().parent / "lib")
spec = find_spec("nvidia.nccl") if find_spec("nvidia") is not None else None
if spec is not None:
    for root in spec.submodule_search_locations or ():
        candidate = Path(root) / "lib"
        if list(candidate.glob("libnccl.so*")):
            print(candidate)
PY
```

`nvidia.nccl.__file__` may be null. Locate CUPTI independently; it may live
under `extras/CUPTI`, an environment package, or a separate toolkit component.
Add a directory to RPATH or linker flags only after the named library is found
there.

## Built-in module is absent

Check the managed include without changing the LAMMPS tree:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/prepare_lammps.py" \
    --lammps-source "<absolute-lammps-source>" \
    --deepmd-source "<absolute-deepmd-source>" \
    --check
```

If it differs, run the preparer without `--check`, then configure a new LAMMPS
build directory. Re-running CMake in the old directory may preserve a source
state from before the include was added.

## Wrong LAMMPS runtime

Verify exact styles rather than grepping any occurrence of `deepmd`:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/verify_lammps.py" \
    --binary "<absolute-lammps-binary>" \
    --model-family "<conventional|dpa4|dpa4c>" \
    --flavor "<host|kokkos-cuda>"
```

Kokkos must be enabled for `/kk`. DPA4C requires `dpa4spin/kk`, not
`deepmd/kk`. A CPU DeePMD C/C++ library cannot serve the Kokkos CUDA pair
styles. `PKG_GPU` does not provide them.

If the exact style exists but rejects the model, check the artifact contract:
`deepmd/kk` needs an edge/graph `.pt2`, one model, and `atom_modify map yes`;
`dpa4spin/kk` needs a DPA4C compact canonical graph and `atom_style spin`.

## Kokkos architecture mismatch

Query numeric capability and compare it with both the plan and selected Kokkos
tree:

```bash
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader
grep -F "kokkos_arch_option(<KOKKOS_ARCH>" \
    "<absolute-lammps-source>/lib/kokkos/cmake/kokkos_arch.cmake"
```

Use exactly one architecture in one build directory. Do not infer it from a
GPU-name substring. If the compiler test requires `nvcc_wrapper`, use the
wrapper from that LAMMPS tree with the host compiler supported by the selected
CUDA toolkit.

## Runtime link failure

Use the bundled verifier so Linux checks `ldd` output and macOS asks `dyld` to
load the library instead of trusting Mach-O metadata alone:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/verify_native.py" \
    --path "<absolute-native-library>"
"<absolute-python>" "<absolute-skill-root>/scripts/verify_lammps.py" \
    --binary "<absolute-lammps-binary>" \
    --model-family "<conventional|dpa4|dpa4c>" \
    --flavor "<host|kokkos-cuda>" \
    --check-links
```

Resolve every `not found` entry through build/install RPATH. Avoid global
`LD_LIBRARY_PATH` and shell-rc changes; they can cause another environment's
CUDA, Torch, TensorFlow, or compiler runtime to shadow the planned libraries.
On macOS, `otool -L` lists install names but does not prove they resolve.

## Insufficient build resources

Compare the probe's free disk, RAM, and CPU count with `build.jobs`. Reduce the
job count and re-run the same build command. Do not change compilers, backend,
or source ref in the same experiment. Preserve the first out-of-memory or disk
error because later compiler failures may be secondary damage.

## Artifact download failure

Use `curl -fL` so HTTP errors fail immediately. Verify file type and checksum
before extraction or execution:

```bash
file "<absolute-download-path>"
printf '%s  %s\n' "<sha256>" "<absolute-download-path>" | sha256sum --check -
```

An absent checksum, HTML response, truncated split archive, unexpected
top-level directory, or checksum mismatch blocks extraction. Obtain a fresh
artifact and trusted checksum for the planned URL; do not disable verification.

For `package.artifact_path`, skip curl and run the same file-type and checksum
checks directly on the absolute local path.
