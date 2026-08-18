---
name: deepmd-install
description: Install DeePMD-kit with pip, conda, dp1s, an offline package, Docker, or source code. Use for PyTorch, TensorFlow, JAX, or Paddle on CPU, CUDA, or ROCm, and for backend-enabled or backend-neutral C/C++ interfaces and DeePMD-enabled LAMMPS.
---

# Install DeePMD-kit

Use the official documentation as the source of version-specific commands.
This skill selects the shortest suitable installation path, applies a small
set of safety rules, and verifies the requested runtime. It does not duplicate
the full installation manual.

## 1. Establish the target

Determine only the choices that affect installation:

- DeePMD-kit release or Git ref;
- PyTorch, TensorFlow, JAX, or Paddle backend, or a backend-neutral C/C++
  library;
- CPU, NVIDIA CUDA, or ROCm runtime;
- Python only, packaged LAMMPS, C/C++, or source-built LAMMPS;
- existing environment or a new user-approved environment.

If the user does not request a development version, prefer the current stable
release. When no method is specified, recommend a source install if the
machine has the required compiler and the user accepts the build time;
otherwise choose the shortest supported package path. Do not create a new
environment when the user requires an existing one. Ask only for a missing
choice that changes the installation path.

The backend-neutral choice applies only to the C/C++ interface and requires a
compatible backend plugin at runtime. It is not a Python installation target.

Inspect the OS, architecture, Python version and prefix, package manager, and
requested accelerator before changing the machine. Resolve the selected
Python executable to an absolute path and use `"<absolute-python>" -m pip`;
do not rely on a bare `pip` or shell activation persisting between commands.

## 2. Read the matching documentation

Assume no local DeePMD-kit checkout exists. Fetch the documentation before
rendering an install command:

1. Use the agent's browser or web-fetch tool to open the direct official page,
   not a search-result summary.

1. For a release, open the Releases page, record its tag, and use the matching
   versioned documentation.

1. If the page is missing or no web-fetch tool is available, fetch the needed
   Markdown file from that exact tag or commit:

   ```bash
   curl -fsSL --retry 2 \
       "https://raw.githubusercontent.com/deepmodeling/deepmd-kit/<REF>/doc/install/<FILE>.md"
   ```

1. If direct GitHub access fails, fetch the same path and exact ref from the
   official Gitee mirror. Reject an HTTP failure, empty response, HTML error
   page, or missing ref.

Read [`references/official-docs.md`](references/official-docs.md) for the URL
map, version-selection rules, exact repository paths, and network fallback.
Use the docs matching the selected version, not `latest` for an older release.

The commands below are route summaries. Before execution, replace every
placeholder with an observed or user-selected value and apply any changed
requirements from the matching official page.

## 3. Choose one installation path

### pip

Use pip for a released Python package and optional packaged LAMMPS. Start from
the backend tab in the official easy-install page:

| Backend    | Minimal package shape                                                                      |
| ---------- | ------------------------------------------------------------------------------------------ |
| PyTorch    | Install the selected PyTorch build, then `deepmd-kit` or `deepmd-kit[torch]` as documented |
| TensorFlow | `deepmd-kit[cpu]` for CPU or the documented GPU/CUDA extras                                |
| JAX        | `deepmd-kit[jax]`, plus the documented JAX accelerator package                             |
| Paddle     | Install the documented Paddle package first, then `deepmd-kit`                             |

Add `lmp` or `ipi` to the extras only when requested and supported by the
selected backend and platform. Install through the absolute target Python:

```bash
"<absolute-python>" -m pip install "<requirement-from-the-matching-docs>"
```

### conda

Use conda-forge for a released package when the user prefers conda:

```bash
"<absolute-conda-or-mamba>" create -n "<environment-name>" \
    -c conda-forge deepmd-kit
```

Add `lammps` or distributed-training packages only when requested. Follow the
linked conda-forge CUDA guidance rather than inventing a toolkit pin. After
creation, resolve the environment's absolute Python before verification.

### dp1s

Use the official one-second installer when the user selects it. Show the
remote script command and obtain confirmation before piping it to a shell:

```bash
curl -fsSL https://dp1s.deepmodeling.com | bash
```

Read the dp1s repository for `DP1S_HOME`, version selection, release-candidate,
and PATH-update options. Resolve the installed `dp` entry point and its Python
prefix instead of assuming that `DP1S_HOME` is the Python environment.

### Offline package

Use the exact asset for the selected release, OS, architecture, and runtime
from the official GitHub Releases page. Follow the release instructions to
assemble split files, verify the published checksum when available, and run
the completed installer. Never execute a partial download, an HTML response,
or an asset selected only by a similar filename.

### Docker

Pull the exact official image tag selected from the package page. For current
official images, the DeePMD environment is under `/opt/deepmd-kit`; confirm the
selected image's absolute `sys.executable` and `sys.prefix` before using it.
The minimal CPU check is:

```bash
docker pull "<official-image:tag>"
docker run --rm --entrypoint /opt/deepmd-kit/bin/python \
    "<official-image:tag>" -c \
    "import sys, deepmd; print(sys.executable, sys.prefix, deepmd.__version__)"
```

Use a different absolute interpreter only when the selected image definition
documents it. For packaged LAMMPS, run the binary inside the same container:

```bash
docker run --rm --entrypoint /opt/deepmd-kit/bin/lmp \
    "<official-image:tag>" -h
```

A host-side `lmp` does not verify the image. Mount inputs read-only, and add
explicit GPU device selection for CUDA.

### Source Python (recommended)

Use source installation for a reproducible build from a selected stable tag,
an unreleased feature, a custom build, or ROCm. Reject a remote or ref beginning
with `-`, then clone and resolve the selected ref safely:

```bash
git clone --no-checkout -- \
    https://github.com/deepmodeling/deepmd-kit.git "<source-directory>"
git -C "<source-directory>" fetch --tags -- origin "<validated-ref>"
git -C "<source-directory>" checkout --detach FETCH_HEAD
git -C "<source-directory>" rev-parse HEAD
```

If GitHub is unavailable, use the official mirror at
`https://gitee.com/deepmodeling/deepmd-kit.git` and resolve the same validated
ref. Do not obtain installation instructions or source through an
unauthenticated proxy.

Read `doc/install/install-from-source.md` at that exact commit, install the
selected backend first, and apply only the documented build variables. Render
the target-defining variables in the same invocation so build defaults cannot
select another runtime or backend:

```bash
DP_VARIANT="<cpu|cuda|rocm>" \
DP_ENABLE_TENSORFLOW="<0|1>" \
DP_ENABLE_PYTORCH="<0|1>" \
"<absolute-python>" -m pip install "<absolute-source-directory>"
```

Enable only the TensorFlow or PyTorch compiled support requested by the user.
For a Python-only JAX or Paddle installation, set both backend variables to
`0` unless the matching documentation requires compiled support. Add the
documented `CUDAToolkit_ROOT` or `ROCM_ROOT` to the same invocation when the
selected runtime requires an explicit toolkit root.

Keep source, build, and install locations distinct.

### Pre-compiled C library

Use this route only when the official page provides an artifact for the
selected version, platform, and backend. Download and unpack it into a
dedicated prefix, then follow the same page for CMake discovery and optional
LAMMPS plugin use. Do not substitute a Python wheel or a C library from another
release.

### C/C++ interface and LAMMPS

For C/C++, choose the backend-enabled or backend-neutral section of the
matching source-install page; do not guess CMake options or backend library
roots. A backend-neutral build must set `ALLOW_NO_BACKEND=ON`, build only the
C/C++ libraries, and provide a compatible backend plugin at runtime. For
LAMMPS, use a packaged `lmp` when it satisfies the request. Otherwise follow
the matching built-in or plugin instructions after the C/C++ interface
succeeds. Enable Kokkos only when the requested LAMMPS runtime requires it,
and select the architecture supported by that exact LAMMPS/Kokkos source tree.

## 4. Verify the requested interface

An installation is complete only after the requested public interface runs:

1. Print `sys.executable`, `sys.prefix`, `deepmd.__version__`, and
   `deepmd.__file__` with the selected absolute Python.
1. Import the selected backend (`deepmd.pt`, `deepmd.tf`, `deepmd.jax`, or
   `deepmd.pd`) and run one minimal tensor operation on the requested device.
1. Run the installed `dp --version` and backend-specific help.
1. For C/C++, confirm the installed headers/libraries, then load a built
   library or run a linked client so the platform loader resolves its dynamic
   dependencies.
1. For LAMMPS, run the selected binary with `-h`, require the exact DeePMD pair
   style needed by the model, and run a short documented example.
1. For Docker, perform all applicable checks inside the selected image.

On failure, stop at the first failing check and read
[`references/failure-modes.md`](references/failure-modes.md). Report the
selected method, version or commit, environment identity, commands executed,
observed verification results, and anything that remains unverified.
