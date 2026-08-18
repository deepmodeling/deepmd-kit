# Installation failure modes

Diagnose the first failing step. Keep its complete output, change one input at
a time, and rerun the smallest command that distinguishes the cause. Do not
restart the whole installation, erase caches, reset a source tree, or remove a
working environment without the user's approval.

## Documentation or asset is unavailable

Confirm the requested version or ref and the exact URL. Try the versioned docs,
then the same file at the exact GitHub ref as described in
[`official-docs.md`](official-docs.md). For a blocked public GitHub URL, use the
documented `gh-proxy.com` fallback. Do not silently substitute a newer release,
different CUDA build, or similarly named asset.

For downloads, retain the HTTP status, file size, and content type. An HTML
error page is not an installer or archive. Reassemble split offline assets in
the documented order and verify a published checksum when one exists.

## Python, pip, and dp identify different environments

Run these checks with the selected absolute interpreter:

```bash
"<absolute-python>" -c \
    "import sys; print(sys.executable); print(sys.prefix)"
"<absolute-python>" -m pip --version
command -v dp
```

The Python prefix, pip location, `dp` entry point, and `deepmd.__file__` must
identify the intended environment. If an import resolves into a source tree,
repeat it from a neutral directory and check `PYTHONPATH`. Do not repair an
identity mismatch by installing the package again with a different `pip`.

## Dependency resolution or backend import fails

Read the first resolver conflict or import error. Compare the selected Python,
OS, architecture, DeePMD-kit version, backend package, and package index with
the matching documentation. Install only the requested backend; adding every
extra often creates conflicts and does not diagnose the missing dependency.

For PyTorch, TensorFlow, JAX, and Paddle, verify the framework itself before
testing DeePMD-kit. A successful `import deepmd` does not prove that the
selected backend is installed.

## CPU, CUDA, or ROCm is not available

Separate three facts: the accelerator is visible to the host, the framework
was built for the requested runtime, and a tensor operation executes on that
device. Check device visibility variables and compare the framework runtime
with the driver/toolkit reported by the system. Do not replace packages until
the failing layer is identified.

A source compiler toolkit and a framework wheel runtime are related but not
identical. Treat a major-version incompatibility as a hard failure; use the
official compatibility guidance for minor-version combinations.

## Source ref or build identity is wrong

Inspect without modifying the checkout:

```bash
git -C "<source-directory>" status --short
git -C "<source-directory>" remote -v
git -C "<source-directory>" rev-parse HEAD
```

Use a separate clone when the remote, commit, or local changes do not match the
request. Never use `git reset --hard` or `git clean` on a user tree. Reject a
remote or ref beginning with `-`; option-like input must not reach `git fetch`
or `git checkout`.

## A source build fails

Preserve the first compiler or CMake error. Verify the absolute compiler,
Python, backend installation, toolkit root, and free disk/RAM. Re-read the
source-install page and build options at the exact commit. If CMake cached a
different compiler, source path, backend, or install prefix, create a new build
directory instead of layering more flags onto the stale cache.

For PyTorch custom operations, confirm that the intended PyTorch is visible to
the build and that build isolation behavior matches the selected version's
documentation. For TensorFlow, JAX, or Paddle C/C++, use the documented library
interface and ABI for that backend; do not substitute a Python package root
unless the docs explicitly support it.

## A native library cannot be loaded

Use `ldd` on Linux or `otool -L` on macOS for the failing library or executable.
Locate the named dependency in the selected environment before changing
`RPATH`, `LD_LIBRARY_PATH`, or linker flags. Framework import success does not
prove that a separately built C++ client or LAMMPS binary uses the same ABI and
libraries.

## LAMMPS lacks the required DeePMD pair style

Run the exact selected binary with `-h`. Distinguish packaged, built-in, and
plugin installations; for plugin mode, confirm that the plugin was loaded.
Require the pair style used by the target model rather than any occurrence of
the word `deepmd`. Kokkos styles require a Kokkos-enabled LAMMPS build and an
architecture supported by that LAMMPS source tree.

If the style exists but rejects a model, diagnose model/runtime compatibility
separately from installation. Rebuild only after confirming that the binary,
DeePMD C/C++ libraries, backend, and model family are the intended combination.

## Docker verification disagrees with the host

Host executables do not verify a container. Run all checks inside the exact
image tag, record the container's absolute `sys.executable` and `sys.prefix`,
and invoke that interpreter directly. For packaged LAMMPS, run the in-container
binary. Mount verification inputs read-only and pass an explicit device with
Docker's GPU option when CUDA is requested.

If the image contains multiple Python installations, do not fall back to a
bare `python`; inspect the image definition or documented environment prefix
and verify `deepmd.__file__` against it.

## Permission, disk, or memory failure

Do not switch silently to `sudo`, a shared system prefix, or a different disk.
Report the required permission or resource and use a user-approved dedicated
prefix. For compilation memory pressure, reduce parallel jobs and rerun the
same build so that compiler, backend, toolkit, and source commit remain fixed.

## Reporting a blocked installation

Report the selected method and version, the exact failing command, its first
actionable error, environment identity, and checks that passed. State what
additional user choice or external change is required. Do not report an
installation as successful when only package resolution or file creation
completed.
