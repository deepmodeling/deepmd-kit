# Easy installation

Use this reference for stable package releases and official artifacts. Read the
version-matched `doc/install/easy-install.md` when package extras, platform
support, or backend installation commands differ from this routing guide. The
published documentation is
<https://docs.deepmodeling.com/projects/deepmd/en/latest/install/easy-install.html>.

## Contents

- [Environment gate](#environment-gate)
- [Pip](#pip)
- [Conda](#conda)
- [dp1s](#dp1s)
- [Offline package](#offline-package)
- [Docker](#docker)
- [Verification](#verification)

## Environment gate

Do not mix conda, pip, offline, and `dp1s` installations in one prefix. Reuse
an environment only when the user selected it and the probe found no conflicting
DeePMD-kit install.

ROCm installations use the source workflow. Packaged LAMMPS and i-PI support
TensorFlow, PyTorch, and JAX; do not add either extra to a Paddle environment.

For a new venv, create it first, resolve its absolute interpreter, update the
plan, and re-run `validate_plan.py` before installing packages:

```bash
"<absolute-bootstrap-python>" -m venv "<absolute-venv-prefix>"
"<absolute-venv-prefix>/bin/python" -c "import sys; print(sys.executable)"
```

Use the platform-equivalent interpreter path on Windows.

## Pip

Install backend packages recorded in `package.backend_packages` before
DeePMD-kit when they require a dedicated index. Render one literal pip command;
omit `--index-url` when `backend_index_url` is null.

```bash
"<absolute-python>" -m pip install \
    "<backend-package-1>" "<backend-package-2>" \
    --index-url "<confirmed-backend-index>"
```

Choose DeePMD-kit extras from the matching checkout documentation:

| Backend        | DeePMD-kit requirement                                                    |
| -------------- | ------------------------------------------------------------------------- |
| PyTorch        | `deepmd-kit[torch]`                                                       |
| TensorFlow CPU | `deepmd-kit[cpu]`                                                         |
| TensorFlow GPU | `deepmd-kit[gpu,cu12]` when the documented CUDA runtime extra is required |
| JAX            | `deepmd-kit[jax]` plus the selected JAX accelerator package               |
| Paddle         | Install the selected Paddle package first, then `deepmd-kit`              |

Append `lmp` and/or `ipi` only when the plan enables them. Append the exact
DeePMD-kit version when `package.deepmd_version` is non-null. Example shape:

```bash
"<absolute-python>" -m pip install "deepmd-kit[torch,lmp]==<version>"
```

The final rendered command must contain a real version or omit the version
specifier; it must not contain `<version>`. Add `--index-url` and
`--extra-index-url` only when the corresponding DeePMD-kit index fields are
non-null.

## Conda

Use the absolute manager and environment name from the plan. Install only the
requested packages; do not add LAMMPS, Horovod, or MPI unless they are in scope.
Treat `package.channels` as authoritative when it is non-empty and preserve its
order by rendering one `-c` option per entry. This includes selected mirrors and
pre-release channels such as `conda-forge/label/deepmd-kit_dev` or
`conda-forge/label/deepmd-kit_rc`. Use `conda-forge` only when the list is empty.
Pass `--override-channels` so configured defaults cannot supply packages outside
the plan.

Render the DeePMD-kit package argument as `deepmd-kit=<version>` when
`package.deepmd_version` is non-null, or as `deepmd-kit` when it is null. The
following non-null example preserves the recorded channel order:

```bash
"<absolute-conda-or-mamba>" create -n "<environment-name>" \
    --override-channels \
    -c "<channel-1>" -c "<channel-2>" "deepmd-kit=<version>"
```

For null `package.deepmd_version` and an empty `package.channels` list, render
the stable unversioned default explicitly:

```bash
"<absolute-conda-or-mamba>" create -n "<environment-name>" \
    --override-channels -c conda-forge deepmd-kit
```

Add `lammps` only for `goal=python+lammps`. Resolve the created interpreter
without assuming shell activation:

```bash
"<absolute-conda-or-mamba>" run -n "<environment-name>" \
    python -c "import sys; print(sys.executable); print(sys.prefix)"
```

Record both absolute paths in the plan, re-run `validate_plan.py`, and then run
the Python verifier. Follow the conda-forge CUDA guidance referenced by
`doc/install/easy-install.md`; do not invent a `cudatoolkit` pin.

## dp1s

Show the exact command and obtain confirmation before piping a remote response
to a shell:

```bash
curl -fsSL https://dp1s.deepmodeling.com | env \
    DP1S_HOME="<absolute-dp1s-home>" \
    DP1S_NO_PATH_UPDATE=1 \
    DEEPMD_VERSION="<deepmd-version>" \
    bash
```

Omit the `DEEPMD_VERSION` assignment when `package.deepmd_version` is null.
`DP1S_NO_PATH_UPDATE` prevents the installer from editing shell startup files.
Apply only additional options selected from the official `dp1s` documentation.
After installation, keep `environment.dp1s_home` unchanged. Resolve the absolute
interpreter from the installed `dp` entry point, run that interpreter to obtain
its `sys.prefix`, record both as `environment.python` and `environment.prefix`,
and re-run validation before verification. The Python prefix normally differs
from `dp1s_home`; pass only `environment.prefix` to `--expected-prefix`.

## Offline package

Use the exact release asset and SHA-256 checksum recorded in the plan. For a
remote artifact:

```bash
curl -fL "<artifact-url>" -o "<absolute-download-path>"
printf '%s  %s\n' "<sha256>" "<absolute-download-path>" | sha256sum --check -
bash "<absolute-download-path>"
```

For `package.artifact_path`, skip curl and verify the local file directly:

```bash
printf '%s  %s\n' "<sha256>" "<absolute-artifact-path>" | sha256sum --check -
bash "<absolute-artifact-path>"
```

Use `shasum -a 256` on systems without `sha256sum`. This workflow accepts one
complete installer only. For a split release, follow the version-matched manual
instructions to produce an already-assembled local installer, and use this
workflow only when a trusted final SHA-256 is available for that complete file.
Never execute an HTML error page or an artifact whose checksum is unknown.

## Docker

Pull the exact image reference from the plan:

```bash
docker pull "<registry/image:tag-or-digest>"
```

Mount the verifier read-only and invoke the backend and accelerator selected by
the plan:

```bash
docker run --rm \
    --mount \
    type=bind,src="<absolute-skill-root>/scripts/verify_python.py",dst=/opt/deepmd-install/verify_python.py,readonly \
    "<registry/image:tag-or-digest>" \
    python /opt/deepmd-install/verify_python.py \
    --backend "<pytorch|tensorflow|jax|paddle>" \
    --accelerator "<cpu|cuda>" \
    --expected-version "<deepmd-version>"
```

For CUDA, add `--gpus 'device=<physical-index>'` from the plan before the
read-only mount. Omit `--expected-version` when the plan does not pin one. Add
only user-selected volume mounts and working directories; never mount a home
directory or source tree read-write for verification.

## Verification

Run the backend-aware verifier with the absolute installed interpreter:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/verify_python.py" \
    --backend "<pytorch|tensorflow|jax|paddle>" \
    --accelerator "<cpu|cuda>" \
    --expected-version "<deepmd-version>" \
    --expected-prefix "<absolute-environment-prefix>"
```

Omit `--expected-version` when `package.deepmd_version` is null.

For packaged host LAMMPS, resolve the executable and require the conventional
host pair style:

```bash
"<absolute-python>" "<absolute-skill-root>/scripts/verify_lammps.py" \
    --binary "<absolute-lammps-binary>" \
    --model-family "<package.lammps_model_family>" \
    --flavor host
```

Packaged LAMMPS does not imply Kokkos `deepmd/kk` or DPA4C
`dpa4spin/kk`. Use the source LAMMPS path when either device pair style is
required.
