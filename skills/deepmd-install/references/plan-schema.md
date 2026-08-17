# Installation plan schema

Store every decision in one JSON document. Keep the plan outside source,
build, and install directories. The plan contains no credentials or shell
commands.

## Contents

- [Top-level fields](#top-level-fields)
- [Conditional objects](#conditional-objects)
- [Validation rules](#validation-rules)
- [Examples](#examples)

## Top-level fields

```json
{
  "schema_version": 1,
  "method": "pip",
  "goal": "python",
  "backend": "pytorch",
  "accelerator": "cuda",
  "environment": {},
  "package": {},
  "source": null,
  "build": null,
  "cpp": null,
  "lammps": null,
  "smoke_test": {
    "enabled": false,
    "gpu": null,
    "example": null
  }
}
```

Allowed values:

- `method`: `pip`, `conda`, `dp1s`, `offline`, `docker`, or `source`.
- `goal`: `python`, `python+lammps`, `python+cpp`, or
  `python+cpp+lammps`.
- `backend`: `pytorch`, `tensorflow`, `jax`, or `paddle`.
- `accelerator`: `cpu`, `cuda`, or `rocm`.

Use `null` for an inapplicable object. Do not add undeclared keys.

## Conditional objects

### `environment`

```json
{
  "kind": "existing",
  "python": "/absolute/path/to/python",
  "manager": null,
  "name": null,
  "prefix": "/absolute/environment/prefix"
}
```

- `kind`: `existing`, `venv`, `conda`, `prefix`, or `container`.
- `python`: required for `pip` and `source`; resolve it after creating a new
  environment and before installing DeePMD-kit.
- `manager` and `name`: required for `conda`; `manager` is the absolute
  `conda` or `mamba` executable.
- `prefix`: required for `venv`, `prefix`, and offline installations.

### `package`

```json
{
  "deepmd_version": null,
  "deepmd_index_url": null,
  "deepmd_extra_index_url": null,
  "backend_packages": [
    "torch"
  ],
  "backend_index_url": null,
  "channels": [
    "conda-forge"
  ],
  "install_lammps": false,
  "install_ipi": false,
  "artifact_url": null,
  "sha256": null,
  "docker_image": null
}
```

- Record exact backend requirements selected from the backend's official
  installer or the version-matched DeePMD-kit documentation.
- Keep the two DeePMD-kit index fields null for the default package index.
  Record a user-selected mirror or the documented pre-release index explicitly.
- Keep `backend_index_url` null when the default package index is intended.
- Require `artifact_url` and `sha256` for `offline`.
- Require an immutable image reference or user-selected tag for `docker`.

### `source`

```json
{
  "directory": "/absolute/path/to/deepmd-kit",
  "remote": "https://github.com/deepmodeling/deepmd-kit.git",
  "ref": "master",
  "editable": false
}
```

Treat `ref` as an opaque Git ref. Resolve it to a commit and record that commit
in the completion report. Do not reset or clean an existing checkout.

### `build`

```json
{
  "variant": "cuda",
  "cc": "/usr/bin/gcc",
  "cxx": "/usr/bin/g++",
  "cuda_home": "/usr/local/cuda-12.8",
  "rocm_root": null,
  "native_optimization": false,
  "jobs": 8
}
```

`build.variant` describes DeePMD-kit's compiled custom operations. It may
differ from `accelerator` for a backend whose accelerator runtime is supplied
entirely by its Python package. Require `cuda_home` for a CUDA build and
`rocm_root` for a ROCm build.

### `cpp`

```json
{
  "install_prefix": "/absolute/dedicated/deepmd-prefix",
  "build_directory": "/absolute/build/deepmd-cpp",
  "tensorflow_root": null,
  "tensorflow_c_root": null,
  "paddle_inference_dir": null
}
```

Use a dedicated install prefix. Never choose `/`, a home directory, a conda
prefix, `/usr`, `/usr/local`, or `$HOME/.local` as a disposable prefix.

For a JAX C++ backend, provide either `tensorflow_root` or
`tensorflow_c_root`. For Paddle C++, provide `paddle_inference_dir`.

### `lammps`

```json
{
  "source_directory": "/absolute/path/to/lammps",
  "build_directory": "/absolute/build/lammps-blackwell120",
  "version": "stable_22Jul2025_update2",
  "url": "https://github.com/lammps/lammps/archive/refs/tags/stable_22Jul2025_update2.tar.gz",
  "sha256": null,
  "flavor": "kokkos-cuda",
  "machine": "blackwell120",
  "kokkos_arch": "BLACKWELL120",
  "mpi": false,
  "model_family": "dpa4c"
}
```

- `flavor`: `host` or `kokkos-cuda`.
- `model_family`: `conventional`, `dpa4`, or `dpa4c`.
- `machine` and `kokkos_arch` are required only for `kokkos-cuda`.
- Use one Kokkos GPU architecture and one build directory per binary.

### `smoke_test`

```json
{
  "enabled": true,
  "gpu": 7,
  "example": "/absolute/path/to/example"
}
```

Require a physical GPU index for a CUDA smoke test. Keep `gpu` null for CPU.
The example path must belong to the selected source checkout or be explicitly
provided by the user.

## Validation rules

The validator enforces these invariants:

1. `pip` and `source` use an absolute Python executable.
1. C/C++ goals use `source` and provide `build` plus `cpp`.
1. Source LAMMPS provides `lammps`; Kokkos CUDA requires the PyTorch backend
   and a CUDA build.
1. DPA4C maps to `dpa4spin`/`dpa4spin/kk`; other families map to
   `deepmd`/`deepmd/kk`.
1. Source directories, build directories, and install prefixes are distinct.
1. Checksums contain exactly 64 hexadecimal characters.
1. Unknown keys and unsupported combinations fail before any state change.

## Examples

### Pip PyTorch CUDA

```json
{
  "schema_version": 1,
  "method": "pip",
  "goal": "python",
  "backend": "pytorch",
  "accelerator": "cuda",
  "environment": {
    "kind": "existing",
    "python": "/opt/conda/envs/deepmd/bin/python",
    "manager": null,
    "name": null,
    "prefix": "/opt/conda/envs/deepmd"
  },
  "package": {
    "deepmd_version": null,
    "deepmd_index_url": null,
    "deepmd_extra_index_url": null,
    "backend_packages": [],
    "backend_index_url": null,
    "channels": [],
    "install_lammps": false,
    "install_ipi": false,
    "artifact_url": null,
    "sha256": null,
    "docker_image": null
  },
  "source": null,
  "build": null,
  "cpp": null,
  "lammps": null,
  "smoke_test": {
    "enabled": false,
    "gpu": null,
    "example": null
  }
}
```

### Source PyTorch CUDA with DPA4C LAMMPS

```json
{
  "schema_version": 1,
  "method": "source",
  "goal": "python+cpp+lammps",
  "backend": "pytorch",
  "accelerator": "cuda",
  "environment": {
    "kind": "existing",
    "python": "/opt/conda/envs/deepmd/bin/python",
    "manager": null,
    "name": null,
    "prefix": "/opt/conda/envs/deepmd"
  },
  "package": {
    "deepmd_version": null,
    "deepmd_index_url": null,
    "deepmd_extra_index_url": null,
    "backend_packages": [],
    "backend_index_url": null,
    "channels": [],
    "install_lammps": false,
    "install_ipi": false,
    "artifact_url": null,
    "sha256": null,
    "docker_image": null
  },
  "source": {
    "directory": "/work/deepmd-kit",
    "remote": "https://github.com/deepmodeling/deepmd-kit.git",
    "ref": "master",
    "editable": false
  },
  "build": {
    "variant": "cuda",
    "cc": "/usr/bin/gcc",
    "cxx": "/usr/bin/g++",
    "cuda_home": "/usr/local/cuda-12.8",
    "rocm_root": null,
    "native_optimization": false,
    "jobs": 8
  },
  "cpp": {
    "install_prefix": "/work/install/deepmd-cuda",
    "build_directory": "/work/build/deepmd-cpp-cuda",
    "tensorflow_root": null,
    "tensorflow_c_root": null,
    "paddle_inference_dir": null
  },
  "lammps": {
    "source_directory": "/work/lammps-stable_22Jul2025_update2",
    "build_directory": "/work/build/lammps-blackwell120",
    "version": "stable_22Jul2025_update2",
    "url": "https://github.com/lammps/lammps/archive/refs/tags/stable_22Jul2025_update2.tar.gz",
    "sha256": null,
    "flavor": "kokkos-cuda",
    "machine": "blackwell120",
    "kokkos_arch": "BLACKWELL120",
    "mpi": false,
    "model_family": "dpa4c"
  },
  "smoke_test": {
    "enabled": true,
    "gpu": 7,
    "example": "/work/deepmd-kit/examples/spin/dpa4c/lmp"
  }
}
```
