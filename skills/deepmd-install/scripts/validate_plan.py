#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Validate a DeePMD-kit installation plan before changing the machine."""

from __future__ import (
    annotations,
)

import argparse
import json
import re
from pathlib import (
    Path,
)
from typing import (
    Any,
)
from urllib.parse import (
    urlparse,
)

SCHEMA_VERSION = 1
METHODS = {"pip", "conda", "dp1s", "offline", "docker", "source"}
GOALS = {"python", "python+lammps", "python+cpp", "python+cpp+lammps"}
BACKENDS = {"pytorch", "tensorflow", "jax", "paddle"}
ACCELERATORS = {"cpu", "cuda", "rocm"}
ENVIRONMENT_KINDS = {"existing", "venv", "conda", "prefix", "container"}
LAMMPS_FLAVORS = {"host", "kokkos-cuda"}
MODEL_FAMILIES = {"conventional", "dpa4", "dpa4c"}
TOP_LEVEL_KEYS = {
    "schema_version",
    "method",
    "goal",
    "backend",
    "accelerator",
    "environment",
    "package",
    "source",
    "build",
    "cpp",
    "lammps",
    "smoke_test",
}
OBJECT_KEYS = {
    "environment": {"kind", "python", "manager", "name", "prefix"},
    "package": {
        "deepmd_version",
        "deepmd_index_url",
        "deepmd_extra_index_url",
        "backend_packages",
        "backend_index_url",
        "channels",
        "install_lammps",
        "install_ipi",
        "artifact_url",
        "artifact_path",
        "sha256",
        "docker_image",
        "lammps_model_family",
    },
    "source": {"directory", "remote", "ref", "commit", "editable"},
    "build": {
        "variant",
        "cc",
        "cxx",
        "cuda_home",
        "rocm_root",
        "native_optimization",
        "jobs",
    },
    "cpp": {
        "install_prefix",
        "build_directory",
        "tensorflow_root",
        "tensorflow_c_root",
        "paddle_inference_dir",
    },
    "lammps": {
        "source_directory",
        "build_directory",
        "version",
        "url",
        "sha256",
        "flavor",
        "machine",
        "kokkos_arch",
        "mpi",
        "model_family",
    },
    "smoke_test": {"enabled", "gpu", "example"},
}


def _is_absolute_path(value: object) -> bool:
    """Return whether ``value`` is a non-empty absolute path string."""
    return isinstance(value, str) and bool(value) and Path(value).is_absolute()


def _canonical(value: str) -> Path:
    """Return a normalized path without requiring it to exist."""
    return Path(value).resolve(strict=False)


def _require_keys(
    data: dict[str, Any], required: set[str], context: str, errors: list[str]
) -> None:
    """Append errors for missing keys in a mapping."""
    for key in sorted(required - data.keys()):
        errors.append(f"{context}.{key}: missing required field")


def _check_unknown_keys(
    data: dict[str, Any], allowed: set[str], context: str, errors: list[str]
) -> None:
    """Append errors for undeclared keys in a mapping."""
    for key in sorted(data.keys() - allowed):
        errors.append(f"{context}.{key}: unknown field")


def _as_object(
    plan: dict[str, Any], key: str, *, required: bool, errors: list[str]
) -> dict[str, Any] | None:
    """Return a conditional object after validating its type and keys."""
    value = plan.get(key)
    if value is None:
        if required:
            errors.append(f"{key}: object is required for this plan")
        return None
    if not isinstance(value, dict):
        errors.append(f"{key}: expected an object or null")
        return None
    _check_unknown_keys(value, OBJECT_KEYS[key], key, errors)
    return value


def _validate_choice(
    value: object, choices: set[str], path: str, errors: list[str]
) -> str | None:
    """Validate a string enum and return the accepted value."""
    if not isinstance(value, str) or value not in choices:
        errors.append(f"{path}: unsupported value {value!r}")
        return None
    return value


def _validate_strings(value: object, path: str, errors: list[str]) -> None:
    """Reject unresolved placeholders and unsafe POSIX-template characters."""
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_strings(item, f"{path}.{key}" if path else key, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_strings(item, f"{path}[{index}]", errors)
    elif isinstance(value, str):
        if "$" in value:
            errors.append(f"{path}: unresolved shell variable is not allowed")
        if re.search(r"<[^<>\r\n]+>", value):
            errors.append(f"{path}: unresolved placeholder is not allowed")
        if any(character in value for character in ('"', "`", "\\")):
            errors.append(f"{path}: unsafe shell-template character is not allowed")
        if any(character in value for character in ("\0", "\n", "\r")):
            errors.append(f"{path}: control character is not allowed")


def _validate_checksum(value: object, path: str, errors: list[str]) -> None:
    """Validate an optional SHA-256 checksum."""
    if value is not None and (
        not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None
    ):
        errors.append(f"{path}: expected 64 hexadecimal SHA-256 characters")


def _validate_https_url(value: object, path: str, errors: list[str]) -> None:
    """Validate an HTTPS URL without embedded credentials."""
    if not isinstance(value, str) or not value:
        errors.append(f"{path}: expected an HTTPS URL")
        return
    parsed = urlparse(value)
    if (
        parsed.scheme == "https"
        and parsed.netloc
        and parsed.username is None
        and parsed.password is None
        and not any(character.isspace() for character in value)
    ):
        return
    errors.append(f"{path}: expected an HTTPS URL without credentials")


def _validate_string_list(value: object, path: str, errors: list[str]) -> None:
    """Validate a list of non-empty strings."""
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        errors.append(f"{path}: expected a list of non-empty strings")


def _validate_environment(
    environment: dict[str, Any], method: str, errors: list[str]
) -> None:
    """Validate method-specific environment fields."""
    _require_keys(environment, {"kind"}, "environment", errors)
    kind = _validate_choice(
        environment.get("kind"), ENVIRONMENT_KINDS, "environment.kind", errors
    )
    if method in {"pip", "source"}:
        python = environment.get("python")
        if not _is_absolute_path(python):
            errors.append(
                "environment.python: pip and source methods require an absolute executable"
            )
    if method == "conda":
        if not _is_absolute_path(environment.get("manager")):
            errors.append("environment.manager: conda requires an absolute executable")
        if not isinstance(environment.get("name"), str) or not environment.get("name"):
            errors.append(
                "environment.name: conda requires a non-empty environment name"
            )
    if kind in {"venv", "prefix"} and not _is_absolute_path(environment.get("prefix")):
        errors.append(f"environment.prefix: {kind} requires an absolute path")
    if method in {"dp1s", "offline"} and not _is_absolute_path(
        environment.get("prefix")
    ):
        errors.append(f"environment.prefix: {method} requires an absolute path")
    if method == "docker" and kind != "container":
        errors.append("environment.kind: docker requires 'container'")


def _validate_package(
    package: dict[str, Any],
    method: str,
    goal: str,
    backend: str,
    accelerator: str,
    errors: list[str],
) -> None:
    """Validate package and artifact selection."""
    for key in ("backend_packages", "channels"):
        if key in package:
            _validate_string_list(package[key], f"package.{key}", errors)
    for key in ("install_lammps", "install_ipi"):
        if key in package and not isinstance(package[key], bool):
            errors.append(f"package.{key}: expected a boolean")
    if package.get("deepmd_version") is not None and (
        not isinstance(package["deepmd_version"], str) or not package["deepmd_version"]
    ):
        errors.append("package.deepmd_version: expected a non-empty string or null")
    if package.get("backend_index_url") is not None:
        _validate_https_url(
            package["backend_index_url"], "package.backend_index_url", errors
        )
    for key in ("deepmd_index_url", "deepmd_extra_index_url"):
        if package.get(key) is not None:
            _validate_https_url(package[key], f"package.{key}", errors)
    _validate_checksum(package.get("sha256"), "package.sha256", errors)
    if method == "offline":
        _require_keys(package, {"sha256"}, "package", errors)
        artifact_url = package.get("artifact_url")
        artifact_path = package.get("artifact_path")
        if (artifact_url is None) == (artifact_path is None):
            errors.append(
                "package: offline installation requires exactly one of "
                "artifact_url or artifact_path"
            )
        elif artifact_url is not None:
            _validate_https_url(artifact_url, "package.artifact_url", errors)
        elif not _is_absolute_path(artifact_path):
            errors.append("package.artifact_path: expected an absolute path")
        if package.get("sha256") is None:
            errors.append("package.sha256: offline installation requires a checksum")
    elif (
        package.get("artifact_url") is not None
        or package.get("artifact_path") is not None
    ):
        errors.append("package.artifact_url/artifact_path: allowed only for offline")
    if method == "docker" and (
        not isinstance(package.get("docker_image"), str)
        or not package.get("docker_image")
    ):
        errors.append("package.docker_image: docker requires an image reference")
    if goal == "python+lammps" and package.get("install_lammps") is not True:
        errors.append(
            "package.install_lammps: python+lammps requires the packaged LAMMPS runtime"
        )
    if goal != "python+lammps" and package.get("install_lammps") is True:
        errors.append("package.install_lammps: requires goal 'python+lammps'")
    if method != "source" and accelerator == "rocm":
        errors.append("accelerator: ROCm requires method 'source'")
    packaged_lammps = goal == "python+lammps" or package.get("install_lammps") is True
    if packaged_lammps:
        family = _validate_choice(
            package.get("lammps_model_family"),
            MODEL_FAMILIES,
            "package.lammps_model_family",
            errors,
        )
        if family in {"dpa4", "dpa4c"} and backend != "pytorch":
            errors.append(
                f"package.lammps_model_family: {family} requires the PyTorch backend"
            )
    elif package.get("lammps_model_family") is not None:
        errors.append(
            "package.lammps_model_family: allowed only when packaged LAMMPS is requested"
        )
    if (
        backend == "paddle"
        and method != "source"
        and (packaged_lammps or package.get("install_ipi") is True)
    ):
        errors.append("package: Paddle does not support packaged LAMMPS or i-PI")


def _validate_source(
    source: dict[str, Any],
    build: dict[str, Any],
    backend: str,
    accelerator: str,
    errors: list[str],
) -> None:
    """Validate source checkout and build fields."""
    _require_keys(source, {"directory", "remote", "ref", "editable"}, "source", errors)
    if not _is_absolute_path(source.get("directory")):
        errors.append("source.directory: expected an absolute path")
    for key in ("remote", "ref"):
        if not isinstance(source.get(key), str) or not source.get(key):
            errors.append(f"source.{key}: expected a non-empty string")
    commit = source.get("commit")
    if commit is not None and (
        not isinstance(commit, str)
        or re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) is None
    ):
        errors.append("source.commit: expected a 7-40 character commit SHA or null")
    if not isinstance(source.get("editable"), bool):
        errors.append("source.editable: expected a boolean")

    _require_keys(
        build,
        {"variant", "cc", "cxx", "native_optimization", "jobs"},
        "build",
        errors,
    )
    variant = _validate_choice(
        build.get("variant"), ACCELERATORS, "build.variant", errors
    )
    for key in ("cc", "cxx"):
        if not _is_absolute_path(build.get(key)):
            errors.append(f"build.{key}: expected an absolute executable")
    if not isinstance(build.get("native_optimization"), bool):
        errors.append("build.native_optimization: expected a boolean")
    jobs = build.get("jobs")
    if not isinstance(jobs, int) or isinstance(jobs, bool) or not 1 <= jobs <= 256:
        errors.append("build.jobs: expected an integer from 1 through 256")
    if variant == "cuda" and not _is_absolute_path(build.get("cuda_home")):
        errors.append("build.cuda_home: CUDA build requires an absolute path")
    if variant == "rocm" and not _is_absolute_path(build.get("rocm_root")):
        errors.append("build.rocm_root: ROCm build requires an absolute path")
    if backend in {"pytorch", "tensorflow"} and variant != accelerator:
        errors.append(
            f"build.variant: {backend} source build must match accelerator {accelerator!r}"
        )


def _validate_cpp(
    cpp: dict[str, Any],
    source: dict[str, Any],
    environment: dict[str, Any],
    backend: str,
    errors: list[str],
) -> None:
    """Validate C/C++ build paths and backend roots."""
    _require_keys(cpp, {"install_prefix", "build_directory"}, "cpp", errors)
    for key in ("install_prefix", "build_directory"):
        if not _is_absolute_path(cpp.get(key)):
            errors.append(f"cpp.{key}: expected an absolute path")
    if not all(
        _is_absolute_path(cpp.get(key)) for key in ("install_prefix", "build_directory")
    ) or not _is_absolute_path(source.get("directory")):
        return
    prefix = _canonical(cpp["install_prefix"])
    build_directory = _canonical(cpp["build_directory"])
    source_directory = _canonical(source["directory"])
    if len({prefix, build_directory, source_directory}) != 3:
        errors.append("cpp: source, build, and install directories must be distinct")
    forbidden = {
        Path("/"),
        Path("/usr"),
        Path("/usr/local"),
        Path.home().resolve(strict=False),
        (Path.home() / ".local").resolve(strict=False),
    }
    environment_prefix = environment.get("prefix")
    if _is_absolute_path(environment_prefix):
        forbidden.add(_canonical(environment_prefix))
    if prefix in forbidden:
        errors.append("cpp.install_prefix: select a dedicated, non-shared prefix")
    if backend == "jax" and not (
        _is_absolute_path(cpp.get("tensorflow_root"))
        or _is_absolute_path(cpp.get("tensorflow_c_root"))
    ):
        errors.append(
            "cpp: JAX requires tensorflow_root or tensorflow_c_root for the C API"
        )
    if backend == "paddle" and not _is_absolute_path(cpp.get("paddle_inference_dir")):
        errors.append("cpp.paddle_inference_dir: Paddle C++ requires an absolute path")


def required_lammps_styles(model_family: str, flavor: str) -> list[str]:
    """Return the pair styles required by a LAMMPS plan.

    Parameters
    ----------
    model_family : str
        `conventional`, `dpa4`, or `dpa4c`.
    flavor : str
        `host` or `kokkos-cuda`.

    Returns
    -------
    list of str
        Exact pair styles that must appear in `lmp -h`.
    """
    base = "dpa4spin" if model_family == "dpa4c" else "deepmd"
    styles = [base]
    if flavor == "kokkos-cuda":
        styles.append(f"{base}/kk")
    return styles


def _validate_lammps(
    lammps: dict[str, Any],
    source: dict[str, Any],
    build: dict[str, Any],
    cpp: dict[str, Any],
    backend: str,
    accelerator: str,
    errors: list[str],
) -> None:
    """Validate source LAMMPS and Kokkos choices."""
    _require_keys(
        lammps,
        {
            "source_directory",
            "build_directory",
            "version",
            "flavor",
            "mpi",
            "model_family",
        },
        "lammps",
        errors,
    )
    for key in ("source_directory", "build_directory"):
        if not _is_absolute_path(lammps.get(key)):
            errors.append(f"lammps.{key}: expected an absolute path")
    if not isinstance(lammps.get("version"), str) or not lammps.get("version"):
        errors.append("lammps.version: expected a non-empty string")
    if not isinstance(lammps.get("mpi"), bool):
        errors.append("lammps.mpi: expected a boolean")
    flavor = _validate_choice(
        lammps.get("flavor"), LAMMPS_FLAVORS, "lammps.flavor", errors
    )
    model_family = _validate_choice(
        lammps.get("model_family"),
        MODEL_FAMILIES,
        "lammps.model_family",
        errors,
    )
    _validate_checksum(lammps.get("sha256"), "lammps.sha256", errors)
    source_directory = lammps.get("source_directory")
    if _is_absolute_path(source_directory) and not Path(source_directory).exists():
        if lammps.get("url") is None:
            errors.append("lammps.url: required when the source directory is absent")
    if lammps.get("url") is not None:
        _validate_https_url(lammps["url"], "lammps.url", errors)
    if all(
        _is_absolute_path(item)
        for item in (
            source.get("directory"),
            lammps.get("source_directory"),
            lammps.get("build_directory"),
            cpp.get("build_directory"),
            cpp.get("install_prefix"),
        )
    ):
        paths = {
            _canonical(source["directory"]),
            _canonical(lammps["source_directory"]),
            _canonical(lammps["build_directory"]),
            _canonical(cpp["build_directory"]),
            _canonical(cpp["install_prefix"]),
        }
        if len(paths) != 5:
            errors.append(
                "lammps: DeePMD source, C/C++ build/install, and LAMMPS "
                "source/build paths must differ"
            )
    if flavor == "kokkos-cuda":
        if backend != "pytorch":
            errors.append(
                "lammps.flavor: Kokkos CUDA graph pair styles require PyTorch"
            )
        if accelerator != "cuda" or build.get("variant") != "cuda":
            errors.append("lammps.flavor: Kokkos CUDA requires CUDA runtime and build")
        if (
            not isinstance(lammps.get("machine"), str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", lammps.get("machine", "")) is None
        ):
            errors.append("lammps.machine: expected a lowercase binary suffix")
        if (
            not isinstance(lammps.get("kokkos_arch"), str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]*", lammps.get("kokkos_arch", "")) is None
        ):
            errors.append("lammps.kokkos_arch: expected a Kokkos architecture name")
    if model_family in {"dpa4", "dpa4c"} and backend != "pytorch":
        errors.append(
            f"lammps.model_family: {model_family} requires the PyTorch backend"
        )


def _validate_smoke_test(
    smoke_test: dict[str, Any], accelerator: str, errors: list[str]
) -> None:
    """Validate smoke-test input and explicit GPU binding."""
    _require_keys(smoke_test, {"enabled"}, "smoke_test", errors)
    enabled = smoke_test.get("enabled")
    if not isinstance(enabled, bool):
        errors.append("smoke_test.enabled: expected a boolean")
        return
    if not enabled:
        return
    if not _is_absolute_path(smoke_test.get("example")):
        errors.append("smoke_test.example: enabled test requires an absolute path")
    gpu = smoke_test.get("gpu")
    if accelerator in {"cuda", "rocm"} and (
        not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0
    ):
        errors.append(
            f"smoke_test.gpu: {accelerator.upper()} test requires a non-negative "
            "physical index"
        )
    if accelerator == "cpu" and gpu is not None:
        errors.append("smoke_test.gpu: CPU test must use null")


def validate_plan(plan: object, *, require_resolved_source: bool = False) -> list[str]:
    """Validate an installation plan.

    Parameters
    ----------
    plan : object
        Parsed JSON value.
    require_resolved_source : bool, optional
        Require `source.commit` for a source build gate.

    Returns
    -------
    list of str
        Validation errors. An empty list means the plan is valid.
    """
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan: expected a JSON object"]
    _check_unknown_keys(plan, TOP_LEVEL_KEYS, "plan", errors)
    _require_keys(
        plan,
        {
            "schema_version",
            "method",
            "goal",
            "backend",
            "accelerator",
            "environment",
            "package",
            "smoke_test",
        },
        "plan",
        errors,
    )
    _validate_strings(plan, "", errors)
    schema_version = plan.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        errors.append(f"schema_version: expected {SCHEMA_VERSION}")
    method = _validate_choice(plan.get("method"), METHODS, "method", errors)
    goal = _validate_choice(plan.get("goal"), GOALS, "goal", errors)
    backend = _validate_choice(plan.get("backend"), BACKENDS, "backend", errors)
    accelerator = _validate_choice(
        plan.get("accelerator"), ACCELERATORS, "accelerator", errors
    )

    environment = _as_object(plan, "environment", required=True, errors=errors)
    package = _as_object(plan, "package", required=True, errors=errors)
    smoke_test = _as_object(plan, "smoke_test", required=True, errors=errors)
    if environment is not None and method is not None:
        _validate_environment(environment, method, errors)
    if (
        package is not None
        and method is not None
        and goal is not None
        and backend is not None
        and accelerator is not None
    ):
        _validate_package(package, method, goal, backend, accelerator, errors)
    if smoke_test is not None and accelerator is not None:
        _validate_smoke_test(smoke_test, accelerator, errors)

    source_required = method == "source"
    source = _as_object(plan, "source", required=source_required, errors=errors)
    build = _as_object(plan, "build", required=source_required, errors=errors)
    if (
        require_resolved_source
        and method == "source"
        and source is not None
        and source.get("commit") is None
    ):
        errors.append("source.commit: resolved source SHA is required for this gate")
    if method != "source" and (source is not None or build is not None):
        errors.append("source/build: allowed only when method is 'source'")
    if (
        source is not None
        and build is not None
        and backend is not None
        and accelerator is not None
    ):
        _validate_source(source, build, backend, accelerator, errors)

    cpp_required = goal in {"python+cpp", "python+cpp+lammps"}
    cpp = _as_object(plan, "cpp", required=cpp_required, errors=errors)
    if cpp_required and method != "source":
        errors.append("goal: C/C++ goals require method 'source'")
    if not cpp_required and cpp is not None:
        errors.append("cpp: object is allowed only for a C/C++ goal")
    if (
        cpp is not None
        and source is not None
        and environment is not None
        and backend is not None
    ):
        _validate_cpp(cpp, source, environment, backend, errors)

    source_lammps_required = goal == "python+cpp+lammps"
    lammps = _as_object(plan, "lammps", required=source_lammps_required, errors=errors)
    if source_lammps_required and method != "source":
        errors.append("goal: source LAMMPS requires method 'source'")
    if not source_lammps_required and lammps is not None:
        errors.append("lammps: object is allowed only for python+cpp+lammps")
    if (
        lammps is not None
        and source is not None
        and build is not None
        and cpp is not None
        and backend is not None
        and accelerator is not None
    ):
        _validate_lammps(lammps, source, build, cpp, backend, accelerator, errors)

    if goal == "python+lammps" and method == "source":
        errors.append("goal: source-built LAMMPS also requires the C/C++ gate")
    if goal == "python+lammps" and method not in {
        "pip",
        "conda",
        "dp1s",
        "offline",
        "docker",
    }:
        errors.append("goal: packaged LAMMPS requires an easy-install method")
    return errors


def _load_plan(path: Path) -> object:
    """Load a JSON plan from disk."""
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main(argv: list[str] | None = None) -> int:
    """Validate a plan file and print a validation summary.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    int
        Zero for a valid plan, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Path to install-plan.json.")
    parser.add_argument(
        "--json", action="store_true", help="Emit a machine-readable result."
    )
    parser.add_argument(
        "--require-resolved-source",
        action="store_true",
        help="Require source.commit before a source build gate.",
    )
    args = parser.parse_args(argv)
    try:
        plan = _load_plan(args.plan)
    except FileNotFoundError:
        errors = [f"plan: file not found: {args.plan}"]
        plan = None
    except PermissionError:
        errors = [f"plan: permission denied: {args.plan}"]
        plan = None
    except json.JSONDecodeError as exc:
        errors = [f"plan: invalid JSON at line {exc.lineno}, column {exc.colno}"]
        plan = None
    else:
        errors = validate_plan(
            plan, require_resolved_source=args.require_resolved_source
        )

    result: dict[str, Any] = {"valid": not errors, "errors": errors}
    if isinstance(plan, dict):
        result["method"] = plan.get("method")
        result["goal"] = plan.get("goal")
        result["backend"] = plan.get("backend")
        result["accelerator"] = plan.get("accelerator")
        lammps = plan.get("lammps")
        if isinstance(lammps, dict):
            family = lammps.get("model_family")
            flavor = lammps.get("flavor")
            if (
                isinstance(family, str)
                and family in MODEL_FAMILIES
                and isinstance(flavor, str)
                and flavor in LAMMPS_FLAVORS
            ):
                result["required_lammps_styles"] = required_lammps_styles(
                    family, flavor
                )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        for error in errors:
            print(f"FAIL  {error}")
    else:
        print(
            "PASS  plan: "
            f"method={result['method']} goal={result['goal']} "
            f"backend={result['backend']} accelerator={result['accelerator']}"
        )
        if "required_lammps_styles" in result:
            print(
                "PASS  required_lammps_styles: "
                + ", ".join(result["required_lammps_styles"])
            )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
