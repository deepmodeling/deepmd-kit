# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the DeePMD-kit installation skill helpers."""

from __future__ import (
    annotations,
)

import importlib.util
import json
import subprocess
import sys
from pathlib import (
    Path,
)
from types import (
    ModuleType,
    SimpleNamespace,
)

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "skills" / "deepmd-install" / "scripts"


def _load_script(name: str) -> ModuleType:
    """Load one helper script as a module."""
    path = SCRIPT_DIRECTORY / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"deepmd_install_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PLAN = _load_script("validate_plan")
PREPARE = _load_script("prepare_lammps")
NATIVE = _load_script("verify_native")
VERIFY = _load_script("verify_python")


def _source_plan() -> dict[str, object]:
    """Return a valid source CUDA DPA4C plan."""
    return {
        "schema_version": 1,
        "method": "source",
        "goal": "python+cpp+lammps",
        "backend": "pytorch",
        "accelerator": "cuda",
        "environment": {
            "kind": "existing",
            "python": "/opt/deepmd/bin/python",
            "manager": None,
            "name": None,
            "prefix": "/opt/deepmd",
        },
        "package": {
            "deepmd_version": None,
            "deepmd_index_url": None,
            "deepmd_extra_index_url": None,
            "backend_packages": [],
            "backend_index_url": None,
            "channels": [],
            "install_lammps": False,
            "install_ipi": False,
            "artifact_url": None,
            "artifact_path": None,
            "sha256": None,
            "docker_image": None,
            "lammps_model_family": None,
        },
        "source": {
            "directory": "/work/deepmd-kit",
            "remote": "https://github.com/deepmodeling/deepmd-kit.git",
            "ref": "master",
            "commit": None,
        },
        "build": {
            "variant": "cuda",
            "cc": "/usr/bin/gcc",
            "cxx": "/usr/bin/g++",
            "cuda_home": "/usr/local/cuda",
            "rocm_root": None,
            "native_optimization": False,
            "jobs": 8,
        },
        "cpp": {
            "install_prefix": "/work/install/deepmd",
            "build_directory": "/work/build/deepmd-cpp",
            "tensorflow_root": None,
            "tensorflow_c_root": None,
            "paddle_inference_dir": None,
        },
        "lammps": {
            "source_directory": "/work/lammps",
            "build_directory": "/work/build/lammps-blackwell120",
            "version": "stable_22Jul2025_update2",
            "url": "https://github.com/lammps/lammps/archive/refs/tags/stable_22Jul2025_update2.tar.gz",
            "sha256": "a" * 64,
            "flavor": "kokkos-cuda",
            "machine": "blackwell120",
            "kokkos_arch": "BLACKWELL120",
            "mpi": False,
            "model_family": "dpa4c",
        },
        "smoke_test": {"enabled": False, "gpu": None, "example": None},
    }


def _easy_plan(method: str) -> dict[str, object]:
    """Return a valid Python-only easy-install plan."""
    environment: dict[str, object] = {
        "kind": "existing",
        "python": "/opt/deepmd/bin/python",
        "manager": None,
        "name": None,
        "prefix": "/opt/deepmd",
    }
    package: dict[str, object] = {
        "deepmd_version": None,
        "deepmd_index_url": None,
        "deepmd_extra_index_url": None,
        "backend_packages": [],
        "backend_index_url": None,
        "channels": [],
        "install_lammps": False,
        "install_ipi": False,
        "artifact_url": None,
        "artifact_path": None,
        "sha256": None,
        "docker_image": None,
        "lammps_model_family": None,
    }
    if method == "conda":
        environment.update(
            {"kind": "conda", "python": None, "manager": "/opt/conda", "name": "deepmd"}
        )
    elif method == "offline":
        environment.update({"kind": "prefix", "python": None})
        package.update(
            {
                "artifact_url": "https://example.invalid/deepmd.sh",
                "sha256": "a" * 64,
            }
        )
    elif method == "docker":
        environment.update({"kind": "container", "python": None, "prefix": None})
        package["docker_image"] = "ghcr.io/deepmodeling/deepmd-kit:master"
    return {
        "schema_version": 1,
        "method": method,
        "goal": "python",
        "backend": "tensorflow",
        "accelerator": "cpu",
        "environment": environment,
        "package": package,
        "source": None,
        "build": None,
        "cpp": None,
        "lammps": None,
        "smoke_test": {"enabled": False, "gpu": None, "example": None},
    }


def test_validate_source_dpa4c_plan() -> None:
    """Accept a complete PyTorch CUDA DPA4C plan."""
    plan = _source_plan()
    assert PLAN.validate_plan(plan) == []
    assert PLAN.required_lammps_styles("dpa4c", "kokkos-cuda") == [
        "dpa4spin",
        "dpa4spin/kk",
    ]


def test_validate_source_build_gate_requires_resolved_commit() -> None:
    """Require the resolved source identity before a source build gate."""
    plan = _source_plan()
    errors = PLAN.validate_plan(plan, require_resolved_source=True)
    assert "source.commit: resolved source SHA is required for this gate" in errors
    source = plan["source"]
    assert isinstance(source, dict)
    source["commit"] = "ed691aab147d9d7686d296e30f46902c08e9fb68"
    assert PLAN.validate_plan(plan, require_resolved_source=True) == []


def test_validate_source_jax_gpu_with_cpu_custom_ops() -> None:
    """Allow JAX to provide GPU execution independently of custom OPs."""
    plan = _source_plan()
    plan["goal"] = "python"
    plan["backend"] = "jax"
    build = plan["build"]
    assert isinstance(build, dict)
    build["variant"] = "cpu"
    build["cuda_home"] = None
    plan["cpp"] = None
    plan["lammps"] = None
    assert PLAN.validate_plan(plan) == []


def test_validate_source_tensorflow_cpu_plan() -> None:
    """Accept a TensorFlow source build with matching CPU custom operations."""
    plan = _source_plan()
    plan["goal"] = "python"
    plan["backend"] = "tensorflow"
    plan["accelerator"] = "cpu"
    build = plan["build"]
    assert isinstance(build, dict)
    build["variant"] = "cpu"
    build["cuda_home"] = None
    plan["cpp"] = None
    plan["lammps"] = None
    assert PLAN.validate_plan(plan) == []


def test_validate_jax_cpp_with_tensorflow_python_libraries() -> None:
    """Allow JAX C++ to use TensorFlow libraries from the Python environment."""
    plan = _source_plan()
    plan["goal"] = "python+cpp"
    plan["backend"] = "jax"
    plan["lammps"] = None
    assert PLAN.validate_plan(plan) == []


def test_validate_jax_cpp_rejects_ambiguous_tensorflow_roots() -> None:
    """Require one unambiguous TensorFlow dependency route for JAX C++."""
    plan = _source_plan()
    plan["goal"] = "python+cpp"
    plan["backend"] = "jax"
    plan["lammps"] = None
    cpp = plan["cpp"]
    assert isinstance(cpp, dict)
    cpp["tensorflow_root"] = "/opt/tensorflow-cpp"
    cpp["tensorflow_c_root"] = "/opt/tensorflow-c"
    errors = PLAN.validate_plan(plan)
    assert any("mutually exclusive" in error for error in errors)


@pytest.mark.parametrize("method", ["pip", "conda", "offline", "docker"])
def test_validate_easy_install_plans(method: str) -> None:
    """Accept the method-specific fields for each easy-install path."""
    assert PLAN.validate_plan(_easy_plan(method)) == []


def test_validate_conda_channels_are_method_specific() -> None:
    """Accept selected conda channels without ignoring them for other methods."""
    plan = _easy_plan("conda")
    package = plan["package"]
    assert isinstance(package, dict)
    package["channels"] = ["conda-forge/label/deepmd-kit_rc", "conda-forge"]
    assert PLAN.validate_plan(plan) == []

    pip_plan = _easy_plan("pip")
    pip_package = pip_plan["package"]
    assert isinstance(pip_package, dict)
    pip_package["channels"] = ["conda-forge"]
    errors = PLAN.validate_plan(pip_plan)
    assert "package.channels: non-empty channels require method 'conda'" in errors


def test_validate_offline_plan_requires_checksum() -> None:
    """Reject an offline artifact without an integrity value."""
    plan = _easy_plan("offline")
    package = plan["package"]
    assert isinstance(package, dict)
    package["sha256"] = None
    errors = PLAN.validate_plan(plan)
    assert "package.sha256: offline installation requires a checksum" in errors


def test_validate_offline_local_artifact() -> None:
    """Accept a local offline artifact only through its dedicated path field."""
    plan = _easy_plan("offline")
    package = plan["package"]
    assert isinstance(package, dict)
    package["artifact_url"] = None
    package["artifact_path"] = "/opt/artifacts/deepmd.sh"
    assert PLAN.validate_plan(plan) == []


def test_validate_offline_artifact_url_requires_https() -> None:
    """Reject a local path routed through the offline curl branch."""
    plan = _easy_plan("offline")
    package = plan["package"]
    assert isinstance(package, dict)
    package["artifact_url"] = "/opt/artifacts/deepmd.sh"
    errors = PLAN.validate_plan(plan)
    assert any(
        "package.artifact_url: expected an HTTPS URL" in error for error in errors
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("method",), []),
        (("goal",), {}),
        (("environment", "kind"), []),
        (("build", "variant"), {}),
        (("lammps", "flavor"), []),
        (("lammps", "model_family"), {}),
    ],
)
def test_validate_plan_rejects_non_string_enums(
    path: tuple[str, ...], value: object
) -> None:
    """Convert malformed JSON enum types into validation errors."""
    plan = _source_plan()
    target: dict[str, object] = plan
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    errors = PLAN.validate_plan(plan)
    assert any("unsupported value" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("source", "ref", "v<ref>"),
        ("package", "backend_packages", ["torch==<version>"]),
        ("package", "docker_image", "image:<tag>"),
        ("environment", "python", "/opt/<user>/python"),
    ],
)
def test_validate_plan_rejects_embedded_placeholders(
    section: str, field: str, value: object
) -> None:
    """Reject placeholder tokens embedded inside otherwise valid values."""
    plan = _source_plan()
    target = plan[section]
    assert isinstance(target, dict)
    target[field] = value
    errors = PLAN.validate_plan(plan)
    assert any("unresolved placeholder" in error for error in errors)


@pytest.mark.parametrize("value", ['bad"value', "bad`value", "bad\nvalue"])
def test_validate_plan_rejects_unsafe_shell_template_values(value: str) -> None:
    """Reject values that can escape the documented POSIX templates."""
    plan = _source_plan()
    source = plan["source"]
    assert isinstance(source, dict)
    source["remote"] = value
    errors = PLAN.validate_plan(plan)
    assert any(
        "unsafe shell-template character" in error or "control character" in error
        for error in errors
    )


def test_validate_plan_keeps_quoted_semicolon_as_data() -> None:
    """Allow semicolons that remain inside the documented quoted argument."""
    plan = _source_plan()
    source = plan["source"]
    assert isinstance(source, dict)
    source["remote"] = "https://example.invalid/repository;mirror.git"
    assert PLAN.validate_plan(plan) == []


def test_validate_plan_applies_platform_specific_backslash_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve Windows paths and reject POSIX shell-escape backslashes."""
    windows_errors: list[str] = []
    monkeypatch.setattr(PLAN, "_platform_name", lambda: "nt")
    PLAN._validate_strings(
        r"C:\DeePMD\python.exe", "environment.python", windows_errors
    )
    assert windows_errors == []

    posix_errors: list[str] = []
    monkeypatch.setattr(PLAN, "_platform_name", lambda: "posix")
    PLAN._validate_strings(r"C:\DeePMD\python.exe", "environment.python", posix_errors)
    assert any("unsafe shell-template character" in item for item in posix_errors)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend_index_url", "/mirror"),
        ("deepmd_index_url", "file:///mirror"),
        ("deepmd_extra_index_url", "http://example.invalid/simple"),
    ],
)
def test_validate_plan_requires_https_package_indexes(field: str, value: str) -> None:
    """Reject local paths and non-HTTPS package indexes."""
    plan = _easy_plan("pip")
    package = plan["package"]
    assert isinstance(package, dict)
    package[field] = value
    errors = PLAN.validate_plan(plan)
    assert any("expected an HTTPS URL" in error for error in errors)


def test_validate_plan_requires_https_lammps_url() -> None:
    """Reject a local path rendered through the LAMMPS curl branch."""
    plan = _source_plan()
    lammps = plan["lammps"]
    assert isinstance(lammps, dict)
    lammps["url"] = "/opt/lammps.tar.gz"
    errors = PLAN.validate_plan(plan)
    assert any("lammps.url: expected an HTTPS URL" in error for error in errors)


def test_validate_plan_requires_checksum_for_lammps_download(tmp_path: Path) -> None:
    """Reject an unverified archive when the LAMMPS source is absent."""
    plan = _source_plan()
    lammps = plan["lammps"]
    assert isinstance(lammps, dict)
    lammps["source_directory"] = str(tmp_path / "missing-lammps")
    lammps["sha256"] = None
    errors = PLAN.validate_plan(plan)
    assert "lammps.sha256: required when the source directory is absent" in errors


def test_validate_plan_allows_existing_lammps_without_archive(
    tmp_path: Path,
) -> None:
    """Allow a verified existing source directory without download fields."""
    source_directory = tmp_path / "lammps"
    source_directory.mkdir()
    plan = _source_plan()
    lammps = plan["lammps"]
    assert isinstance(lammps, dict)
    lammps.update(
        {"source_directory": str(source_directory), "url": None, "sha256": None}
    )
    assert PLAN.validate_plan(plan) == []


def test_validate_plan_rejects_lammps_source_file(tmp_path: Path) -> None:
    """Reject an existing file where a LAMMPS source directory is required."""
    source_path = tmp_path / "lammps"
    source_path.write_text("not a source tree", encoding="utf-8")
    plan = _source_plan()
    lammps = plan["lammps"]
    assert isinstance(lammps, dict)
    lammps["source_directory"] = str(source_path)
    errors = PLAN.validate_plan(plan)
    assert "lammps.source_directory: existing path is not a directory" in errors


@pytest.mark.parametrize("field", ["build_directory", "install_prefix"])
def test_validate_plan_rejects_lammps_cpp_path_collisions(field: str) -> None:
    """Keep LAMMPS source/build paths out of the C/C++ build and prefix."""
    plan = _source_plan()
    cpp = plan["cpp"]
    lammps = plan["lammps"]
    assert isinstance(cpp, dict)
    assert isinstance(lammps, dict)
    lammps["build_directory"] = cpp[field]
    errors = PLAN.validate_plan(plan)
    assert any("C/C++ build/install" in error for error in errors)


@pytest.mark.parametrize("cpp_field", ["build_directory", "install_prefix"])
def test_validate_plan_rejects_lammps_source_in_cpp_paths(cpp_field: str) -> None:
    """Keep the LAMMPS source tree out of C/C++ build and install paths."""
    plan = _source_plan()
    cpp = plan["cpp"]
    lammps = plan["lammps"]
    assert isinstance(cpp, dict)
    assert isinstance(lammps, dict)
    lammps["source_directory"] = cpp[cpp_field]
    errors = PLAN.validate_plan(plan)
    assert any("C/C++ build/install" in error for error in errors)


def test_validate_plan_rejects_easy_rocm() -> None:
    """Route ROCm installations through the source workflow."""
    plan = _easy_plan("pip")
    plan["accelerator"] = "rocm"
    errors = PLAN.validate_plan(plan)
    assert "accelerator: ROCm requires method 'source'" in errors


@pytest.mark.parametrize("feature", ["lammps", "ipi"])
def test_validate_plan_rejects_paddle_packaged_native_tools(feature: str) -> None:
    """Reject packaged LAMMPS and i-PI for the unsupported Paddle backend."""
    plan = _easy_plan("pip")
    plan["backend"] = "paddle"
    package = plan["package"]
    assert isinstance(package, dict)
    if feature == "lammps":
        plan["goal"] = "python+lammps"
        package["install_lammps"] = True
        package["lammps_model_family"] = "conventional"
    else:
        package["install_ipi"] = True
    errors = PLAN.validate_plan(plan)
    assert "package: Paddle does not support packaged LAMMPS or i-PI" in errors


def test_validate_packaged_lammps_requires_model_family() -> None:
    """Require exact pair-style identity for packaged LAMMPS."""
    plan = _easy_plan("pip")
    plan["goal"] = "python+lammps"
    package = plan["package"]
    assert isinstance(package, dict)
    package["install_lammps"] = True
    errors = PLAN.validate_plan(plan)
    assert any("package.lammps_model_family" in error for error in errors)
    package["lammps_model_family"] = "dpa4c"
    plan["backend"] = "pytorch"
    assert PLAN.validate_plan(plan) == []


def test_validate_rocm_smoke_test_requires_physical_gpu() -> None:
    """Require explicit ROCm device binding for an enabled smoke test."""
    plan = _source_plan()
    plan["goal"] = "python"
    plan["accelerator"] = "rocm"
    plan["backend"] = "jax"
    build = plan["build"]
    smoke_test = plan["smoke_test"]
    assert isinstance(build, dict)
    assert isinstance(smoke_test, dict)
    build.update({"variant": "cpu", "cuda_home": None})
    plan["cpp"] = None
    plan["lammps"] = None
    smoke_test.update({"enabled": True, "gpu": None, "example": "/work/example"})
    errors = PLAN.validate_plan(plan)
    assert any(
        "ROCM test requires a non-negative physical index" in error for error in errors
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("backend", "tensorflow"), "Kokkos CUDA graph pair styles require PyTorch"),
        (("source", {"directory": "$HOME/deepmd"}), "unresolved shell variable"),
        (("unknown", True), "unknown field"),
    ],
)
def test_validate_plan_rejects_invalid_input(
    mutation: tuple[str, object], message: str
) -> None:
    """Reject unsupported combinations, placeholders, and unknown keys."""
    plan = _source_plan()
    key, value = mutation
    if key == "source":
        source = plan["source"]
        assert isinstance(source, dict)
        source.update(value)
    else:
        plan[key] = value
    errors = PLAN.validate_plan(plan)
    assert any(message in error for error in errors)


def test_validate_plan_rejects_shared_cpp_prefix() -> None:
    """Reject a C/C++ install into the selected Python environment."""
    plan = _source_plan()
    cpp = plan["cpp"]
    assert isinstance(cpp, dict)
    cpp["install_prefix"] = "/opt/deepmd"
    errors = PLAN.validate_plan(plan)
    assert "cpp.install_prefix: select a dedicated, non-shared prefix" in errors


def test_prepare_lammps_is_idempotent(tmp_path: Path) -> None:
    """Create one quoted managed include and preserve it on a second run."""
    lammps = tmp_path / "lammps tree"
    deepmd = tmp_path / "deepmd tree"
    cmake_file = lammps / "cmake" / "CMakeLists.txt"
    builtin = deepmd / "source" / "lmp" / "builtin.cmake"
    cmake_file.parent.mkdir(parents=True)
    builtin.parent.mkdir(parents=True)
    cmake_file.write_text("cmake_minimum_required(VERSION 3.25)\n", encoding="utf-8")
    builtin.write_text("# builtin\n", encoding="utf-8")

    assert PREPARE.prepare(lammps, deepmd, check=False) is False
    updated = cmake_file.read_text(encoding="utf-8")
    assert updated.count(PREPARE.BEGIN_MARKER) == 1
    assert f"include([=[{builtin}]=])" in updated
    assert PREPARE.prepare(lammps, deepmd, check=True) is True


def test_prepare_lammps_replaces_one_legacy_include() -> None:
    """Replace a legacy include without retaining a duplicate."""
    builtin = Path("/work/deepmd/source/lmp/builtin.cmake")
    original = "include(/old/deepmd/source/lmp/builtin.cmake)\n"
    updated = PREPARE.render_updated_text(original, builtin)
    assert "/old/deepmd" not in updated
    assert updated.count("source/lmp/builtin.cmake") == 1


def test_prepare_lammps_rejects_duplicate_legacy_includes() -> None:
    """Reject an ambiguous source tree with multiple unmanaged includes."""
    original = "\n".join(
        (
            "include(/one/source/lmp/builtin.cmake)",
            "include(/two/source/lmp/builtin.cmake)",
        )
    )
    with pytest.raises(ValueError, match="multiple DeePMD includes"):
        PREPARE.render_updated_text(
            original, Path("/work/deepmd/source/lmp/builtin.cmake")
        )


def test_prepare_lammps_rejects_managed_and_unmanaged_includes() -> None:
    """Reject an extra include outside the managed block."""
    builtin = Path("/work/deepmd/source/lmp/builtin.cmake")
    original = "\n".join(
        (
            PREPARE.BEGIN_MARKER,
            f"include([=[{builtin}]=])",
            PREPARE.END_MARKER,
            "include(/other/source/lmp/builtin.cmake)",
        )
    )
    with pytest.raises(ValueError, match="unmanaged DeePMD include"):
        PREPARE.render_updated_text(original, builtin)


def _write_fake_lammps(path: Path, styles: str) -> None:
    """Write an executable that emits a deterministic LAMMPS help surface."""
    path.write_text(f"#!/usr/bin/env python3\nprint({styles!r})\n", encoding="utf-8")
    path.chmod(0o755)


def test_native_link_check_rejects_not_found_with_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when ldd reports an unresolved library despite return code zero."""
    binary = tmp_path / "libexample.so"
    binary.write_bytes(b"native")
    monkeypatch.setattr(NATIVE, "_system_name", lambda: "Linux")
    monkeypatch.setattr(NATIVE, "_find_executable", lambda _name: "/usr/bin/ldd")
    monkeypatch.setattr(
        NATIVE,
        "_run",
        lambda _command: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="libmissing.so => not found\n", stderr=""
        ),
    )
    result = NATIVE.check_dynamic_links(binary)
    assert result.passed is False
    assert "libmissing.so => not found" in result.detail


def test_native_link_check_uses_dyld_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail when the isolated Darwin loader cannot resolve a dependency."""
    library = tmp_path / "libexample.dylib"
    library.write_bytes(b"native")
    commands: list[list[str]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr="OSError: Library not loaded: @rpath/libmissing.dylib",
        )

    monkeypatch.setattr(NATIVE, "_system_name", lambda: "Darwin")
    monkeypatch.setattr(NATIVE, "_run", run)
    result = NATIVE.check_dynamic_links(library)
    assert result.passed is False
    assert "Library not loaded" in result.detail
    assert commands[0][:2] == [sys.executable, "-c"]
    assert "otool" not in commands[0]


def test_native_link_default_pattern_matches_platform() -> None:
    """Select the native-library suffix without changing explicit patterns."""
    assert NATIVE._default_pattern("Linux") == "libdeepmd*.so"
    assert NATIVE._default_pattern("Darwin") == "libdeepmd*.dylib"


def test_native_link_collection_preserves_explicit_pattern(tmp_path: Path) -> None:
    """Use a caller-provided directory pattern without platform substitution."""
    selected = tmp_path / "libdeepmd.custom"
    ignored = tmp_path / "libdeepmd.so"
    selected.write_bytes(b"native")
    ignored.write_bytes(b"native")
    assert NATIVE._collect_paths([], tmp_path, "*.custom") == [selected.resolve()]


def test_native_link_check_fails_closed_on_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject an unimplemented platform instead of reporting false success."""
    library = tmp_path / "deepmd.dll"
    library.write_bytes(b"native")
    monkeypatch.setattr(NATIVE, "_system_name", lambda: "Windows")
    result = NATIVE.check_dynamic_links(library)
    assert result.passed is False
    assert result.detail == "unsupported platform: Windows"


def test_verify_python_checks_version_and_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Require both interpreter and imported module to use the planned prefix."""
    environment = tmp_path / "environment"
    module_file = environment / "lib" / "deepmd" / "__init__.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")
    fake_deepmd = SimpleNamespace(__version__="3.2.0", __file__=str(module_file))
    monkeypatch.setitem(sys.modules, "deepmd", fake_deepmd)
    monkeypatch.setattr(VERIFY, "_interpreter_prefix", lambda: environment.resolve())
    passing = VERIFY._check_deepmd("3.2.0", str(environment))
    wrong_version = VERIFY._check_deepmd("3.1.0", str(environment))
    wrong_prefix = VERIFY._check_deepmd("3.2.0", str(tmp_path / "other"))
    assert passing.passed is True
    assert wrong_version.passed is False
    assert wrong_prefix.passed is False


def test_verify_python_rejects_shadowed_or_missing_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject a checkout import even when the interpreter prefix is correct."""
    environment = tmp_path / "environment"
    checkout_file = tmp_path / "checkout" / "deepmd" / "__init__.py"
    checkout_file.parent.mkdir(parents=True)
    checkout_file.write_text("", encoding="utf-8")
    fake_deepmd = SimpleNamespace(__version__="3.2.0", __file__=str(checkout_file))
    monkeypatch.setitem(sys.modules, "deepmd", fake_deepmd)
    monkeypatch.setattr(VERIFY, "_interpreter_prefix", lambda: environment.resolve())
    assert VERIFY._check_deepmd("3.2.0", str(environment)).passed is False
    fake_deepmd.__file__ = None
    assert VERIFY._check_deepmd("3.2.0", str(environment)).passed is False


def test_verify_python_matches_abbreviated_source_commit() -> None:
    """Match the abbreviated commit stored in build metadata to a full SHA."""
    assert VERIFY._commits_match("e59966be", "e59966be1234567890abcdef1234567890abcdef")
    assert not VERIFY._commits_match(
        "e59966be", "a59966be1234567890abcdef1234567890abcdef"
    )


@pytest.mark.parametrize(
    ("build_info", "expected"),
    [
        ({"is_cuda_build": True, "is_rocm_build": False}, "cuda"),
        ({"is_cuda_build": False, "is_rocm_build": True}, "rocm"),
    ],
)
def test_tensorflow_runtime_detection(
    build_info: dict[str, bool], expected: str
) -> None:
    """Distinguish TensorFlow CUDA and ROCm build metadata."""
    tensorflow = SimpleNamespace(
        sysconfig=SimpleNamespace(get_build_info=lambda: build_info)
    )
    assert VERIFY._tensorflow_accelerator(tensorflow) == expected
    assert VERIFY._tensorflow_accelerator(tensorflow) != (
        "rocm" if expected == "cuda" else "cuda"
    )


@pytest.mark.parametrize(
    ("requested", "build_info"),
    [
        ("cuda", {"is_cuda_build": False, "is_rocm_build": True}),
        ("rocm", {"is_cuda_build": True, "is_rocm_build": False}),
    ],
)
def test_tensorflow_rejects_wrong_gpu_runtime(
    requested: str,
    build_info: dict[str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail TensorFlow verification when GPU presence masks a runtime mismatch."""
    deepmd = ModuleType("deepmd")
    deepmd.__path__ = []  # type: ignore[attr-defined]
    deepmd_tf = ModuleType("deepmd.tf")
    tensorflow = ModuleType("tensorflow")
    tensorflow.__version__ = "test"  # type: ignore[attr-defined]
    tensorflow.sysconfig = SimpleNamespace(  # type: ignore[attr-defined]
        get_build_info=lambda: build_info
    )
    tensorflow.config = SimpleNamespace(  # type: ignore[attr-defined]
        list_physical_devices=lambda _kind: [object()]
    )
    monkeypatch.setitem(sys.modules, "deepmd", deepmd)
    monkeypatch.setitem(sys.modules, "deepmd.tf", deepmd_tf)
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    checks = VERIFY._check_tensorflow(requested)
    assert checks[-1].passed is False
    assert f"expected={requested}" in checks[-1].detail


@pytest.mark.parametrize(
    ("client_platform", "device_kind", "platform_version", "expected"),
    [
        ("gpu", "NVIDIA RTX PRO 6000", "CUDA 13.0", "cuda"),
        ("gpu", "AMD Instinct MI300X", "ROCm 7.0", "rocm"),
        ("gpu", "AMD-compatible adapter", "CUDA 13.0", "cuda"),
        ("cuda", "vendor text is not authoritative", "unknown", "cuda"),
    ],
)
def test_jax_runtime_detection(
    client_platform: str,
    device_kind: str,
    platform_version: str,
    expected: str,
) -> None:
    """Distinguish JAX CUDA and ROCm client metadata."""
    device = SimpleNamespace(
        platform="gpu",
        device_kind=device_kind,
        client=SimpleNamespace(
            platform=client_platform, platform_version=platform_version
        ),
    )
    assert VERIFY._jax_accelerator([device]) == expected
    assert VERIFY._jax_accelerator([device]) != (
        "rocm" if expected == "cuda" else "cuda"
    )


def test_jax_runtime_detection_fails_closed_on_conflicting_backends() -> None:
    """Reject a device set that reports both CUDA and ROCm clients."""
    devices = [
        SimpleNamespace(
            platform="gpu",
            client=SimpleNamespace(platform=runtime, platform_version=runtime),
        )
        for runtime in ("cuda", "rocm")
    ]
    assert VERIFY._jax_accelerator(devices) is None


@pytest.mark.parametrize(
    ("requested", "device_kind", "platform_version"),
    [
        ("cuda", "AMD Instinct MI300X", "ROCm 7.0"),
        ("rocm", "NVIDIA RTX PRO 6000", "CUDA 13.0"),
    ],
)
def test_jax_rejects_wrong_gpu_runtime(
    requested: str,
    device_kind: str,
    platform_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail JAX verification when a GPU uses the other runtime."""
    deepmd = ModuleType("deepmd")
    deepmd.__path__ = []  # type: ignore[attr-defined]
    deepmd_jax = ModuleType("deepmd.jax")
    jax = ModuleType("jax")
    jax.__path__ = []  # type: ignore[attr-defined]
    jax.__version__ = "test"  # type: ignore[attr-defined]
    device = SimpleNamespace(
        platform="gpu",
        device_kind=device_kind,
        client=SimpleNamespace(platform="gpu", platform_version=platform_version),
    )
    jax.devices = lambda: [device]  # type: ignore[attr-defined]
    jax_numpy = ModuleType("jax.numpy")
    monkeypatch.setitem(sys.modules, "deepmd", deepmd)
    monkeypatch.setitem(sys.modules, "deepmd.jax", deepmd_jax)
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "jax.numpy", jax_numpy)
    checks = VERIFY._check_jax(requested)
    assert checks[-1].passed is False
    assert f"expected={requested}" in checks[-1].detail


def test_docker_reference_uses_backend_aware_verifier() -> None:
    """Keep Docker verification backend-aware and read-only mounted."""
    reference = (
        REPOSITORY_ROOT / "skills" / "deepmd-install" / "references" / "easy-install.md"
    ).read_text(encoding="utf-8")
    assert "verify_python.py" in reference
    assert '--backend "<pytorch|tensorflow|jax|paddle>"' in reference
    assert "readonly" in reference


def test_conda_reference_renders_planned_channels() -> None:
    """Keep selected conda channels authoritative with a stable default."""
    reference = (
        REPOSITORY_ROOT / "skills" / "deepmd-install" / "references" / "easy-install.md"
    ).read_text(encoding="utf-8")
    assert "package.channels" in reference
    assert '-c "<channel-1>" -c "<channel-2>"' in reference
    assert "Use `conda-forge` only when the list is empty" in reference


def test_verify_lammps_dpa4c_cli(tmp_path: Path) -> None:
    """Accept the exact DPA4C host and Kokkos pair styles."""
    binary = tmp_path / "lmp_dpa4c"
    _write_fake_lammps(binary, "Pair styles: dpa4spin dpa4spin/kk")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIRECTORY / "verify_lammps.py"),
            "--binary",
            str(binary),
            "--model-family",
            "dpa4c",
            "--flavor",
            "kokkos-cuda",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS  pair_style:dpa4spin/kk" in completed.stdout


def test_verify_lammps_requires_exact_style_token(tmp_path: Path) -> None:
    """Reject a near-name token instead of accepting a substring."""
    binary = tmp_path / "lmp_near_name"
    _write_fake_lammps(binary, "Pair styles: deepmd deepmd/kkx")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIRECTORY / "verify_lammps.py"),
            "--binary",
            str(binary),
            "--model-family",
            "dpa4",
            "--flavor",
            "kokkos-cuda",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "FAIL  pair_style:deepmd/kk" in completed.stdout


def test_verify_python_rejects_backend_incompatible_checks() -> None:
    """Reject PyTorch-only flags before importing another backend."""
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIRECTORY / "verify_python.py"),
            "--backend",
            "jax",
            "--accelerator",
            "cpu",
            "--expect-custom-op",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "PyTorch-only checks require --backend pytorch" in completed.stderr


def test_probe_cli_emits_json() -> None:
    """Emit the stable top-level probe schema without requiring optional tools."""
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_DIRECTORY / "probe_env.py"), "--json"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert set(report) == {
        "system",
        "environment",
        "python",
        "tools",
        "nvidia",
        "rocm",
        "packages",
    }
    assert Path(report["python"]["executable"]).is_absolute()
