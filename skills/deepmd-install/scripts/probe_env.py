#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Probe machine facts required to plan a DeePMD-kit installation."""

from __future__ import (
    annotations,
)

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )

ENVIRONMENT_VARIABLES = (
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDAToolkit_ROOT",
    "ROCM_ROOT",
    "ROCM_PATH",
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
    "CC",
    "CXX",
    "CUDACXX",
    "CUDAHOSTCXX",
    "DP_VARIANT",
    "DP_ENABLE_PYTORCH",
    "DP_ENABLE_TENSORFLOW",
    "PATH",
)
TOOLS = (
    "python",
    "python3",
    "pip",
    "conda",
    "mamba",
    "git",
    "cmake",
    "ninja",
    "gcc",
    "g++",
    "clang",
    "clang++",
    "nvcc",
    "hipcc",
    "mpicxx",
    "nvidia-smi",
    "rocm-smi",
    "rocminfo",
    "docker",
    "dp",
    "lmp",
)
VERSION_TOOLS = {
    "pip": ("--version",),
    "conda": ("--version",),
    "mamba": ("--version",),
    "git": ("--version",),
    "cmake": ("--version",),
    "ninja": ("--version",),
    "gcc": ("--version",),
    "g++": ("--version",),
    "clang": ("--version",),
    "clang++": ("--version",),
    "nvcc": ("--version",),
    "hipcc": ("--version",),
    "mpicxx": ("--version",),
    "docker": ("--version",),
}
PACKAGE_DISTRIBUTIONS = {
    "deepmd": ("deepmd-kit",),
    "pytorch": ("torch",),
    "tensorflow": ("tensorflow", "tensorflow-cpu"),
    "jax": ("jax",),
    "paddle": ("paddlepaddle", "paddlepaddle-gpu"),
}


def _run(
    command: Sequence[str], *, timeout: float = 10.0, cwd: Path | None = None
) -> dict[str, Any]:
    """Run a read-only command and return a structured result."""
    executable = command[0]
    resolved = shutil.which(executable)
    if resolved is None and not Path(executable).is_file():
        return {
            "status": "unavailable",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "output": "",
        }
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "output": "",
        }
    except OSError as exc:
        return {
            "status": "error",
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "output": f"{type(exc).__name__}: {exc}",
        }
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    output = "\n".join(part for part in (stdout, stderr) if part)
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output": output,
    }


def _first_line(value: str) -> str:
    """Return the first non-empty output line."""
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def _probe_system() -> dict[str, Any]:
    """Return operating-system, memory, CPU, and disk facts."""
    libc_name, libc_version = platform.libc_ver()
    memory_total: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    memory_total = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, IndexError):
            memory_total = None
    elif platform.system() == "Darwin":
        result = _run(["sysctl", "-n", "hw.memsize"])
        if result["status"] == "ok":
            try:
                memory_total = int(result["output"])
            except ValueError:
                memory_total = None
    disk = shutil.disk_usage(Path.cwd())
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": {"name": libc_name or None, "version": libc_version or None},
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": memory_total,
        "working_directory": str(Path.cwd()),
        "disk_free_bytes": disk.free,
    }


def _probe_python() -> dict[str, Any]:
    """Return facts for the interpreter executing this probe."""
    pip_result = _run([sys.executable, "-m", "pip", "--version"])
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "pip": pip_result,
    }


def _probe_tools() -> dict[str, Any]:
    """Return executable paths and concise version strings."""
    result: dict[str, Any] = {}
    for name in TOOLS:
        path = shutil.which(name)
        entry: dict[str, Any] = {"path": path}
        if path is not None and name in VERSION_TOOLS:
            version = _run([path, *VERSION_TOOLS[name]])
            entry["version"] = (
                _first_line(version["output"])
                if version["status"] in {"ok", "failed"}
                else None
            )
            entry["version_status"] = version["status"]
        result[name] = entry
    return result


def _query_nvidia(fields: Sequence[str]) -> dict[str, Any]:
    """Query NVIDIA GPUs with a fixed CSV field list."""
    result = _run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if result["status"] != "ok":
        return {"status": result["status"], "error": result["output"], "gpus": []}
    rows = list(csv.reader(result["output"].splitlines(), skipinitialspace=True))
    gpus = [dict(zip(fields, row, strict=False)) for row in rows if row]
    return {"status": "ok", "error": None, "gpus": gpus}


def _probe_nvidia() -> dict[str, Any]:
    """Return driver, occupancy, and numeric compute capability per GPU."""
    fields = (
        "index",
        "uuid",
        "name",
        "driver_version",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "compute_cap",
    )
    result = _query_nvidia(fields)
    if result["status"] == "ok":
        return result
    fallback_fields = fields[:-1]
    fallback = _query_nvidia(fallback_fields)
    if fallback["status"] == "ok":
        for gpu in fallback["gpus"]:
            gpu["compute_cap"] = None
        fallback["compute_capability_status"] = "unsupported by nvidia-smi"
    return fallback


def _probe_rocm() -> dict[str, Any]:
    """Return ROCm tool availability and concise device output."""
    hipcc = shutil.which("hipcc")
    rocminfo = shutil.which("rocminfo")
    rocm_smi = shutil.which("rocm-smi")
    result: dict[str, Any] = {
        "hipcc": hipcc,
        "rocminfo": rocminfo,
        "rocm_smi": rocm_smi,
    }
    if hipcc is not None:
        output = _run([hipcc, "--version"])
        result["hipcc_version"] = _first_line(output["output"])
    if rocm_smi is not None:
        output = _run([rocm_smi, "--showproductname", "--showmeminfo", "vram"])
        result["device_summary"] = output
    return result


def _distribution_version(names: Sequence[str]) -> dict[str, str] | None:
    """Return the first installed distribution name and version."""
    for name in names:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        return {"distribution": name, "version": version}
    return None


def _isolated_python_json(code: str) -> dict[str, Any]:
    """Run an isolated import probe and decode its JSON output."""
    result = _run(
        [sys.executable, "-I", "-c", code], timeout=30.0, cwd=Path(sys.prefix)
    )
    if result["status"] != "ok":
        return {"status": result["status"], "error": result["output"]}
    try:
        decoded = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"status": "invalid-json", "error": result["output"]}
    if not isinstance(decoded, dict):
        return {"status": "invalid-json", "error": result["output"]}
    return {"status": "ok", **decoded}


def _probe_torch_runtime() -> dict[str, Any]:
    """Return PyTorch accelerator visibility in an isolated interpreter."""
    code = r"""
import json
import torch

devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
print(
    json.dumps(
        {
            "version": torch.__version__,
            "file": torch.__file__,
            "cuda_version": torch.version.cuda,
            "hip_version": torch.version.hip,
            "accelerator_available": torch.cuda.is_available(),
            "devices": devices,
        }
    )
)
"""
    return _isolated_python_json(code)


def _probe_deepmd_runtime() -> dict[str, Any]:
    """Return installed DeePMD-kit location and compiled build variant."""
    code = r"""
import json
import deepmd
from deepmd.env import GLOBAL_CONFIG

result = {
    "version": getattr(deepmd, "__version__", "unknown"),
    "file": getattr(deepmd, "__file__", None),
    "build_variant": GLOBAL_CONFIG.get("dp_variant"),
    "enable_pytorch": GLOBAL_CONFIG.get("enable_pytorch"),
    "enable_tensorflow": GLOBAL_CONFIG.get("enable_tensorflow"),
}
if result["enable_pytorch"]:
    from deepmd.pt.cxx_op import ENABLE_CUSTOMIZED_OP
    result["pytorch_custom_op"] = bool(ENABLE_CUSTOMIZED_OP)
print(json.dumps(result))
"""
    return _isolated_python_json(code)


def _probe_packages() -> dict[str, Any]:
    """Return installed backend distributions and selected runtime details."""
    packages: dict[str, Any] = {
        key: _distribution_version(names)
        for key, names in PACKAGE_DISTRIBUTIONS.items()
    }
    if packages["pytorch"] is not None:
        packages["pytorch"]["runtime"] = _probe_torch_runtime()
    if packages["deepmd"] is not None:
        packages["deepmd"]["runtime"] = _probe_deepmd_runtime()
    return packages


def build_report() -> dict[str, Any]:
    """Build the complete read-only probe report.

    Returns
    -------
    dict
        Machine facts used by the installation planner.
    """
    return {
        "system": _probe_system(),
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT_VARIABLES},
        "python": _probe_python(),
        "tools": _probe_tools(),
        "nvidia": _probe_nvidia(),
        "rocm": _probe_rocm(),
        "packages": _probe_packages(),
    }


def _print_human(report: dict[str, Any]) -> None:
    """Print a readable report without discarding structured values."""
    for section, value in report.items():
        print(f"== {section} ==")
        print(json.dumps(value, indent=2, sort_keys=True))
        print()


def main(argv: list[str] | None = None) -> int:
    """Probe the machine and print JSON or human-readable output.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    int
        Always zero; unavailable optional tools are represented in the report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Emit one machine-readable JSON object."
    )
    args = parser.parse_args(argv)
    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
