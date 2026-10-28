# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configure LibTorch capability detection without requiring physical GPUs."""

import shutil
import subprocess
from pathlib import (
    Path,
)

import pytest


@pytest.mark.parametrize("toolkit", ["CPU", "CUDA", "ROCM"])
@pytest.mark.parametrize("torch_runtime", ["CPU", "CUDA", "HIP"])
def test_torch_gpu_capability(tmp_path: Path, toolkit: str, torch_runtime: str) -> None:
    """Only a GPU-enabled LibTorch may enable preselection, including on HIP."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is not installed")
    module = Path(__file__).resolve().parents[2] / "cmake/torch_gpu_support.cmake"
    setup = {
        "CPU": "",
        "CUDA": "set(TORCH_CUDA_LIBRARIES torch_cuda)",
        "HIP": "add_library(torch_hip INTERFACE IMPORTED)",
    }[torch_runtime]
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(TorchCapability NONE)\n"
        f"set(USE_{toolkit}_TOOLKIT TRUE)\n"
        f"{setup}\n"
        f'include("{module.as_posix()}")\n'
        f'if(NOT "${{DEEPMD_TORCH_HAS_GPU}}" STREQUAL "{str(torch_runtime != "CPU").upper()}")\n'
        '  message(FATAL_ERROR "Incorrect generic GPU capability")\n'
        "endif()\n"
        f'if(NOT "${{DEEPMD_TORCH_HAS_CUDA}}" STREQUAL "{str(torch_runtime == "CUDA").upper()}")\n'
        '  message(FATAL_ERROR "Incorrect CUDA-only capability")\n'
        "endif()\n"
    )
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        check=True,
        capture_output=True,
        text=True,
    )
