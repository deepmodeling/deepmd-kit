# SPDX-License-Identifier: LGPL-3.0-or-later
"""Build-option regressions that do not require either neural-network backend."""

import pytest

from backend import (
    read_env,
)


@pytest.mark.parametrize("interface", ["python", "ipi", "lammps"])
@pytest.mark.parametrize("tensorflow", ["0", "1", None])
def test_cpp_interface_respects_tensorflow_selection(
    monkeypatch, interface, tensorflow
):
    for key in (
        "DP_VARIANT",
        "DP_ENABLE_IPI",
        "DP_LAMMPS_VERSION",
        "DP_ENABLE_TENSORFLOW",
        "DP_ENABLE_PYTORCH",
        "DP_BUILD_TESTING",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DP_ENABLE_PYTORCH", "1")
    if tensorflow is not None:
        monkeypatch.setenv("DP_ENABLE_TENSORFLOW", tensorflow)
    if interface == "ipi":
        monkeypatch.setenv("DP_ENABLE_IPI", "1")
    elif interface == "lammps":
        monkeypatch.setenv("DP_LAMMPS_VERSION", "test-version")
    tf_enabled = tensorflow != "0"

    def find_tensorflow():
        assert tf_enabled, "Disabled TensorFlow must not be discovered"
        return "/mock/tensorflow", []

    monkeypatch.setattr(read_env, "find_tensorflow", find_tensorflow)
    monkeypatch.setattr(read_env, "get_tf_version", lambda path: "2.20.0")
    monkeypatch.setattr(read_env, "find_pytorch", lambda: ("/mock/torch", []))
    monkeypatch.setattr(read_env, "get_pt_version", lambda path: "2.11.0")
    read_env.get_argument_from_env.cache_clear()
    try:
        _, args, _, scripts, _, _ = read_env.get_argument_from_env()
    finally:
        read_env.get_argument_from_env.cache_clear()
    assert (
        "-DENABLE_TENSORFLOW=ON" if tf_enabled else "-DENABLE_TENSORFLOW=OFF"
    ) in args
    if interface != "python":
        selected = "TRUE" if tf_enabled else "FALSE"
        assert f"-DUSE_TF_PYTHON_LIBS:BOOL={selected}" in args
        assert "-DBUILD_CPP_IF:BOOL=TRUE" in args
    else:
        assert "-DBUILD_CPP_IF:BOOL=FALSE" in args
        assert "-DUSE_TF_PYTHON_LIBS:BOOL=TRUE" not in args
    assert ("dp_ipi" in scripts) == (interface == "ipi")
