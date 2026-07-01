# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import platform
from types import (
    SimpleNamespace,
)

import pytest

if platform.system() not in {"Linux", "Darwin"}:
    pytest.skip("deepmd.lmp supports Linux and Darwin", allow_module_level=True)

from deepmd import (
    lmp,
)


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_tensorflow_library_paths_skip_missing_tensorflow(monkeypatch):
    def fake_import_module(module_name):
        if module_name == "deepmd.tf.env":
            raise _missing_module("tensorflow")
        raise AssertionError(module_name)

    monkeypatch.setattr(lmp, "import_module", fake_import_module)

    assert lmp._get_tensorflow_library_paths() == ([], [])


def test_tensorflow_library_paths_include_libpython_for_old_tensorflow(monkeypatch):
    tf_env = SimpleNamespace(
        TF_VERSION="2.11.0",
        tf=SimpleNamespace(
            sysconfig=SimpleNamespace(get_lib=lambda: "/opt/tensorflow")
        ),
    )
    find_libpython = SimpleNamespace(find_libpython=lambda: "/opt/libpython.so")

    def fake_import_module(module_name):
        if module_name == "deepmd.tf.env":
            return tf_env
        if module_name == "find_libpython":
            return find_libpython
        raise AssertionError(module_name)

    monkeypatch.setattr(lmp, "import_module", fake_import_module)

    assert lmp._get_tensorflow_library_paths() == (
        ["/opt/tensorflow", "/opt/tensorflow/python"],
        ["/opt/libpython.so"],
    )


def test_pytorch_library_paths_skip_missing_torch(monkeypatch):
    def fake_import_module(module_name):
        if module_name == "torch":
            raise _missing_module("torch")
        raise AssertionError(module_name)

    monkeypatch.setattr(lmp, "import_module", fake_import_module)

    assert lmp._get_pytorch_library_paths() == []


def test_configure_lammps_environment_keeps_op_dir_without_backends(monkeypatch):
    monkeypatch.delenv(lmp.lib_env, raising=False)
    monkeypatch.delenv(lmp.preload_env, raising=False)
    monkeypatch.setattr(lmp, "_get_tensorflow_library_paths", lambda: ([], []))
    monkeypatch.setattr(lmp, "_get_pytorch_library_paths", lambda: [])
    monkeypatch.setattr(lmp, "get_library_path", lambda module, filename: [])

    lmp._configure_lammps_environment()

    assert os.environ[lmp.lib_env] == lmp.op_dir
    assert os.environ[lmp.preload_env] == ""


def test_configure_lammps_environment_adds_installed_backend_paths(monkeypatch):
    monkeypatch.setenv(lmp.lib_env, "/existing/lib")
    monkeypatch.setenv(lmp.preload_env, "/existing/preload")
    monkeypatch.setattr(
        lmp,
        "_get_tensorflow_library_paths",
        lambda: (["/tensorflow", "/tensorflow/python"], ["/libpython.so"]),
    )
    monkeypatch.setattr(lmp, "_get_pytorch_library_paths", lambda: ["/torch/lib"])
    monkeypatch.setattr(lmp, "get_library_path", lambda module, filename: [])

    lmp._configure_lammps_environment()

    assert os.environ[lmp.lib_env] == ":".join(
        [
            "/existing/lib",
            "/tensorflow",
            "/tensorflow/python",
            "/torch/lib",
            lmp.op_dir,
        ]
    )
    assert os.environ[lmp.preload_env] == "/existing/preload:/libpython.so"
