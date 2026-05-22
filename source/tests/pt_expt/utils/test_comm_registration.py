# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import torch

from deepmd.pt_expt.utils import (
    comm,
)


def test_ensure_comm_registered_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(comm, "_registered", False)
    calls: dict[str, object] = {
        "check": 0,
        "fake_names": [],
        "autograd": 0,
    }

    def _check() -> None:
        calls["check"] = int(calls["check"]) + 1

    def _register_fake(name: str):
        fake_names = calls["fake_names"]
        assert isinstance(fake_names, list)
        fake_names.append(name)

        def _decorator(fn):
            return fn

        return _decorator

    def _register_autograd(*args, **kwargs) -> None:
        calls["autograd"] = int(calls["autograd"]) + 1

    monkeypatch.setattr(comm, "_check_underlying_ops_loaded", _check)
    monkeypatch.setattr(torch.library, "register_fake", _register_fake)
    monkeypatch.setattr(torch.library, "register_autograd", _register_autograd)

    comm.ensure_comm_registered()
    comm.ensure_comm_registered()

    assert calls["check"] == 1
    assert calls["fake_names"] == [
        "deepmd_export::border_op",
        "deepmd_export::border_op_backward",
    ]
    assert calls["autograd"] == 1


def test_ensure_comm_registered_tolerates_duplicate_autograd(monkeypatch) -> None:
    monkeypatch.setattr(comm, "_registered", False)
    monkeypatch.setattr(comm, "_check_underlying_ops_loaded", lambda: None)

    def _register_fake(name: str):
        def _decorator(fn):
            raise RuntimeError("already registered")

        return _decorator

    def _register_autograd(*args, **kwargs) -> None:
        raise RuntimeError("already has an autograd implementation")

    monkeypatch.setattr(torch.library, "register_fake", _register_fake)
    monkeypatch.setattr(torch.library, "register_autograd", _register_autograd)

    comm.ensure_comm_registered()

    assert comm._registered is True
