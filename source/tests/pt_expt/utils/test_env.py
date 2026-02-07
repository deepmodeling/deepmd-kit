# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import logging

import torch

import deepmd.env as common_env


def test_env_threads_guard_handles_runtimeerror(monkeypatch) -> None:
    def raise_err(*_args, **_kwargs) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(common_env, "set_default_nthreads", lambda: None)
    monkeypatch.setattr(common_env, "get_default_nthreads", lambda: (1, 1))
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)
    monkeypatch.setattr(torch, "set_num_interop_threads", raise_err)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch, "set_num_threads", raise_err)

    messages: list[str] = []
    original_warning = logging.Logger.warning

    def capture_warning(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def]
        messages.append(str(msg))
        return original_warning(self, msg, *args, **kwargs)

    monkeypatch.setattr(logging.Logger, "warning", capture_warning)
    import deepmd.pt_expt.utils.env as env

    importlib.reload(env)

    assert any("Could not set torch interop threads" in msg for msg in messages)
    assert any("Could not set torch intra threads" in msg for msg in messages)
