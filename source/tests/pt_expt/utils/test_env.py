# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import logging

import torch

import deepmd.env as common_env


def test_lmdb_num_workers_override(monkeypatch) -> None:
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "12")
    assert common_env.get_lmdb_num_workers() == 12


def test_lmdb_num_workers_rejects_invalid_override(monkeypatch) -> None:
    monkeypatch.setenv("DP_LMDB_NUM_WORKERS", "-1")
    try:
        common_env.get_lmdb_num_workers()
    except ValueError as error:
        assert "must be non-negative" in str(error)
    else:
        raise AssertionError("negative DP_LMDB_NUM_WORKERS was accepted")


def test_lmdb_num_workers_partitions_node_budget(monkeypatch) -> None:
    monkeypatch.delenv("DP_LMDB_NUM_WORKERS", raising=False)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    monkeypatch.setattr(common_env.os, "sched_getaffinity", lambda _pid: set(range(80)))
    assert common_env.get_lmdb_num_workers() == 16


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
