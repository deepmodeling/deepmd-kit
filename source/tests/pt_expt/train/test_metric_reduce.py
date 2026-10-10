# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for Torch metric-window reduction helpers."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import pytest

torch = pytest.importorskip("torch")

from deepmd.dpmodel.train.metrics import (
    MetricAccumulator,
)
from deepmd.pt_expt.train import metrics as metrics_mod
from deepmd.pt_expt.train.metrics import (
    _collective_device,
    all_reduce_metric_accumulator,
)


def test_all_reduce_is_noop_without_process_group() -> None:
    accumulator = MetricAccumulator({"task": ("rmse",)})
    accumulator.add("task", {"rmse": 1.5})
    all_reduce_metric_accumulator(accumulator)
    assert accumulator.average("task") == {"rmse": 1.5}


def test_collective_device_follows_nccl_not_host_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NCCL ranks must agree on CUDA even when the local window is host-only."""
    monkeypatch.setattr(metrics_mod.dist, "get_backend", lambda group=None: "nccl")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert _collective_device() == torch.device("cuda", 0)

    monkeypatch.setattr(
        metrics_mod.dist,
        "get_backend",
        lambda group=None: "cuda:nccl,cpu:gloo",
    )
    assert _collective_device() == torch.device("cuda", 0)


def test_collective_device_uses_cpu_for_gloo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics_mod.dist, "get_backend", lambda group=None: "gloo")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _collective_device() == torch.device("cpu")


def test_all_reduce_uses_group_device_for_host_only_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-only flat_state must still all-reduce on the group device."""
    captured: dict[str, Any] = {}

    def fake_all_reduce(tensor: torch.Tensor, op=None, group=None) -> None:
        captured["device"] = tensor.device
        tensor.fill_(0.0)

    monkeypatch.setattr(metrics_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(metrics_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(metrics_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(metrics_mod.dist, "get_backend", lambda group=None: "nccl")
    monkeypatch.setattr(metrics_mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    # Keep tensor construction on CPU in this CPU-only CI image while still
    # asserting that the requested device is the NCCL CUDA device.
    real_tensor = torch.tensor

    def fake_tensor(data, dtype=None, device=None, **kwargs):
        captured["requested_device"] = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        return real_tensor(data, dtype=dtype, device="cpu", **kwargs)

    monkeypatch.setattr(torch, "tensor", fake_tensor)

    accumulator = MetricAccumulator({"task": ("rmse",)})
    # Optional-label NaNs leave sums as host floats (never promoted to CUDA).
    accumulator.add("task", {"rmse": float("nan")})
    assert all(not torch.is_tensor(v) for v in accumulator.flat_state())

    all_reduce_metric_accumulator(accumulator)
    assert captured["requested_device"] == torch.device("cuda", 0)
