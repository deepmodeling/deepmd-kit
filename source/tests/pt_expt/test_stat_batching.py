# SPDX-License-Identifier: LGPL-3.0-or-later
from unittest.mock import (
    patch,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils.stat import (
    batch_stat_forward,
)
from deepmd.utils.errors import (
    OutOfMemoryError,
)

from .model.test_get_model_dpa4 import (
    _make_raw_model_config,
    _normalize_model,
)
from .test_change_bias import (
    _make_config,
)


@pytest.mark.parametrize("oom", [False, True])
def test_batch_stat_forward(monkeypatch, oom):
    """Bound device work, retry OOM, and slice every per-frame input together."""
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "6")
    frames, atoms = 5, 3
    coord = np.arange(frames * atoms * 3).reshape(frames, atoms, 3)
    atype = np.zeros((frames, atoms), dtype=np.int64)
    box = np.arange(frames * 9).reshape(frames, 9)
    inputs = {
        "fparam": np.arange(frames * 2).reshape(frames, 2),
        "aparam": np.arange(frames * atoms).reshape(frames, atoms, 1),
        "charge_spin": np.arange(frames * 2).reshape(frames, 2),
        "spin": coord + 1,
    }
    attempts = []
    successful = []

    def forward(c, t, b, **kwargs):
        assert not torch.is_grad_enabled()
        attempts.append(len(c))
        if oom and len(c) > 1:
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        ids = c[:, 0, 0] // (atoms * 3)
        np.testing.assert_array_equal(t, atype[ids])
        np.testing.assert_array_equal(b, box[ids])
        for key, value in kwargs.items():
            np.testing.assert_array_equal(value, inputs[key][ids])
        successful.extend(ids)
        return {"energy": c.sum(axis=-1, keepdims=True)}

    with torch.enable_grad():
        result = batch_stat_forward(forward)(coord, atype, box, **inputs)
        assert torch.is_grad_enabled()
    np.testing.assert_array_equal(result["energy"], coord.sum(axis=-1, keepdims=True))
    assert successful == list(range(frames))
    assert max(attempts) <= 2
    if oom:
        assert attempts[0] == 2
        assert 1 in attempts


def test_single_frame_oom(monkeypatch):
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "1")

    def forward(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")

    with pytest.raises(OutOfMemoryError, match="batch size is 1"):
        batch_stat_forward(forward)(np.zeros((1, 3, 3)), np.zeros((1, 3)), None)


@pytest.mark.parametrize("graph", [False, True])
def test_atomic_statistics_wrapper(monkeypatch, graph):
    """The actual atomic-model statistics wrapper applies batching and no_grad."""
    monkeypatch.setenv("DP_INFER_BATCH_SIZE", "2")
    params = (
        _normalize_model(_make_raw_model_config())
        if graph
        else _make_config("unused")["model"]
    )
    model = get_model(params)
    atomic = model.atomic_model
    coord = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]] * 3)
    atype = np.array([[0, 1]] * 3, dtype=np.int64)
    method = "forward_common_atomic_graph" if graph else "forward_common_atomic"
    original = getattr(atomic, method)
    calls = []

    def checked(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert args[1].shape[0] == 2 if graph else args[0].shape[0] == 1
        calls.append(1)
        return original(*args, **kwargs)

    with patch.object(atomic, method, side_effect=checked):
        result = atomic._get_forward_wrapper_func()(coord, atype, None)
    assert len(calls) == 3
    assert result["energy"].shape[:2] == (3, 2)
    with torch.no_grad():
        expected = BaseAtomicModel._get_forward_wrapper_func(atomic)(coord, atype, None)
    np.testing.assert_allclose(result["energy"], expected["energy"], rtol=0, atol=1e-12)
