# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt_expt.utils import (
    serialization,
)


class _DummyModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def test_trace_with_comm_rejects_non_comm_model_before_registration(
    monkeypatch,
) -> None:
    import deepmd.pt_expt.model.model as model_module

    monkeypatch.setattr(
        model_module.BaseModel,
        "deserialize",
        classmethod(lambda cls, data: _DummyModel()),
    )
    monkeypatch.setattr(serialization, "_collect_metadata", lambda model, is_spin: {})
    monkeypatch.setattr(serialization, "_needs_with_comm_artifact", lambda model: False)
    monkeypatch.setattr(
        serialization,
        "_make_sample_inputs",
        lambda model, nframes, has_spin: (
            torch.zeros((1, 4, 3), dtype=torch.float64),
            torch.zeros((1, 4), dtype=torch.int32),
            torch.zeros((1, 2, 4), dtype=torch.int64),
            torch.zeros((1, 4), dtype=torch.int64),
            None,
            None,
        ),
    )

    with pytest.raises(ValueError, match="nothing to compile"):
        serialization._trace_and_export(
            {"model": {"type": "ener"}},
            with_comm_dict=True,
        )
