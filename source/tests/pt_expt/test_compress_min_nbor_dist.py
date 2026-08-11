# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for reading the stored minimal neighbor distance in pt_expt compress."""

from typing import (
    Any,
)

import pytest

from deepmd.main import (
    parse_args,
)
from deepmd.pt_expt.entrypoints.compress import (
    _read_saved_min_nbor_dist,
)


class _FakeModel:
    """Stand-in exposing only the getter that the reader calls."""

    def __init__(self, min_nbor_dist: float | None) -> None:
        self._min_nbor_dist = min_nbor_dist

    def get_min_nbor_dist(self) -> float | None:
        return self._min_nbor_dist


@pytest.mark.parametrize(
    ("model_min_nbor_dist", "model_dict", "expected"),
    [
        # the model buffer wins over both metadata locations
        (
            1.5,
            {"min_nbor_dist": 9.0, "@variables": {"min_nbor_dist": 8.0}},
            (1.5, "the model"),
        ),
        # the top-level key, written by pt_expt's own compress
        (
            None,
            {"min_nbor_dist": 2.5, "@variables": {"min_nbor_dist": 8.0}},
            (2.5, "the model file"),
        ),
        # "@variables", the cross-backend location that dp convert-backend writes
        (
            None,
            {"@variables": {"min_nbor_dist": 3.5}},
            (3.5, "the model file (@variables)"),
        ),
        # nothing stored anywhere
        (None, {}, (None, "")),
        (None, {"@variables": {}}, (None, "")),
        (None, {"@variables": None}, (None, "")),
    ],
)
def test_read_saved_min_nbor_dist(
    model_min_nbor_dist: float | None,
    model_dict: dict[str, Any],
    expected: tuple[float | None, str],
) -> None:
    """Every storage location is honored, in order of precedence."""
    assert _read_saved_min_nbor_dist(_FakeModel(model_min_nbor_dist), model_dict) == (
        expected
    )


def test_recompute_min_nbor_dist_flag_defaults_to_false() -> None:
    """The compress parser exposes the flag and leaves it off by default."""
    args = ["--pt-expt", "compress", "-i", "in.pt2", "-o", "out.pt2"]
    assert parse_args(args).recompute_min_nbor_dist is False
    assert parse_args([*args, "--recompute-min-nbor-dist"]).recompute_min_nbor_dist


def test_auto_batch_size_follows_the_selected_device() -> None:
    """Growth is allowed only when pt_expt actually runs on a GPU.

    ``DEVICE`` is CPU whenever ``DEVICE=cpu`` is set, even on a CUDA host,
    where an expanding batch would risk an unrecoverable host OOM.
    """
    from deepmd.pt_expt.utils.auto_batch_size import (
        AutoBatchSize,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )

    assert AutoBatchSize().is_gpu_available() == (DEVICE.type == "cuda")


def test_graph_lower_output_keeps_min_nbor_dist(monkeypatch: Any) -> None:
    """The graph-lower branch carries the value into the exported archive.

    ``model.serialize()`` does not include the runtime buffer, so without this
    the compressed artifact loses a value that was just recovered.
    """
    import deepmd.pt_expt.entrypoints.compress as compress_mod
    import deepmd.pt_expt.model.graph_lower as graph_lower_mod
    import deepmd.pt_expt.model.model as model_mod
    import deepmd.pt_expt.utils.tabulate_ops as tabulate_ops_mod

    class _Model(_FakeModel):
        def enable_compression(self, *args: Any, **kwargs: Any) -> None:
            pass

        def serialize(self) -> dict:
            return {"type": "compressed"}

    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        compress_mod,
        "serialize_from_file",
        lambda _: {"model": {}, "@variables": {"min_nbor_dist": 1.25}},
    )
    monkeypatch.setattr(
        model_mod.BaseModel, "deserialize", staticmethod(lambda _: _Model(None))
    )
    monkeypatch.setattr(tabulate_ops_mod, "ensure_fake_registered", lambda: None)
    monkeypatch.setattr(graph_lower_mod, "model_uses_graph_lower", lambda _: True)
    monkeypatch.setattr(
        compress_mod,
        "deserialize_to_file",
        lambda output, data, **kwargs: captured.update(data=data),
    )

    compress_mod.enable_compression(input_file="in.pt2", output="out.pt2")

    assert captured["data"]["min_nbor_dist"] == 1.25
