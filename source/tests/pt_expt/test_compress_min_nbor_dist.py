# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for reading the stored minimal neighbor distance in pt_expt compress."""

import logging
from typing import (
    Any,
)

import pytest

from deepmd.main import (
    parse_args,
)
from deepmd.pt_expt.entrypoints import compress as compress_mod


class _FakeModel:
    """Stand-in exposing only what the compression entry point calls."""

    def __init__(
        self,
        min_nbor_dist: float | None,
        *,
        needs_min_nbor_dist: bool = True,
    ) -> None:
        self._min_nbor_dist = min_nbor_dist
        self._needs_min_nbor_dist = needs_min_nbor_dist

    def get_min_nbor_dist(self) -> float | None:
        return self._min_nbor_dist

    def compression_needs_min_nbor_dist(self) -> bool:
        return self._needs_min_nbor_dist


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
    got = compress_mod._read_saved_min_nbor_dist(
        _FakeModel(model_min_nbor_dist), model_dict
    )
    assert got == expected


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


def test_spin_model_min_nbor_dist_reaches_the_backbone() -> None:
    """Assigning on the wrapper must reach the model that gets compressed.

    ``SpinModel.__getattr__`` only delegates reads, while both
    ``get_min_nbor_dist`` and ``enable_compression`` run on the backbone.
    """
    from deepmd.pt_expt.model.spin_model import (
        SpinModel,
    )

    class _Backbone:
        min_nbor_dist: float | None = None

    class _Wrapper:
        # the property under test, on a stand-in that needs no torch module
        min_nbor_dist = SpinModel.min_nbor_dist

        def __init__(self) -> None:
            self.backbone_model = _Backbone()

    wrapper = _Wrapper()
    wrapper.min_nbor_dist = 1.25
    assert wrapper.backbone_model.min_nbor_dist == 1.25
    assert wrapper.min_nbor_dist == 1.25


def _run_compression(
    monkeypatch: Any,
    *,
    saved: float | None,
    needs: bool,
    flag: bool,
    training_script: str | None = "train.json",
) -> tuple[dict, list, Any]:
    """Compress a stand-in model and report what reached the archive.

    Returns
    -------
    dict
        The data handed to ``deserialize_to_file``.
    list
        One entry per training-data scan, so an empty list proves none ran.
    Any
        The model, to inspect what the run left on it.
    """
    import deepmd.pt_expt.model.graph_lower as graph_lower_mod
    import deepmd.pt_expt.model.model as model_mod
    import deepmd.pt_expt.utils.tabulate_ops as tabulate_ops_mod

    class _Model(_FakeModel):
        def enable_compression(self, *args: Any, **kwargs: Any) -> None:
            pass

        def serialize(self) -> dict:
            return {"type": "compressed"}

    model = _Model(None, needs_min_nbor_dist=needs)
    model_dict: dict[str, Any] = {"model": {}}
    if saved is not None:
        model_dict["@variables"] = {"min_nbor_dist": saved}

    captured: dict[str, Any] = {}
    scans: list[str] = []

    monkeypatch.setattr(compress_mod, "serialize_from_file", lambda _: model_dict)
    monkeypatch.setattr(
        model_mod.BaseModel, "deserialize", staticmethod(lambda _: model)
    )
    monkeypatch.setattr(tabulate_ops_mod, "ensure_fake_registered", lambda: None)
    monkeypatch.setattr(graph_lower_mod, "model_uses_graph_lower", lambda _: True)
    monkeypatch.setattr(
        compress_mod,
        "deserialize_to_file",
        lambda output, data, **kwargs: captured.update(data=data),
    )
    monkeypatch.setattr(
        compress_mod,
        "_measure_min_nbor_dist",
        lambda script: (scans.append(script), 9.99)[1],
    )

    compress_mod.enable_compression(
        input_file="in.pt2",
        output="out.pt2",
        training_script=training_script,
        recompute_min_nbor_dist=flag,
    )
    return captured["data"], scans, model


@pytest.mark.parametrize("flag", [False, True])
def test_unused_statistic_keeps_a_saved_distance(monkeypatch: Any, flag: bool) -> None:
    """A model that ignores the statistic must not lose a measured one.

    The recompute flag is inapplicable here, so it may not replace a real
    distance with the placeholder the compression ABI is given.
    """
    data, scans, model = _run_compression(
        monkeypatch, saved=0.75, needs=False, flag=flag
    )

    assert data["min_nbor_dist"] == 0.75
    assert scans == []
    assert model.min_nbor_dist == 0.75


@pytest.mark.parametrize("flag", [False, True])
def test_unused_statistic_leaves_an_absent_distance_absent(
    monkeypatch: Any, flag: bool
) -> None:
    """Never measured means absent, not zero.

    ``enable_compression`` still needs a number, but that placeholder stays
    local to the call instead of being recorded as a measurement.
    """
    data, scans, model = _run_compression(
        monkeypatch, saved=None, needs=False, flag=flag
    )

    assert "min_nbor_dist" not in data
    assert scans == []
    assert model.min_nbor_dist is None


def test_unused_statistic_warns_the_flag_is_inapplicable(
    monkeypatch: Any, caplog: Any
) -> None:
    """The flag is ignored loudly, not silently."""
    with caplog.at_level(logging.WARNING):
        _run_compression(monkeypatch, saved=0.75, needs=False, flag=True)

    assert "--recompute-min-nbor-dist does not apply" in caplog.text


@pytest.mark.parametrize(
    ("saved", "flag", "expected", "scanned"),
    [
        # a saved value is reused, so no pass over the training data
        (1.25, False, 1.25, False),
        # an explicit request measures again and wins over the saved value
        (1.25, True, 9.99, True),
        # nothing saved leaves measuring as the only option
        (None, False, 9.99, True),
    ],
)
def test_consumed_statistic_paths(
    monkeypatch: Any,
    saved: float | None,
    flag: bool,
    expected: float,
    scanned: bool,
) -> None:
    """A model whose tabulation consumes the statistic is unaffected."""
    data, scans, _ = _run_compression(monkeypatch, saved=saved, needs=True, flag=flag)

    assert data["min_nbor_dist"] == expected
    assert bool(scans) is scanned


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        (True, "Recomputing the minimal neighbor distance requires"),
        (False, "does not have a minimum neighbor distance"),
    ],
)
def test_measuring_requires_a_training_script(
    monkeypatch: Any, flag: bool, message: str
) -> None:
    """Both routes into measuring keep their training-script validation."""
    with pytest.raises(ValueError, match=message):
        _run_compression(
            monkeypatch,
            saved=None if not flag else 1.25,
            needs=True,
            flag=flag,
            training_script=None,
        )
