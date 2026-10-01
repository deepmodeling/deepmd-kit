# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for grouped-mode auto-detection (dpa_adapt.trainer._systems_are_grouped).

Detection must scan every ``set.*`` directory of every given system list, not
just the first set of the first system, and must reject any mix of grouped
and ungrouped sets (including a partially-marked set, or grouped
train_systems paired with ungrouped valid_systems) with a clear error instead
of silently under- or over-detecting grouped mode.
"""

from __future__ import (
    annotations,
)

import numpy as np
import pytest

from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.trainer import (
    _systems_are_grouped,
)

_MARKERS = ("group_id", "weight", "pool_mask")


def _make_system(tmp_path, name: str, *, markers: tuple[str, ...] | None) -> str:
    """Create a system dir with one set.000, optionally carrying markers.

    ``markers=None`` writes no marker files; an explicit tuple writes only
    those (so a strict subset of _MARKERS produces a "partial" set).
    """
    sysdir = tmp_path / name
    set_dir = sysdir / "set.000"
    set_dir.mkdir(parents=True)
    for marker in markers or ():
        np.save(set_dir / f"{marker}.npy", np.zeros(1))
    return str(sysdir)


def test_all_ungrouped_returns_false(tmp_path):
    systems = [
        _make_system(tmp_path, "a", markers=None),
        _make_system(tmp_path, "b", markers=None),
    ]
    assert _systems_are_grouped(("train_systems", systems)) is False


def test_all_grouped_returns_true(tmp_path):
    systems = [
        _make_system(tmp_path, "a", markers=_MARKERS),
        _make_system(tmp_path, "b", markers=_MARKERS),
    ]
    assert _systems_are_grouped(("train_systems", systems)) is True


def test_first_system_ungrouped_later_grouped_raises(tmp_path):
    """The bug this fix targets: checking only the first system's first set
    would leave grouped mode disabled here, silently dropping the second
    system's group labels instead of erroring.
    """
    systems = [
        _make_system(tmp_path, "a", markers=None),
        _make_system(tmp_path, "b", markers=_MARKERS),
    ]
    with pytest.raises(DPADataError, match="Inconsistent grouped markers"):
        _systems_are_grouped(("train_systems", systems))


def test_first_system_grouped_later_ungrouped_raises(tmp_path):
    """The mirror bug: checking only the first system's first set would
    enable grouped mode here and fail (or train inconsistently) once the
    second, marker-less system was reached.
    """
    systems = [
        _make_system(tmp_path, "a", markers=_MARKERS),
        _make_system(tmp_path, "b", markers=None),
    ]
    with pytest.raises(DPADataError, match="Inconsistent grouped markers"):
        _systems_are_grouped(("train_systems", systems))


def test_partial_markers_raises(tmp_path):
    systems = [_make_system(tmp_path, "a", markers=("group_id",))]
    with pytest.raises(DPADataError, match="only some of"):
        _systems_are_grouped(("train_systems", systems))


def test_grouped_train_ungrouped_valid_raises(tmp_path):
    train = [_make_system(tmp_path, "train", markers=_MARKERS)]
    valid = [_make_system(tmp_path, "valid", markers=None)]
    with pytest.raises(DPADataError, match="Inconsistent grouped markers"):
        _systems_are_grouped(("train_systems", train), ("valid_systems", valid))


def test_grouped_train_and_valid_returns_true(tmp_path):
    train = [_make_system(tmp_path, "train", markers=_MARKERS)]
    valid = [_make_system(tmp_path, "valid", markers=_MARKERS)]
    assert (
        _systems_are_grouped(("train_systems", train), ("valid_systems", valid)) is True
    )


def test_no_systems_returns_false():
    assert _systems_are_grouped(("train_systems", [])) is False
