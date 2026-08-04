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
