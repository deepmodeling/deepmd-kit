# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark comparisons must use the same checkpoint and input frames."""

from types import (
    SimpleNamespace,
)

import numpy as np
import pytest

from .compare_so2_scalar_capture import (
    compare,
    configure_capture_route,
    validate_identity,
)


@pytest.mark.parametrize("route", ["BASE", "CSC5", "TORCH"])
@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("eligible", [False, True])
def test_capture_route_support(route, available, eligible):
    """A pre-feature checkout cannot masquerade as a completed CSC5 run."""
    scalar_path = SimpleNamespace()
    if available:

        def original(conv, x):
            return eligible

        scalar_path._save_competition_scalar = original
    if route == "CSC5" and not available:
        with pytest.raises(ValueError, match="CSC5 is unavailable"):
            configure_capture_route(route, scalar_path)
        return
    configure_capture_route(route, scalar_path)
    if available:
        assert scalar_path._save_competition_scalar(None, None) == (
            eligible and route != "BASE"
        )
        if route != "BASE":
            assert scalar_path._save_competition_scalar is original


@pytest.mark.parametrize("key", ["model_sha256", "input_sha256"])
@pytest.mark.parametrize("missing", [False, True])
def test_reject_mismatched_identity(key, missing):
    """Reject a changed or absent identity instead of reporting accuracy."""
    result = {"input_sha256": "input"}
    manifest = {"model_sha256": "model"}
    right_result, right_manifest = dict(result), dict(manifest)
    right = right_result if key == "input_sha256" else right_manifest
    if missing:
        right.pop(key)
    else:
        right[key] = "different"
    with pytest.raises(ValueError, match=key):
        validate_identity(result, right_result, manifest, right_manifest)


def test_matching_identity():
    """An identical checkpoint and input pair remains comparable."""
    result = {"input_sha256": "input"}
    manifest = {"model_sha256": "model"}
    validate_identity(result, result, manifest, manifest)


def test_compare_checks_cross_repeat_pairs():
    """Equal matching repeats must not hide variability across repeats."""
    scales = {"energy": 1.0, "forces": 10.0, "virial": 100.0, "stress": 1000.0}
    arrays = {
        f"{repeat}_{frame}_{field}": np.array([(1.0 + repeat) * scales[field]])
        for repeat in range(3)
        for frame in range(3)
        for field in scales
    }
    result = compare(arrays, arrays)
    for field, scale in scales.items():
        assert not result[field]["passed"]
        assert result[field]["max_abs"] == 2.0 * scale


@pytest.mark.parametrize("changed_frame", [0, 1, 2])
def test_compare_uses_reference_and_each_frame(changed_frame):
    """Distinct routes must expose a discrepancy confined to any one frame."""
    scales = {"energy": 1.0, "forces": 10.0, "virial": 100.0, "stress": 1000.0}
    candidate = {
        f"{repeat}_{frame}_{field}": np.array([scale])
        for repeat in range(3)
        for frame in range(3)
        for field, scale in scales.items()
    }
    reference = {key: value.copy() for key, value in candidate.items()}
    for field, scale in scales.items():
        reference[f"0_{changed_frame}_{field}"] += 2 * scale
    result = compare(candidate, reference)
    for field, scale in scales.items():
        assert not result[field]["passed"]
        assert result[field]["max_abs"] == 2 * scale
        assert result[field]["worst"]["frame"] == changed_frame
        assert result[field]["worst"]["reference_repeat"] == 0
        assert result[field]["worst"]["index"] == [0]
        assert result[field]["worst"]["candidate"] == scale
        assert result[field]["worst"]["reference"] == 3 * scale
