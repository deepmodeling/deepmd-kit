# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark comparisons must use the same checkpoint and input frames."""

import numpy as np
import pytest

from .compare_so2_scalar_capture import (
    compare,
    validate_identity,
)


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
    arrays = {
        f"{repeat}_{frame}_{field}": np.array([1.0 + repeat])
        for repeat in range(3)
        for frame in range(3)
        for field in ("energy", "forces", "virial", "stress")
    }
    result = compare(arrays, arrays)
    for field in result.values():
        assert not field["passed"]
        assert field["max_abs"] == 2.0
