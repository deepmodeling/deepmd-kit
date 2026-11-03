# SPDX-License-Identifier: LGPL-3.0-or-later
"""Benchmark comparisons must use the same checkpoint and input frames."""

import pytest

from .compare_so2_scalar_capture import validate_identity


@pytest.mark.parametrize("key", ["model_sha256", "input_sha256"])
@pytest.mark.parametrize("missing", [False, True])
def test_reject_mismatched_identity(key, missing):
    """Reject a changed or absent identity instead of reporting accuracy."""
    left = {"model_sha256": "model", "input_sha256": "input"}
    right = dict(left)
    if missing:
        right.pop(key)
    else:
        right[key] = "different"
    with pytest.raises(ValueError, match=key):
        validate_identity(left, right, left, right)


def test_matching_identity():
    """An identical checkpoint and input pair remains comparable."""
    identity = {"model_sha256": "model", "input_sha256": "input"}
    validate_identity(identity, identity, identity, identity)
