# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for spin inputs in the dpmodel DeepEval backend."""

from pathlib import (
    Path,
)

import numpy as np
import pytest

from deepmd.infer import (
    DeepEval,
)


MODEL_FILE = Path(__file__).with_name("deeppot_dpa_spin.yaml")
ATOM_TYPES = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
COORD = np.array(
    [
        12.83,
        2.56,
        2.18,
        12.09,
        2.87,
        2.74,
        0.25,
        3.32,
        1.68,
        3.36,
        3.00,
        1.81,
        3.51,
        2.51,
        2.60,
        4.27,
        3.22,
        1.56,
    ],
    dtype=np.float64,
)
SPIN = np.array(
    [
        0.13,
        0.02,
        0.03,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.14,
        0.10,
        0.12,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
BOX = np.diag([13.0, 13.0, 13.0]).reshape(-1)


def test_spin_is_forwarded_and_sliced_by_auto_batch() -> None:
    """A flattened multi-frame spin input must follow coordinate batching."""
    coords = np.concatenate([COORD, COORD])
    boxes = np.concatenate([BOX, BOX])
    spins = np.concatenate([SPIN, 2.0 * SPIN])

    # Six atoms per batch forces the two frames through separate model calls.
    batched_eval = DeepEval(MODEL_FILE, auto_batch_size=len(ATOM_TYPES))
    actual = batched_eval.eval(coords, boxes, ATOM_TYPES, spin=spins)

    unbatched_eval = DeepEval(MODEL_FILE, auto_batch_size=False)
    expected_by_frame = [
        unbatched_eval.eval(COORD, BOX, ATOM_TYPES, spin=frame_spin)
        for frame_spin in (SPIN, 2.0 * SPIN)
    ]
    expected = tuple(
        np.concatenate([frame_result[index] for frame_result in expected_by_frame])
        for index in range(len(actual))
    )

    assert len(actual) == 5  # energy, force, virial, magnetic force, magnetic mask
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, equal_nan=True)
    # Distinct spin vectors must reach the model instead of being ignored.
    assert actual[0][0, 0] != pytest.approx(actual[0][1, 0])


def test_spin_model_requires_spin_input() -> None:
    """Report the missing model input at the evaluator boundary."""
    evaluator = DeepEval(MODEL_FILE, auto_batch_size=False)

    with pytest.raises(ValueError, match="spin must be provided"):
        evaluator.eval(COORD, BOX, ATOM_TYPES)
