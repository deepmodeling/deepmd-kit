# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for backend-level DeepEval parameter normalization."""

import numpy as np
import pytest

from deepmd.infer.deep_eval import (
    _standardize_fparam_aparam,
)

NFRAMES = 3
NATOMS = 4
DIM_FPARAM = 2
DIM_APARAM = 2
FPARAM = np.array([0.25, -0.5], dtype=np.float64)
APARAM_PER_ATOM = np.arange(NATOMS * DIM_APARAM, dtype=np.float64).reshape(
    NATOMS, DIM_APARAM
)
APARAM_ALL_ATOMS = np.array([0.3, -0.2], dtype=np.float64)


@pytest.mark.parametrize(
    ("fparam", "expected"),
    [
        (FPARAM.tolist(), np.tile(FPARAM, (NFRAMES, 1))),
        (
            np.arange(NFRAMES * DIM_FPARAM).reshape(NFRAMES, DIM_FPARAM),
            np.arange(NFRAMES * DIM_FPARAM).reshape(NFRAMES, DIM_FPARAM),
        ),
    ],
    ids=("shared", "per-frame"),
)
def test_standardize_fparam(fparam, expected) -> None:
    """Frame parameters become a canonical frame-major matrix."""
    actual, _ = _standardize_fparam_aparam(
        fparam,
        None,
        NFRAMES,
        NATOMS,
        DIM_FPARAM,
        DIM_APARAM,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("aparam", "expected"),
    [
        (
            APARAM_PER_ATOM,
            np.tile(APARAM_PER_ATOM, (NFRAMES, 1, 1)),
        ),
        (
            APARAM_ALL_ATOMS.tolist(),
            np.tile(APARAM_ALL_ATOMS, (NFRAMES, NATOMS, 1)),
        ),
        (
            np.arange(NFRAMES * NATOMS * DIM_APARAM).reshape(
                NFRAMES, NATOMS, DIM_APARAM
            ),
            np.arange(NFRAMES * NATOMS * DIM_APARAM).reshape(
                NFRAMES, NATOMS, DIM_APARAM
            ),
        ),
    ],
    ids=("shared-per-atom", "shared-all-atoms", "per-frame"),
)
def test_standardize_aparam(aparam, expected) -> None:
    """Atomic shorthand is expanded before a batcher can slice its atom axis."""
    _, actual = _standardize_fparam_aparam(
        None,
        aparam,
        NFRAMES,
        NATOMS,
        DIM_FPARAM,
        DIM_APARAM,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("fparam", "aparam", "message"),
    [
        (np.zeros(3), None, "wrong size of frame param"),
        (None, np.zeros(3), "wrong size of atomic param"),
    ],
)
def test_invalid_parameter_size_is_rejected(fparam, aparam, message) -> None:
    """Report the documented contract instead of a backend reshape failure."""
    with pytest.raises(RuntimeError, match=message):
        _standardize_fparam_aparam(
            fparam,
            aparam,
            NFRAMES,
            NATOMS,
            DIM_FPARAM,
            DIM_APARAM,
        )
