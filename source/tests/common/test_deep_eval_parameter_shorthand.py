# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for backend-level DeepEval parameter normalization."""

import numpy as np
import pytest

from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.serialization import (
    save_dp_model,
)
from deepmd.infer.deep_eval import (
    DeepEval,
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
    assert actual.flags.writeable
    assert actual.flags.c_contiguous


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
    assert actual.flags.writeable
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("aparam", [APARAM_PER_ATOM, APARAM_ALL_ATOMS])
def test_shared_parameters_can_be_broadcast_views(aparam: np.ndarray) -> None:
    """Opted-in shared inputs retain their storage across a long trajectory."""
    nframes = 10000
    fparam_view, aparam_view = _standardize_fparam_aparam(
        FPARAM,
        aparam,
        nframes,
        NATOMS,
        DIM_FPARAM,
        DIM_APARAM,
        copy_shared=False,
    )
    assert fparam_view.shape == (nframes, DIM_FPARAM)
    assert aparam_view.shape == (nframes, NATOMS, DIM_APARAM)
    assert np.shares_memory(fparam_view, FPARAM)
    assert np.shares_memory(aparam_view, aparam)
    assert fparam_view.strides[0] == aparam_view.strides[0] == 0
    assert not fparam_view.flags.writeable
    assert not aparam_view.flags.writeable
    expected_aparam = aparam if aparam.ndim == 2 else np.tile(aparam, (NATOMS, 1))
    for frame in (0, nframes - 1):
        np.testing.assert_array_equal(fparam_view[frame], FPARAM)
        np.testing.assert_array_equal(aparam_view[frame], expected_aparam)


@pytest.mark.parametrize("aparam", [APARAM_PER_ATOM, APARAM_ALL_ATOMS])
def test_dpmodel_batches_shared_parameter_views(
    tmp_path, monkeypatch: pytest.MonkeyPatch, aparam: np.ndarray
) -> None:
    """The real dpmodel consumes read-only views after slicing their frame axis."""
    model_def = {
        "type_map": ["X"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4],
            "rcut": 3.0,
            "rcut_smth": 2.5,
            "neuron": [2],
            "axis_neuron": 1,
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [2],
            "numb_fparam": DIM_FPARAM,
            "numb_aparam": DIM_APARAM,
            "seed": 1,
        },
    }
    path = str(tmp_path / "params.dp")
    save_dp_model(
        path,
        {
            "model": get_model(model_def).serialize(),
            "model_def_script": model_def,
            "backend": "dpmodel",
        },
    )
    coords = np.tile(np.arange(NATOMS * 3).reshape(1, NATOMS, 3) * 0.1, (NFRAMES, 1, 1))
    cells = np.tile(np.eye(3) * 10.0, (NFRAMES, 1, 1))
    atom_types = np.zeros(NATOMS, dtype=np.int32)
    backend = DeepEval(path, auto_batch_size=NATOMS).deep_eval
    evaluate_batch = backend._eval_model
    batch_sizes = []

    def check_batch(coords, cells, atom_types, fparam, aparam_batch, request_defs):
        batch_sizes.append(coords.shape[0])
        assert np.shares_memory(fparam, FPARAM)
        assert np.shares_memory(aparam_batch, aparam)
        assert not fparam.flags.writeable
        assert not aparam_batch.flags.writeable
        return evaluate_batch(
            coords, cells, atom_types, fparam, aparam_batch, request_defs
        )

    monkeypatch.setattr(backend, "_eval_model", check_batch)
    actual = backend.eval(coords, cells, atom_types, fparam=FPARAM, aparam=aparam)
    expected = DeepEval(path, auto_batch_size=False).deep_eval.eval(
        coords,
        cells,
        atom_types,
        fparam=np.tile(FPARAM, (NFRAMES, 1)),
        aparam=np.tile(
            aparam.reshape(-1, DIM_APARAM),
            (NFRAMES, NATOMS if aparam.ndim == 1 else 1, 1),
        ),
    )
    assert len(batch_sizes) > 1
    assert sum(batch_sizes) == NFRAMES
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_allclose(
            actual[name], expected[name], rtol=1e-10, atol=1e-10, equal_nan=True
        )


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
