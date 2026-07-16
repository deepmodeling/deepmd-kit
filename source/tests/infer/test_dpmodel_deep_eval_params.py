# SPDX-License-Identifier: LGPL-3.0-or-later
"""Direct-backend tests for dpmodel DeepEval parameter shorthand."""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.model.model import (
    get_model,
)
from deepmd.dpmodel.utils.serialization import (
    save_dp_model,
)
from deepmd.infer import (
    DeepEval,
)

MODEL_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20],
        "rcut_smth": 0.5,
        "rcut": 6.0,
        "neuron": [3, 6],
        "resnet_dt": False,
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "type": "ener",
        "neuron": [5, 5],
        "resnet_dt": True,
        "precision": "float64",
        "atom_ener": [],
        "seed": 1,
        "numb_fparam": 2,
        "numb_aparam": 2,
    },
}
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
).reshape(len(ATOM_TYPES), 3)
BOX = np.diag([13.0, 13.0, 13.0])
NFRAMES = 2
COORDS = np.tile(COORD, (NFRAMES, 1, 1))
BOXES = np.tile(BOX, (NFRAMES, 1, 1))
FPARAM = np.array([0.25, -0.5], dtype=np.float64)
APARAM_PER_ATOM = (
    np.arange(len(ATOM_TYPES) * 2, dtype=np.float64).reshape(len(ATOM_TYPES), 2) / 10.0
)
APARAM_GLOBAL = np.array([0.3, -0.2], dtype=np.float64)


@pytest.fixture(scope="module")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Serialize a two-dimensional fparam/aparam model for direct inference."""
    model = get_model(copy.deepcopy(MODEL_CONFIG))
    path = tmp_path_factory.mktemp("dpmodel_params") / "model.dp"
    save_dp_model(
        str(path),
        {
            "model": model.serialize(),
            "model_def_script": MODEL_CONFIG,
            "backend": "dpmodel",
        },
    )
    return str(path)


def _backend(model_file: str, auto_batch_size: bool | int):
    """Return the low-level backend without public input normalization."""
    return DeepEval(model_file, auto_batch_size=auto_batch_size).deep_eval


def _assert_outputs_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], equal_nan=True)


@pytest.mark.parametrize("auto_batch_size", [False, len(ATOM_TYPES)])
def test_shared_fparam_is_tiled_before_batching(
    model_file: str, auto_batch_size: bool | int
) -> None:
    """A single frame-parameter vector applies to every input frame."""
    evaluator = _backend(model_file, auto_batch_size)
    full_aparam = np.tile(APARAM_PER_ATOM, (NFRAMES, 1, 1))
    expected = evaluator.eval(
        COORDS,
        BOXES,
        ATOM_TYPES,
        fparam=np.tile(FPARAM, (NFRAMES, 1)),
        aparam=full_aparam,
    )
    actual = evaluator.eval(
        COORDS,
        BOXES,
        ATOM_TYPES,
        fparam=FPARAM.tolist(),
        aparam=full_aparam,
    )

    _assert_outputs_equal(actual, expected)


@pytest.mark.parametrize("auto_batch_size", [False, len(ATOM_TYPES)])
@pytest.mark.parametrize(
    ("shared_aparam", "full_aparam"),
    [
        (
            APARAM_PER_ATOM,
            np.tile(APARAM_PER_ATOM, (NFRAMES, 1, 1)),
        ),
        (
            APARAM_GLOBAL,
            np.tile(APARAM_GLOBAL, (NFRAMES, len(ATOM_TYPES), 1)),
        ),
    ],
    ids=("per-atom", "all-atoms"),
)
def test_shared_aparam_is_tiled_before_batching(
    model_file: str,
    auto_batch_size: bool | int,
    shared_aparam: np.ndarray,
    full_aparam: np.ndarray,
) -> None:
    """Both documented atomic-parameter shorthand forms are frame-major."""
    evaluator = _backend(model_file, auto_batch_size)
    full_fparam = np.tile(FPARAM, (NFRAMES, 1))
    expected = evaluator.eval(
        COORDS,
        BOXES,
        ATOM_TYPES,
        fparam=full_fparam,
        aparam=full_aparam,
    )
    actual = evaluator.eval(
        COORDS,
        BOXES,
        ATOM_TYPES,
        fparam=full_fparam,
        aparam=shared_aparam,
    )

    _assert_outputs_equal(actual, expected)


@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        ("fparam", np.zeros(3), "wrong size of frame param"),
        ("aparam", np.zeros(3), "wrong size of atomic param"),
    ],
)
def test_invalid_parameter_size_has_clear_error(
    model_file: str, parameter: str, value: np.ndarray, message: str
) -> None:
    """Reject invalid sizes before NumPy emits an opaque reshape error."""
    evaluator = _backend(model_file, auto_batch_size=False)
    parameters = {
        "fparam": np.tile(FPARAM, (NFRAMES, 1)),
        "aparam": np.tile(APARAM_PER_ATOM, (NFRAMES, 1, 1)),
    }
    parameters[parameter] = value

    with pytest.raises(RuntimeError, match=message):
        evaluator.eval(COORDS, BOXES, ATOM_TYPES, **parameters)
