# SPDX-License-Identifier: LGPL-3.0-or-later
"""Virtual padding must not contribute to dp test force/Hessian metrics."""

from types import (
    SimpleNamespace,
)

import numpy as np
import pytest

from deepmd.infer.model_test.base import (
    ChunkContext,
)
from deepmd.infer.model_test.ener import (
    EnerTester,
)
from deepmd.utils.weight_avg import (
    merge_weighted_errors,
    weighted_average,
)


class _Evaluator:
    """Controlled predictions for testing metric reduction, not inference."""

    has_efield = False
    has_spin = False
    has_hessian = True

    def __init__(self, force, hessian):
        self.force = force
        self.hessian = hessian

    def get_dim_fparam(self):
        return 0

    def get_dim_aparam(self):
        return 0

    def has_chg_spin_ebd(self):
        return False

    def eval(self, coord, box, atype, **kwargs):
        nframes = len(coord)
        return (
            np.zeros((nframes, 1)),
            self.force,
            np.zeros((nframes, 9)),
            self.hessian,
        )


def _evaluate(
    types,
    padding_error,
    *,
    detail_file=None,
    find_force=1.0,
    mixed_type=True,
    zero_real_pref=False,
):
    types = np.asarray(types)
    nframes, natoms = types.shape
    valid = np.repeat(types >= 0, 3, axis=1)
    valid_hessian = valid[:, :, None] & valid[:, None, :]
    # Different errors in every row/column catch masking just one Hessian axis.
    force_error = np.arange(valid.size).reshape(valid.shape) / 10 + 1
    hessian_error = np.arange(valid_hessian.size).reshape(valid_hessian.shape) / 10 + 1
    force = np.where(valid, force_error, padding_error)
    hessian = np.where(valid_hessian, hessian_error, padding_error)
    atom_pref = np.arange(valid.size).reshape(valid.shape) + 1.0
    if zero_real_pref:
        atom_pref[valid] = 0.0
    labels = {
        "type": types,
        "box": np.zeros((nframes, 9)),
        "coord": np.zeros((nframes, natoms * 3)),
        "energy": np.zeros((nframes, 1)),
        "virial": np.zeros((nframes, 9)),
        "force": np.zeros_like(force),
        "hessian": np.zeros((nframes, (natoms * 3) ** 2)),
        "atom_pref": atom_pref,
        "find_energy": 0.0,
        "find_force": find_force,
        "find_virial": 0.0,
        "find_atom_pref": 1.0,
    }
    context = ChunkContext("padding", detail_file, False, 0)
    tester = EnerTester(_Evaluator(force, hessian), atomic=False)
    data = SimpleNamespace(mixed_type=mixed_type, pbc=False)
    errors = tester.evaluate_chunk(data, labels, context)

    chunks = []
    for frame in range(nframes):
        frame_labels = {
            key: value[frame : frame + 1] if isinstance(value, np.ndarray) else value
            for key, value in labels.items()
        }
        frame_tester = EnerTester(
            _Evaluator(force[frame : frame + 1], hessian[frame : frame + 1]),
            atomic=False,
        )
        chunks.append(
            frame_tester.evaluate_chunk(
                data, frame_labels, ChunkContext("padding", None, False, frame)
            )
        )
    return (
        errors,
        chunks,
        force_error[valid],
        hessian_error[valid_hessian],
        atom_pref[valid],
    )


@pytest.mark.parametrize(
    "types",
    [
        [[0, 0], [0, 0]],
        [[0, -1], [-1, 0]],
        [[0, 0], [-1, 0]],
        [[-1, -1], [0, -1]],
    ],
)
@pytest.mark.parametrize("padding_error", [0.0, 10.0, np.nan])
def test_virtual_atom_metrics(types, padding_error):
    errors, chunks, force, hessian, pref = _evaluate(types, padding_error)
    expected = {
        "mae_f": (np.mean(abs(force)), force.size),
        "rmse_f": (np.sqrt(np.mean(force**2)), force.size),
        "mae_h": (np.mean(abs(hessian)), hessian.size),
        "rmse_h": (np.sqrt(np.mean(hessian**2)), hessian.size),
        "mae_fw": (np.average(abs(force), weights=pref), pref.sum()),
        "rmse_fw": (np.sqrt(np.average(force**2, weights=pref)), pref.sum()),
    }
    for actual in (errors, merge_weighted_errors(chunks)):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value)
    # The same weights are also consumed by run-level, cross-system reduction.
    for key, value in weighted_average(chunks).items():
        np.testing.assert_allclose(value, expected[key][0])


def test_all_virtual_chunk_has_no_atom_metrics():
    errors, chunks, *_ = _evaluate([[-1, -1]], np.nan)
    assert errors == {}
    assert merge_weighted_errors(chunks) == {}


def test_fixed_atom_types_broadcast_across_frames():
    mixed, *_ = _evaluate([[0, -1], [0, -1]], np.nan)
    fixed, *_ = _evaluate([[0, -1], [0, -1]], np.nan, mixed_type=False)
    assert mixed == fixed


def test_only_virtual_atoms_have_force_weights():
    errors, chunks, *_ = _evaluate([[0, -1], [-1, 0]], np.nan, zero_real_pref=True)
    assert errors.keys() == {"mae_f", "rmse_f", "mae_h", "rmse_h"}
    for key, value in merge_weighted_errors(chunks).items():
        np.testing.assert_allclose(value, errors[key])


def test_missing_force_labels_do_not_report_force():
    errors, *_ = _evaluate([[0, -1]], np.nan, find_force=0.0)
    assert errors.keys() == {"mae_h", "rmse_h"}


def test_padding_keeps_detail_file_layout(tmp_path):
    detail = str(tmp_path / "details")
    _evaluate([[0, -1], [-1, 0]], 10.0, detail_file=detail)
    assert np.loadtxt(detail + ".f.out").shape == (4, 6)
    assert np.loadtxt(detail + ".h.out").shape == (72, 2)
