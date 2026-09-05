# SPDX-License-Identifier: LGPL-3.0-or-later
"""Padding must not enter real or magnetic dp test spin-force metrics."""

from types import (
    SimpleNamespace,
)

import numpy as np
import pytest

from deepmd.infer.model_test.ener import (
    SpinEnerTester,
    _OptionalEnerOutputs,
)
from deepmd.utils.weight_avg import (
    merge_weighted_errors,
    weighted_average,
)

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")


def _evaluate_spin(
    types,
    legacy,
    padding,
    padding_side="prediction",
    *,
    fixed_types=False,
    find_force=1.0,
    find_magnetic=1.0,
):
    """Evaluate controlled arrays with the real spin tester, including chunks."""
    types = np.asarray(types)
    nframes, natoms = types.shape
    valid = types >= 0
    force = np.arange(nframes * natoms * 3, dtype=float).reshape(nframes, natoms, 3)
    force = force / 10 + 1
    magnetic = force * 2
    reference = np.zeros_like(force)
    magnetic_reference = np.zeros_like(magnetic)
    force[~valid] = padding if padding_side in ("prediction", "both") else 0.0
    magnetic[~valid] = padding if padding_side in ("prediction", "both") else 0.0
    reference[~valid] = padding if padding_side in ("reference", "both") else 0.0
    magnetic_reference[~valid] = (
        padding if padding_side in ("reference", "both") else 0.0
    )
    # Deliberately include padding in the supplied mask: the tester must
    # intersect it with valid atom types, and preserve the raw detail rows.
    mask = types != 1
    dp = SimpleNamespace(get_ntypes_spin=lambda: int(legacy), get_ntypes=lambda: 3)
    tester = SpinEnerTester(dp, atomic=False)

    def evaluate(frame_slice):
        """Exercise the production force-metric entry point without inference."""
        errors = {}
        atype = types[0] if fixed_types else types[frame_slice]
        details = tester.force_errors(
            errors,
            data=None,
            test_data={
                "force": reference[frame_slice].reshape(-1, natoms * 3),
                "force_mag": magnetic_reference[frame_slice].reshape(-1, natoms * 3),
                "find_force_mag": find_magnetic,
            },
            atype=atype,
            natoms=natoms,
            prediction_force=force[frame_slice].reshape(-1, natoms * 3),
            optional_outputs=_OptionalEnerOutputs(
                None,
                None,
                None if legacy else magnetic[frame_slice].reshape(-1, natoms * 3),
                None if legacy else mask[frame_slice],
                None,
            ),
            find_force=find_force,
            find_atom_pref=0.0,
        )
        return errors, details

    before = [a.copy() for a in (force, reference, magnetic, magnetic_reference, mask)]
    errors, details = evaluate(slice(None))
    chunks = [evaluate(slice(i, i + 1))[0] for i in range(nframes)]
    expected = {}
    real_mask = valid & (types < 2) if legacy else valid
    mag_mask = types == 2 if legacy else valid & mask
    magnetic_prediction = force if legacy else magnetic
    magnetic_labels = reference if legacy else magnetic_reference
    for suffix, selection, prediction, labels, found in (
        ("fr", real_mask, force, reference, find_force),
        ("fm", mag_mask, magnetic_prediction, magnetic_labels, find_magnetic),
    ):
        if found and np.any(selection):
            delta = prediction[selection] - labels[selection]
            expected["mae_" + suffix] = (np.mean(abs(delta)), delta.size)
            expected["rmse_" + suffix] = (np.sqrt(np.mean(delta**2)), delta.size)

    # Detail output intentionally retains the existing padding layout.
    raw_real_mask = types < 2 if legacy else np.ones_like(valid)
    raw_mag_mask = types == 2 if legacy else mask
    np.testing.assert_equal(details.prediction_real, force[raw_real_mask])
    np.testing.assert_equal(details.reference_real, reference[raw_real_mask])
    if find_magnetic:
        np.testing.assert_equal(
            details.prediction_magnetic, magnetic_prediction[raw_mag_mask]
        )
        np.testing.assert_equal(
            details.reference_magnetic, magnetic_labels[raw_mag_mask]
        )
    else:
        assert details.prediction_magnetic is None
        assert details.reference_magnetic is None
    for original, actual in zip(
        before, (force, reference, magnetic, magnetic_reference, mask), strict=True
    ):
        np.testing.assert_equal(actual, original)
    return errors, chunks, expected


def _assert_metrics(errors, chunks, expected):
    """Check both in-chunk values and chunk/system aggregation weights."""
    for actual in (errors, merge_weighted_errors(chunks)):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_allclose(actual[key], value)
    actual = weighted_average(chunks)
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key][0])


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "types",
    [
        [[0, 1, 2], [2, 0, 1]],
        [[0, -1, 2], [-1, 0, 2]],
        [[0, 1, 2], [0, -1, 2]],
        [[-1, -1, -1], [0, 1, 2]],
        [[-1, -1, -1], [-1, -1, -1]],
    ],
)
@pytest.mark.parametrize("padding", [0.0, 10.0, np.nan])
@pytest.mark.parametrize("padding_side", ["prediction", "reference", "both"])
def test_spin_virtual_atom_metrics(types, legacy, padding, padding_side):
    """Zero, nonzero and NaN padding cannot change either metric or weight."""
    _assert_metrics(*_evaluate_spin(types, legacy, padding, padding_side))


@pytest.mark.parametrize("legacy", [False, True])
def test_spin_fixed_types_broadcast_across_frames(legacy):
    """A single type row selects valid force rows in every frame."""
    types = [[0, -1, 2], [0, -1, 2]]
    result = _evaluate_spin(types, legacy, np.nan, fixed_types=True)
    _assert_metrics(*result)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("find_force,find_magnetic", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_spin_missing_force_labels(legacy, find_force, find_magnetic):
    """Each available label controls only its corresponding force metric."""
    _assert_metrics(
        *_evaluate_spin(
            [[0, -1, 2]],
            legacy,
            np.nan,
            find_force=find_force,
            find_magnetic=find_magnetic,
        )
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_spin_without_magnetic_atoms_omits_magnetic_metrics(legacy):
    """Empty magnetic selections do not produce NaNs or zero-weight metrics."""
    result = _evaluate_spin([[1, -1, -1], [1, 1, -1]], legacy, np.nan)
    _assert_metrics(*result)
    assert result[0].keys() == {"mae_fr", "rmse_fr"}


def test_legacy_spin_without_real_rows_can_report_magnetic_metrics():
    """A magnetic-only legacy chunk does not reduce an empty real-force array."""
    result = _evaluate_spin([[2, -1]], True, np.nan)
    _assert_metrics(*result)
    assert result[0].keys() == {"mae_fm", "rmse_fm"}


def test_spin_missing_magnetic_arrays_still_raises():
    """A requested magnetic metric needs both arrays and the model mask."""
    tester = SpinEnerTester(
        SimpleNamespace(get_ntypes_spin=lambda: 0, get_ntypes=lambda: 1),
        atomic=False,
    )
    with pytest.raises(RuntimeError, match="magnetic force arrays and mask"):
        tester.force_errors(
            {},
            data=None,
            test_data={"force": np.zeros((1, 6)), "find_force_mag": 1.0},
            atype=np.array([[0, -1]]),
            natoms=2,
            prediction_force=np.zeros((1, 6)),
            optional_outputs=_OptionalEnerOutputs(None, None, None, None, None),
            find_force=1.0,
            find_atom_pref=0.0,
        )
