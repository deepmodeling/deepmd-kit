# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for virtual placeholders in dpmodel SpinModel inputs."""

import numpy as np
import pytest

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.model.model import (
    get_model,
)


MODEL_CONFIG = {
    "type_map": ["A", "B", "C"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [4, 4, 4],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [3, 6],
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "type": "ener",
        "neuron": [5, 5],
        "precision": "float64",
        "seed": 1,
    },
    # Keep the final real type magnetic so an accidental ``mask[-1]`` lookup
    # is observable instead of being hidden by a zero scale.
    "spin": {"use_spin": [False, False, True], "virtual_scale": [0.5]},
}


@pytest.fixture
def model():
    """Build a small NumPy spin model with a magnetic final real type."""
    return get_model(MODEL_CONFIG)


def test_dense_spin_expansion_preserves_virtual_types(model) -> None:
    """A dense placeholder and its spin partner must both stay virtual."""
    coord = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
    atype = np.array([[0, -1, 2]], dtype=np.int32)
    spin = np.ones_like(coord)

    coord_updated, atype_updated, coord_corr = model.process_spin_input(
        coord, atype, spin
    )

    np.testing.assert_array_equal(atype_updated, [[0, -1, 2, 3, -1, 5]])
    # The placeholder's virtual partner has neither a displacement nor a
    # virial correction, even though the final real type is magnetic.
    np.testing.assert_array_equal(coord_updated[:, 4], coord[:, 1])
    np.testing.assert_array_equal(coord_corr[:, 4], 0.0)

    _, magnetic_output, magnetic_mask = model.process_spin_output(
        atype, np.ones((1, 6, 3), dtype=np.float64)
    )
    np.testing.assert_array_equal(magnetic_output[:, 1], 0.0)
    np.testing.assert_array_equal(magnetic_mask[:, 1], False)

    prediction = model(coord, atype, spin, box=None)
    np.testing.assert_array_equal(prediction["atom_energy"][:, 1], 0.0)
    np.testing.assert_array_equal(prediction["mask_mag"][:, 1], False)

    changed_coord = coord.copy()
    changed_spin = spin.copy()
    changed_coord[:, 1] += 100.0
    changed_spin[:, 1] += 100.0
    changed_prediction = model(changed_coord, atype, changed_spin, box=None)
    np.testing.assert_allclose(changed_prediction["energy"], prediction["energy"])
    np.testing.assert_allclose(
        changed_prediction["atom_energy"][:, [0, 2]],
        prediction["atom_energy"][:, [0, 2]],
    )


def test_lower_spin_expansion_preserves_virtual_types(model) -> None:
    """Local and ghost placeholders remain negative in the switched layout."""
    extended_coord = np.arange(12, dtype=np.float64).reshape(1, 4, 3)
    extended_atype = np.array([[0, -1, 2, -1]], dtype=np.int32)
    extended_spin = np.ones_like(extended_coord)
    nlist = np.array([[[2, -1], [0, -1]]], dtype=np.int32)
    mapping = np.array([[0, 1, 0, 1]], dtype=np.int32)

    (
        coord_updated,
        atype_updated,
        _,
        _,
        coord_corr,
    ) = model.process_spin_input_lower(
        extended_coord,
        extended_atype,
        extended_spin,
        nlist,
        mapping=mapping,
    )

    np.testing.assert_array_equal(atype_updated, [[0, -1, 3, -1, 2, -1, 5, -1]])
    for real_index, virtual_index in ((1, 3), (3, 7)):
        np.testing.assert_array_equal(
            coord_updated[:, virtual_index], extended_coord[:, real_index]
        )
        np.testing.assert_array_equal(coord_corr[:, virtual_index], 0.0)

    _, magnetic_output, magnetic_mask = model.process_spin_output_lower(
        extended_atype,
        np.ones((1, 8, 3), dtype=np.float64),
        nloc=2,
    )
    np.testing.assert_array_equal(magnetic_output[:, [1, 3]], 0.0)
    np.testing.assert_array_equal(magnetic_mask[:, [1, 3]], False)


def test_virtual_type_lookup_supports_array_api_strict(model) -> None:
    """The masked lookup must not rely on NumPy negative-index semantics."""
    xp = pytest.importorskip("array_api_strict")
    coord = xp.asarray(np.arange(9, dtype=np.float64).reshape(1, 3, 3))
    atype = xp.asarray(np.array([[0, -1, 2]], dtype=np.int64))
    spin = xp.ones_like(coord)

    coord_updated, atype_updated, coord_corr = model.process_spin_input(
        coord, atype, spin
    )
    np.testing.assert_array_equal(to_numpy_array(atype_updated), [[0, -1, 2, 3, -1, 5]])
    np.testing.assert_array_equal(to_numpy_array(coord_updated)[:, 4], [[3, 4, 5]])
    np.testing.assert_array_equal(to_numpy_array(coord_corr)[:, 4], 0.0)
