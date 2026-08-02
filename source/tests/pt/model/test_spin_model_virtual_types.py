# SPDX-License-Identifier: LGPL-3.0-or-later
"""The pt SpinModel must treat virtual placeholders like the dpmodel one.

``deepmd/pt_expt`` subclasses the dpmodel ``SpinModel`` without overriding
these methods, so a divergence here means the same weights give different
answers depending on the backend.  ``-1`` reaches this code in ordinary use:
``deepmd/utils/data.py`` appends it as the virtual-atom padding for mixed-type
systems, and batched extended regions are padded to a uniform ``nall``.
"""

import unittest

import numpy as np
import torch

from deepmd.dpmodel.model.model import get_model as get_dp_model
from deepmd.pt.model.model import get_model as get_pt_model
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
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
    # Keep the final real type magnetic so an accidental ``mask[-1]`` lookup is
    # observable instead of being hidden by a zero scale.
    "spin": {"use_spin": [False, False, True], "virtual_scale": [0.5]},
}

# The first real type is magnetic here, which is when the pt clamp-to-row-0
# lookup gave a padded slot a real spin scale and a True magnetic mask.
MAGNETIC_FIRST_CONFIG = {
    **MODEL_CONFIG,
    "spin": {"use_spin": [True, False, False], "virtual_scale": [0.5]},
}


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).to(device=env.DEVICE)


class TestPtSpinModelVirtualTypes(unittest.TestCase):
    """Every pt lookup must mask ``atype < 0`` exactly as dpmodel does."""

    def test_dense_spin_expansion_preserves_virtual_types(self) -> None:
        model = get_pt_model(MODEL_CONFIG)
        coord = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
        atype = np.array([[0, -1, 2]], dtype=np.int64)
        spin = np.ones_like(coord)

        coord_updated, atype_updated, coord_corr = model.process_spin_input(
            _tensor(coord), _tensor(atype), _tensor(spin)
        )

        np.testing.assert_array_equal(
            to_numpy_array(atype_updated), [[0, -1, 2, 3, -1, 5]]
        )
        np.testing.assert_array_equal(to_numpy_array(coord_updated)[:, 4], coord[:, 1])
        np.testing.assert_array_equal(to_numpy_array(coord_corr)[:, 4], 0.0)

    def test_lower_spin_expansion_preserves_virtual_types(self) -> None:
        model = get_pt_model(MODEL_CONFIG)
        extended_coord = np.arange(12, dtype=np.float64).reshape(1, 4, 3)
        extended_atype = np.array([[0, -1, 2, -1]], dtype=np.int64)
        extended_spin = np.ones_like(extended_coord)
        nlist = np.array([[[2, -1], [0, -1]]], dtype=np.int64)

        (
            coord_updated,
            atype_updated,
            _,
            _,
            coord_corr,
        ) = model.process_spin_input_lower(
            _tensor(extended_coord),
            _tensor(extended_atype),
            _tensor(extended_spin),
            _tensor(nlist),
            mapping=_tensor(np.array([[0, 1, 0, 1]], dtype=np.int64)),
        )

        np.testing.assert_array_equal(
            to_numpy_array(atype_updated), [[0, -1, 3, -1, 2, -1, 5, -1]]
        )
        for real_index, virtual_index in ((1, 3), (3, 7)):
            np.testing.assert_array_equal(
                to_numpy_array(coord_updated)[:, virtual_index],
                extended_coord[:, real_index],
            )
            np.testing.assert_array_equal(
                to_numpy_array(coord_corr)[:, virtual_index], 0.0
            )

    def test_matches_dpmodel_when_the_first_real_type_is_magnetic(self) -> None:
        """The clamp-to-row-0 lookup only diverged when type 0 is magnetic."""
        pt_model = get_pt_model(MAGNETIC_FIRST_CONFIG)
        dp_model = get_dp_model(MAGNETIC_FIRST_CONFIG)
        coord = np.arange(9, dtype=np.float64).reshape(1, 3, 3)
        atype = np.array([[0, -1, 2]], dtype=np.int64)
        spin = np.ones_like(coord)
        out_tensor = np.ones((1, 6, 3), dtype=np.float64)

        pt_coord, pt_atype, pt_corr = pt_model.process_spin_input(
            _tensor(coord), _tensor(atype), _tensor(spin)
        )
        dp_coord, dp_atype, dp_corr = dp_model.process_spin_input(coord, atype, spin)
        np.testing.assert_array_equal(to_numpy_array(pt_atype), dp_atype)
        np.testing.assert_allclose(to_numpy_array(pt_coord), dp_coord)
        np.testing.assert_allclose(to_numpy_array(pt_corr), dp_corr)
        # The placeholder gets no displacement even though type 0 is magnetic.
        np.testing.assert_array_equal(to_numpy_array(pt_coord)[:, 4], coord[:, 1])

        _, pt_mag, pt_mask = pt_model.process_spin_output(
            _tensor(atype), _tensor(out_tensor)
        )
        _, dp_mag, dp_mask = dp_model.process_spin_output(atype, out_tensor)
        np.testing.assert_allclose(to_numpy_array(pt_mag), dp_mag)
        np.testing.assert_array_equal(to_numpy_array(pt_mask), dp_mask)
        # mask_mag feeds the magnetic-force loss, so a True here would count a
        # padded slot as a real magnetic atom.
        np.testing.assert_array_equal(to_numpy_array(pt_mask)[:, 1], False)


if __name__ == "__main__":
    unittest.main()
