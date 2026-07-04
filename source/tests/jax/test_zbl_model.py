# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the JAX ZBL model factory deep-copies input and injects type_map.

The JAX (and TF2) ZBL factory used to mutate the caller's config in place (via
``pop("type")``) and did not propagate ``type_map`` into the descriptor and
fitting sub-configs, unlike the standard and dpmodel factories. This checks the
factory leaves the input dict unchanged and that the constructed descriptor and
fitting carry the model ``type_map``.
"""

import os
import unittest
from copy import (
    deepcopy,
)

from deepmd.jax.model.model import (
    get_zbl_model,
)

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRTAB = os.path.join(
    TESTS_DIR, "pt", "water", "data", "zbl_tab_potential", "H2O_tab_potential.txt"
)


def _zbl_config() -> dict:
    return {
        "type_map": ["O", "H", "B"],
        "use_srtab": SRTAB,
        "sw_rmin": 0.2,
        "sw_rmax": 4.0,
        "smin_alpha": 0.1,
        # ZBL wraps a linear atomic model, which requires a mixed-type descriptor
        "descriptor": {
            "type": "se_atten",
            "sel": 40,
            "rcut_smth": 0.5,
            "rcut": 4.0,
            "neuron": [3, 6],
            "axis_neuron": 2,
            "attn": 8,
            "attn_layer": 2,
            "attn_dotr": True,
            "attn_mask": False,
            "set_davg_zero": True,
            "type_one_side": True,
            "seed": 1,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [5, 5],
            "seed": 1,
        },
    }


class TestJAXZBLModelFactory(unittest.TestCase):
    def test_does_not_mutate_input(self) -> None:
        data = _zbl_config()
        orig = deepcopy(data)
        get_zbl_model(data)
        self.assertEqual(data, orig)

    def test_injects_type_map_into_subconfigs(self) -> None:
        data = _zbl_config()
        model = get_zbl_model(data)
        dp_atomic = model.atomic_model.models[0]
        self.assertEqual(dp_atomic.descriptor.get_type_map(), data["type_map"])
        self.assertEqual(dp_atomic.fitting_net.get_type_map(), data["type_map"])


if __name__ == "__main__":
    unittest.main()
