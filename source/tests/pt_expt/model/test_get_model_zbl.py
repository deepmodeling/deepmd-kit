# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test legacy ZBL routing in the pt_expt model factory."""

import os
import unittest

from deepmd.pt_expt.model import (
    DPZBLModel,
    get_model,
)

from ...seed import (
    GLOBAL_SEED,
)

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TAB_FILE = os.path.join(
    TESTS_DIR,
    "pt",
    "model",
    "water",
    "data",
    "zbl_tab_potential",
    "H2O_tab_potential.txt",
)


class TestGetModelZBL(unittest.TestCase):
    """Ensure ``use_srtab`` selects the backend-native ZBL model."""

    def test_get_model_routes_use_srtab_to_zbl(self) -> None:
        model = get_model(
            {
                "type_map": ["O", "H", "B"],
                "use_srtab": TAB_FILE,
                "smin_alpha": 0.37,
                "sw_rmin": 0.2,
                "sw_rmax": 4.0,
                "descriptor": {
                    "type": "dpa1",
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "sel": 20,
                    "neuron": [3, 6],
                    "axis_neuron": 2,
                    "attn": 4,
                    "attn_layer": 2,
                    "attn_dotr": True,
                    "attn_mask": False,
                    "activation_function": "tanh",
                    "set_davg_zero": True,
                    "type_one_side": True,
                    "seed": GLOBAL_SEED,
                },
                "fitting_net": {
                    "type": "ener",
                    "neuron": [],
                    "seed": GLOBAL_SEED,
                },
            }
        )
        self.assertIsInstance(model, DPZBLModel)
        self.assertEqual(model.atomic_model.smin_alpha, 0.37)


if __name__ == "__main__":
    unittest.main()
