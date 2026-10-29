# SPDX-License-Identifier: LGPL-3.0-or-later
"""Keep graph orchestration separate from descriptor preparation semantics."""

import copy
import unittest
from unittest.mock import (
    call,
    patch,
)

from deepmd.dpmodel.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.utils.argcheck import (
    normalize,
)


def normalized_model(descriptor: dict) -> dict:
    """Exercise the same strict schema normalization as training input."""
    return normalize(
        {
            "model": {
                "type_map": ["H"],
                "descriptor": descriptor,
                "fitting_net": {},
            },
            "learning_rate": {"start_lr": 0.001, "decay_steps": 10},
            "loss": {},
            "training": {
                "numb_steps": 1,
                "training_data": {"systems": ["unused"]},
            },
        },
        check=True,
    )["model"]


class TestNeighborPreparationCompatibility(unittest.TestCase):
    """Mock data measurement only, retaining model and descriptor dispatch."""

    @patch("deepmd.utils.update_sel.BaseUpdateSel.get_nbor_stat")
    def test_omitted_dpa1_sel_keeps_auto_statistics(self, stats) -> None:
        stats.return_value = (0.5, [25])
        model = normalized_model({"type": "dpa1", "rcut": 6.0})
        self.assertEqual(model["descriptor"]["sel"], "auto")
        before = copy.deepcopy(model)
        updated, minimum = BaseModel.update_sel(None, ["H"], model)
        stats.assert_called_once_with(None, ["H"], 6.0, mixed_type=True)
        self.assertEqual(updated["descriptor"]["sel"], 28)
        self.assertEqual(minimum, 0.5)
        self.assertEqual(model, before)

    @patch("deepmd.utils.update_sel.BaseUpdateSel.get_nbor_stat")
    def test_mixed_hybrid_prepares_and_constructs(self, stats) -> None:
        stats.side_effect = [(0.5, [4]), (0.4, [4])]
        model = normalized_model(
            {
                "type": "hybrid",
                "list": [
                    {
                        "type": "dpa1",
                        "rcut": 6.0,
                        "rcut_smth": 0.5,
                        "sel": 8,
                        "neuron": [8, 8],
                        "axis_neuron": 4,
                        "attn": 8,
                        "attn_layer": 0,
                        "tebd_dim": 4,
                    },
                    {
                        "type": "se_e2_a",
                        "rcut": 6.0,
                        "rcut_smth": 0.5,
                        "sel": [8],
                        "neuron": [8, 8],
                        "axis_neuron": 4,
                    },
                ],
            }
        )
        updated, minimum = BaseModel.update_sel(None, ["H"], model)
        self.assertEqual(
            stats.call_args_list,
            [
                call(None, ["H"], 6.0, mixed_type=True),
                call(None, ["H"], 6.0, mixed_type=False),
            ],
        )
        self.assertEqual(minimum, 0.4)
        descriptor = BaseDescriptor(**updated["descriptor"], ntypes=1)
        self.assertEqual(descriptor.get_sel(), [8])

    @patch("deepmd.utils.update_sel.BaseUpdateSel.get_nbor_stat")
    def test_dpa4c_only_measures_minimum_distance(self, stats) -> None:
        stats.return_value = (0.5, [0])
        model = normalized_model({"type": "dpa4c", "rcut": 6.0})
        before = copy.deepcopy(model)
        updated, minimum = BaseModel.update_sel(None, ["H"], model)
        # This tiny cutoff deliberately does not discover neighbor capacity.
        stats.assert_called_once_with(None, None, 1e-6, mixed_type=True)
        self.assertEqual(minimum, 0.5)
        self.assertEqual(updated, before)
        self.assertEqual(model, before)
        self.assertNotIn("sel", updated["descriptor"])

    @patch("deepmd.utils.update_sel.BaseUpdateSel.get_nbor_stat")
    def test_dpa4_preserves_explicit_and_auto_capacity(self, stats) -> None:
        stats.return_value = (0.5, [25])
        for sel, expected in [(32, 32), ("auto", 28), ("auto:1.5", 40)]:
            with self.subTest(sel=sel):
                stats.reset_mock()
                model = normalized_model({"type": "dpa4", "rcut": 6.0, "sel": sel})
                before = copy.deepcopy(model)
                updated, minimum = BaseModel.update_sel(None, ["H"], model)
                stats.assert_called_once_with(None, ["H"], 6.0, mixed_type=True)
                self.assertEqual(updated["descriptor"]["sel"], expected)
                self.assertEqual(minimum, 0.5)
                self.assertEqual(model, before)
