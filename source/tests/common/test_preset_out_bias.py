# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest

import numpy as np

from deepmd.utils.finetune import (
    get_index_between_two_maps,
)
from deepmd.utils.preset_out_bias import (
    normalize_preset_out_bias,
    override_assigned_bias,
    remap_preset_out_bias,
)


class TestNormalizePresetOutBias(unittest.TestCase):
    def setUp(self) -> None:
        self.type_map = ["O", "H", "B"]

    def test_none(self) -> None:
        self.assertIsNone(normalize_preset_out_bias(None, self.type_map))

    def test_list_form(self) -> None:
        preset = {"energy": [None, "1.", 3], "dipole": [None, [0, 1, 2], None]}
        out = normalize_preset_out_bias(preset, self.type_map)
        self.assertEqual(out["energy"], [None, [1.0], [3.0]])
        self.assertEqual(out["dipole"], [None, [0.0, 1.0, 2.0], None])
        # the configuration dict is left untouched
        self.assertEqual(preset["energy"], [None, "1.", 3])

    def test_element_dict_form(self) -> None:
        out = normalize_preset_out_bias(
            {"energy": {"B": 3.0, "H": [1.0]}, "dipole": {}}, self.type_map
        )
        self.assertEqual(out["energy"], [None, [1.0], [3.0]])
        self.assertEqual(out["dipole"], [None, None, None])

    def test_array_entries(self) -> None:
        out = normalize_preset_out_bias(
            {
                "polar": [np.eye(2), None, np.array(4.0)],
                "energy": np.array([[1.0], [2.0], [3.0]]),
            },
            self.type_map,
        )
        self.assertEqual(out["polar"], [[[1.0, 0.0], [0.0, 1.0]], None, [4.0]])
        self.assertEqual(out["energy"], [[1.0], [2.0], [3.0]])

    def test_idempotent_and_json_round_trip(self) -> None:
        out = normalize_preset_out_bias(
            {"energy": {"H": -13.6}, "polar": {"O": [[1.0, 0.0], [0.0, 1.0]]}},
            self.type_map,
        )
        self.assertEqual(normalize_preset_out_bias(out, self.type_map), out)
        self.assertEqual(
            normalize_preset_out_bias(json.loads(json.dumps(out)), self.type_map), out
        )

    def test_errors(self) -> None:
        for bad in (
            {"energy": {"C": 3.0}},
            {"energy": [None]},
            {"energy": 1.0},
            {"energy": [None, 1.0 + 2.0j, None]},
            {"energy": [None, "1.0 + 2.0j", None]},
            {"dipole": [None, [0.0, None, 2.0], None]},
        ):
            with self.assertRaises(ValueError):
                normalize_preset_out_bias(bad, self.type_map)


class TestRemapPresetOutBias(unittest.TestCase):
    def test_follows_out_bias_remap(self) -> None:
        old_map = ["O", "H", "B", "C"]
        preset = normalize_preset_out_bias(
            {"energy": {"O": 1.0, "B": 3.0, "C": 4.0}}, old_map
        )
        # per-type reference values in the layout of a stored out_bias
        out_bias = np.array([1.0, np.nan, 3.0, 4.0])
        for new_map in (
            ["C", "O", "H", "B"],
            ["B", "N", "O"],
            ["N", "F"],
            ["H"],
        ):
            remap_index, _ = get_index_between_two_maps(old_map, new_map)
            remapped = remap_preset_out_bias(preset, remap_index)["energy"]
            reference = np.concatenate([out_bias, np.full(len(new_map), np.nan)])[
                remap_index
            ]
            self.assertEqual(len(remapped), len(new_map))
            for got, want in zip(remapped, reference, strict=True):
                if np.isnan(want):
                    self.assertIsNone(got)
                else:
                    self.assertEqual(got, [want])

    def test_none(self) -> None:
        self.assertIsNone(remap_preset_out_bias(None, [0, 1]))


class TestOverrideAssignedBias(unittest.TestCase):
    def test_flattened_bias_against_shaped_preset(self) -> None:
        bias = np.arange(12, dtype=np.float64).reshape(3, 4)
        assigned = np.full((3, 2, 2), np.nan)
        assigned[1] = [[10.0, 11.0], [12.0, 13.0]]
        out = override_assigned_bias(bias, assigned)
        expected = bias.copy()
        expected[1] = [10.0, 11.0, 12.0, 13.0]
        np.testing.assert_array_equal(out, expected)
        self.assertEqual(out.shape, bias.shape)
        # the input is left untouched
        self.assertEqual(bias[1, 0], 4.0)

    def test_none(self) -> None:
        bias = np.ones((2, 3))
        self.assertIs(override_assigned_bias(bias, None), bias)
