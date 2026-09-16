# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.utils.econf_embd import (
    electronic_configuration_embedding,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
)
from deepmd.utils.preset_out_bias import (
    bundled_preset_out_bias_tables,
    load_preset_out_bias_table,
    normalize_preset_out_bias,
    override_assigned_bias,
    preset_out_bias_rows,
    remap_preset_out_bias,
    resolve_preset_out_bias_tables,
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

    def test_unknown_elements_ignored(self) -> None:
        out = normalize_preset_out_bias(
            {"energy": {"C": 3.0, "H": -13.6, "B": None}}, self.type_map
        )
        self.assertEqual(out["energy"], [None, [-13.6], None])

    def test_json_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "energy_bias.json"
            table.write_text(json.dumps({"H": -13.6, "B": None, "C": 3.0}))
            out = normalize_preset_out_bias({"energy": str(table)}, self.type_map)
            self.assertEqual(out["energy"], [None, [-13.6], None])
            # a relative path resolves against the working directory
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                out = normalize_preset_out_bias(
                    {"energy": "energy_bias.json"}, self.type_map
                )
            finally:
                os.chdir(cwd)
            self.assertEqual(out["energy"], [None, [-13.6], None])
            (Path(tmp) / "list.json").write_text("[1, 2]")
            with self.assertRaises(ValueError):
                normalize_preset_out_bias(
                    {"energy": str(Path(tmp) / "list.json")}, self.type_map
                )

    def test_errors(self) -> None:
        for bad in (
            {"energy": [None]},
            {"energy": 1.0},
            {"energy": [None, 1.0 + 2.0j, None]},
            {"energy": [None, "1.0 + 2.0j", None]},
            {"energy": [None, np.inf, None]},
            {"energy": [None, -np.inf, None]},
            {"dipole": [None, [0.0, None, 2.0], None]},
        ):
            with self.assertRaises(ValueError):
                normalize_preset_out_bias(bad, self.type_map)


class TestBundledTables(unittest.TestCase):
    def test_tables(self) -> None:
        tables = bundled_preset_out_bias_tables()
        self.assertEqual(
            sorted(tables), ["oc20", "odac25", "omat24", "omc25", "omol25"]
        )
        for name, table in tables.items():
            with self.subTest(name=name):
                self.assertLessEqual(
                    set(table), set(electronic_configuration_embedding)
                )
                self.assertTrue(np.all(np.isfinite(list(table.values()))))
        self.assertEqual(len(tables["omat24"]), 89)
        self.assertEqual(tables["omat24"]["H"], -1.11700253)

    def test_name_before_path(self) -> None:
        tables = bundled_preset_out_bias_tables()
        self.assertEqual(load_preset_out_bias_table("omat24"), tables["omat24"])
        # an element of the type map outside the table stays unassigned
        out = normalize_preset_out_bias({"energy": "omat24"}, ["O", "H", "Po"])
        self.assertEqual(out["energy"], [[-1.54797136], [-1.11700253], None])
        config = {"type_map": ["H"], "preset_out_bias": {"energy": "omol25"}}
        resolved = resolve_preset_out_bias_tables(config)["preset_out_bias"]
        self.assertEqual(resolved, {"energy": tables["omol25"]})
        with self.assertRaises(ValueError):
            load_preset_out_bias_table("no_such_table.json")


class TestResolvePresetOutBiasTables(unittest.TestCase):
    def test_single_and_multi_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            table = Path(tmp) / "energy_bias.json"
            table.write_text(json.dumps({"H": -13.6}))
            single = {"type_map": ["H"], "preset_out_bias": {"energy": str(table)}}
            out = resolve_preset_out_bias_tables(single)
            self.assertEqual(out["preset_out_bias"], {"energy": {"H": -13.6}})
            # the configuration passed in is left untouched
            self.assertEqual(single["preset_out_bias"], {"energy": str(table)})
            multi = {
                "preset_out_bias": {"energy": str(table)},
                "model_dict": {
                    "a": single,
                    "b": {"type_map": ["H"], "preset_out_bias": {"energy": {"H": 1.0}}},
                    "c": {"type_map": ["H"]},
                },
            }
            out = resolve_preset_out_bias_tables(multi)
            # a preset next to model_dict is resolved as well
            self.assertEqual(out["preset_out_bias"], {"energy": {"H": -13.6}})
            self.assertEqual(
                out["model_dict"]["a"]["preset_out_bias"], {"energy": {"H": -13.6}}
            )
            self.assertIs(out["model_dict"]["b"], multi["model_dict"]["b"])
            self.assertIs(out["model_dict"]["c"], multi["model_dict"]["c"])

    def test_without_tables(self) -> None:
        config = {"type_map": ["H"], "preset_out_bias": {"energy": [1.0]}}
        self.assertIs(resolve_preset_out_bias_tables(config), config)
        config = {"type_map": ["H"]}
        self.assertIs(resolve_preset_out_bias_tables(config), config)


class TestPresetOutBiasRows(unittest.TestCase):
    def setUp(self) -> None:
        self.type_map = ["O", "H", "B"]
        self.preset = normalize_preset_out_bias(
            {"energy": {"H": -13.6}, "dipole": {"H": [1.0, 2.0, 3.0]}}, self.type_map
        )
        self.stored = np.arange(2 * 3 * 3, dtype=np.float64).reshape(2, 3, 3)
        self.keys = ["energy", "dipole"]
        self.sizes = [1, 3]

    def rows(self, observed: list[str], keep_unassigned: bool) -> dict[str, np.ndarray]:
        return preset_out_bias_rows(
            self.preset,
            self.type_map,
            observed,
            self.stored,
            self.keys,
            self.sizes,
            keep_unassigned,
        )

    def test_zero_or_stored_for_unassigned_types(self) -> None:
        rows = self.rows(["H"], keep_unassigned=False)
        np.testing.assert_array_equal(rows["energy"], [[0.0], [-13.6], [0.0]])
        np.testing.assert_array_equal(
            rows["dipole"], [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
        )
        rows = self.rows(["H"], keep_unassigned=True)
        np.testing.assert_array_equal(
            rows["energy"], [[self.stored[0, 0, 0]], [-13.6], [self.stored[0, 2, 0]]]
        )
        np.testing.assert_array_equal(
            rows["dipole"], [self.stored[1, 0], [1.0, 2.0, 3.0], self.stored[1, 2]]
        )

    def test_unassigned_output_is_not_fixed(self) -> None:
        self.preset = normalize_preset_out_bias(
            {"energy": [None, None, None], "dipole": {"H": [1.0, 2.0, 3.0]}},
            self.type_map,
        )
        self.assertEqual(list(self.rows(["H"], keep_unassigned=False)), ["dipole"])

    def test_observed_types_must_be_assigned(self) -> None:
        with self.assertRaisesRegex(ValueError, r"energy.*\['O'\]"):
            self.rows(["O", "H"], keep_unassigned=False)

    def test_unknown_and_excluded_observed_types(self) -> None:
        # an observed name outside the type map is ignored and an excluded
        # type needs no preset
        rows = preset_out_bias_rows(
            self.preset,
            self.type_map,
            ["H", "Fe", "B"],
            self.stored,
            self.keys,
            self.sizes,
            keep_unassigned=False,
            excluded_types=[2],
        )
        np.testing.assert_array_equal(rows["energy"], [[0.0], [-13.6], [0.0]])
        with self.assertRaisesRegex(ValueError, r"energy.*\['B'\]"):
            self.rows(["H", "B"], keep_unassigned=False)


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
