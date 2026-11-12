# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.utils.preset_out_bias import (
    normalize_preset_out_bias,
)
from deepmd.utils.vacuum_reference import (
    reference_charge_spin,
    reference_spin,
    store_reference_charge_spin,
    unpaired_electrons,
)


class TestVacuumReference(unittest.TestCase):
    def test_unpaired_electrons(self) -> None:
        # Hund's rule ground states: 1s1, 1s2, 2p3, 2p4, 3d6 4s2, 4f7 5d1 6s2
        type_map = ["H", "He", "N", "O", "Fe", "Gd"]
        np.testing.assert_array_equal(unpaired_electrons(type_map), [1, 0, 3, 2, 4, 8])

    def test_unknown_type_name(self) -> None:
        with self.assertRaises(ValueError):
            unpaired_electrons(["O", "H1"])

    def test_reference_conditions(self) -> None:
        # a table that names the state of its isolated atoms sets the
        # reference of every type it names
        type_map = ["O", "H", "Fe"]
        preset = normalize_preset_out_bias(
            {
                "energy": {
                    "O": {"value": -3.0, "charge": 0, "multiplicity": 3},
                    "H": {"value": -13.6, "charge": 0, "multiplicity": 2},
                    "Fe": {"value": -4.0, "charge": 0, "multiplicity": 5},
                }
            },
            type_map,
        )
        charge_spin = reference_charge_spin(type_map, preset, default=[0.0, 1.0])
        np.testing.assert_array_equal(charge_spin, [[0.0, 3.0], [0.0, 2.0], [0.0, 5.0]])
        self.assertEqual(charge_spin.dtype, np.float64)
        spin = reference_spin(type_map)
        np.testing.assert_array_equal(
            spin, [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 4.0]]
        )

    def test_reference_conditions_of_a_table_without_one(self) -> None:
        # a table computed under one common setting leaves the condition to
        # the model, which references every type it assigns to its default one
        preset = normalize_preset_out_bias(
            {"energy": {"O": -3.0, "H": -13.6, "Fe_U": -4.0}}, ["O", "H", "Fe_U"]
        )
        charge_spin = reference_charge_spin(["O", "H", "Fe_U"], preset, [0.0, 1.0])
        # a type that is no element needs no name either
        np.testing.assert_array_equal(charge_spin, [[0.0, 1.0]] * 3)
        # an assigned type without a condition and without a default is an error
        with self.assertRaisesRegex(ValueError, "default_chg_spin"):
            reference_charge_spin(["O", "H", "Fe_U"], preset, None)

    def test_types_without_a_preset_need_no_condition(self) -> None:
        # a table of some elements on a type map of all of them: only the
        # assigned types need a condition, an unassigned one references nothing
        type_map = ["O", "H", "Po"]
        preset = normalize_preset_out_bias(
            {
                "energy": {
                    "O": {"value": -3.0, "charge": 0, "multiplicity": 3},
                    "H": {"value": -13.6, "charge": 0, "multiplicity": 2},
                }
            },
            type_map,
        )
        charge_spin = reference_charge_spin(type_map, preset, None)
        np.testing.assert_array_equal(charge_spin[:2], [[0.0, 3.0], [0.0, 2.0]])
        self.assertEqual(charge_spin.shape, (3, 2))
        # an unassigned type takes no condition of the table
        self.assertNotIn(charge_spin[2].tolist(), [[0.0, 3.0], [0.0, 2.0]])
        # the entries of the output whose bias the preset assigns are read
        np.testing.assert_array_equal(
            reference_charge_spin(type_map, preset, [0.0, 1.0], "energy")[:2],
            [[0.0, 3.0], [0.0, 2.0]],
        )
        # a table with one entry per type is required
        with self.assertRaisesRegex(ValueError, "one entry per type"):
            reference_charge_spin(["O"], preset, None)

    def test_stored_conditions_cover_assigned_types_only(self) -> None:
        # the resolved condition of an assigned type is stored, the row of an
        # unassigned one stays unnamed so that a later preset names its state
        type_map = ["O", "H", "Po"]
        preset = normalize_preset_out_bias(
            {"energy": {"O": -3.0, "H": -13.6}}, type_map
        )
        conditions = reference_charge_spin(type_map, preset, [0.0, 1.0])
        store_reference_charge_spin(type_map, preset, conditions)
        self.assertEqual(preset["@charge_spin"], [[0.0, 1.0], [0.0, 1.0], None])
        # storing again from the stored preset changes nothing
        store_reference_charge_spin(
            type_map, preset, reference_charge_spin(type_map, preset, [0.0, 1.0])
        )
        self.assertEqual(preset["@charge_spin"], [[0.0, 1.0], [0.0, 1.0], None])
        # an unassigned type keeps no condition even when its row was resolved
        # under a default different from the one it later takes
        preset["@charge_spin"][:2] = [[0.0, 3.0], [0.0, 2.0]]
        np.testing.assert_array_equal(
            reference_charge_spin(type_map, preset, [1.0, 2.0])[:2],
            [[0.0, 3.0], [0.0, 2.0]],
        )
        self.assertEqual(preset["@charge_spin"][2], None)
        # a preset is left unset when the model carries none
        store_reference_charge_spin(type_map, None, conditions)

    def test_reference_spin_named_with_the_preset(self) -> None:
        # a named moment replaces the unpaired-electron count of its type only
        type_map = ["O", "H", "Fe_U"]
        preset = normalize_preset_out_bias(
            {
                "energy": {
                    "O": {"value": -3.0, "spin": 1.25},
                    "H": -13.6,
                    "Fe_U": {"value": -4.0, "spin": 0.0},
                }
            },
            type_map,
        )
        np.testing.assert_array_equal(
            reference_spin(type_map, preset),
            [[0.0, 0.0, 1.25], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        )
        # a preset without moments leaves every type at its ground state
        plain = normalize_preset_out_bias(
            {"energy": {"O": -3.0, "H": -13.6}}, ["O", "H"]
        )
        np.testing.assert_array_equal(
            reference_spin(["O", "H"], plain), reference_spin(["O", "H"])
        )
        # a type that is no element needs its moment named
        with self.assertRaisesRegex(ValueError, "Fe_U"):
            reference_spin(
                type_map, normalize_preset_out_bias({"energy": {}}, type_map)
            )


if __name__ == "__main__":
    unittest.main()
