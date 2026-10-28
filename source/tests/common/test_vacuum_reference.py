# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.utils.vacuum_reference import (
    reference_charge_spin,
    reference_spin,
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
        type_map = ["O", "H"]
        charge_spin = reference_charge_spin(type_map)
        np.testing.assert_array_equal(charge_spin, [[0.0, 3.0], [0.0, 2.0]])
        self.assertEqual(charge_spin.dtype, np.float64)
        spin = reference_spin(type_map)
        np.testing.assert_array_equal(spin, [[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
