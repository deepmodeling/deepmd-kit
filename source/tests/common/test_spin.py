# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np

from deepmd.utils.spin import (
    Spin,
)

CUR_DIR = os.path.dirname(__file__)


class SpinTest(unittest.TestCase):
    def setUp(self) -> None:
        type_map_1 = ["H", "O"]
        self.use_spin_1 = [False, False]
        self.virtual_scale_1 = [0.1, 0.1]

        type_map_2 = ["B", "Ni", "O"]
        self.use_spin_2 = [False, True, False]
        self.virtual_scale_2 = [0.1, 0.1, 0.1]

        type_map_3 = ["H", "O", "B", "Ni", "O"]
        self.use_spin_3 = [False, False, False, True, False]
        self.virtual_scale_3 = [0.1, 0.1, 0.1, 0.1, 0.1]

        self.virtual_scale_float = 0.1
        self.virtual_scale_nspin = [0.1]

        self.spin_1 = Spin(self.use_spin_1, self.virtual_scale_1)
        self.spin_2 = Spin(self.use_spin_2, self.virtual_scale_2)
        self.spin_3 = Spin(self.use_spin_3, self.virtual_scale_3)
        self.spin_3_float = Spin(self.use_spin_3, self.virtual_scale_float)
        self.spin_3_nspin = Spin(self.use_spin_3, self.virtual_scale_nspin)

        self.expect_virtual_scale_mask_1 = np.array([0.0, 0.0])
        self.expect_virtual_scale_mask_2 = np.array([0.0, 0.1, 0.0])
        self.expect_virtual_scale_mask_3 = np.array([0.0, 0.0, 0.0, 0.1, 0.0])

        self.expect_pair_exclude_types_1 = [
            [2, 0],
            [2, 1],
            [2, 2],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ]
        self.expect_pair_exclude_types_2 = [
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
            [3, 5],
            [5, 0],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
        ]
        self.expect_pair_exclude_types_3 = [
            [5, 0],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
            [5, 5],
            [5, 6],
            [5, 7],
            [5, 8],
            [5, 9],
            [6, 0],
            [6, 1],
            [6, 2],
            [6, 3],
            [6, 4],
            [6, 5],
            [6, 6],
            [6, 7],
            [6, 8],
            [6, 9],
            [7, 0],
            [7, 1],
            [7, 2],
            [7, 3],
            [7, 4],
            [7, 5],
            [7, 6],
            [7, 7],
            [7, 8],
            [7, 9],
            [9, 0],
            [9, 1],
            [9, 2],
            [9, 3],
            [9, 4],
            [9, 5],
            [9, 6],
            [9, 7],
            [9, 8],
            [9, 9],
        ]

    def test_ntypes(self) -> None:
        self.assertEqual(self.spin_1.get_ntypes_real(), 2)
        self.assertEqual(self.spin_1.get_ntypes_spin(), 0)
        self.assertEqual(self.spin_1.get_ntypes_real_and_spin(), 2)
        self.assertEqual(self.spin_1.get_ntypes_input(), 4)

        self.assertEqual(self.spin_2.get_ntypes_real(), 3)
        self.assertEqual(self.spin_2.get_ntypes_spin(), 1)
        self.assertEqual(self.spin_2.get_ntypes_real_and_spin(), 4)
        self.assertEqual(self.spin_2.get_ntypes_input(), 6)

        self.assertEqual(self.spin_3.get_ntypes_real(), 5)
        self.assertEqual(self.spin_3.get_ntypes_spin(), 1)
        self.assertEqual(self.spin_3.get_ntypes_real_and_spin(), 6)
        self.assertEqual(self.spin_3.get_ntypes_input(), 10)

    def test_use_spin(self) -> None:
        np.testing.assert_allclose(self.spin_1.get_use_spin(), self.use_spin_1)
        np.testing.assert_allclose(self.spin_2.get_use_spin(), self.use_spin_2)
        np.testing.assert_allclose(self.spin_3.get_use_spin(), self.use_spin_3)

    def test_mask(self) -> None:
        np.testing.assert_allclose(
            self.spin_1.get_virtual_scale_mask(), self.expect_virtual_scale_mask_1
        )
        np.testing.assert_allclose(
            self.spin_2.get_virtual_scale_mask(), self.expect_virtual_scale_mask_2
        )
        np.testing.assert_allclose(
            self.spin_3.get_virtual_scale_mask(), self.expect_virtual_scale_mask_3
        )

    def test_exclude_types(self) -> None:
        self.assertEqual(
            sorted(self.spin_1.get_pair_exclude_types()),
            sorted(self.expect_pair_exclude_types_1),
        )
        self.assertEqual(
            sorted(self.spin_2.get_pair_exclude_types()),
            sorted(self.expect_pair_exclude_types_2),
        )
        self.assertEqual(
            sorted(self.spin_3.get_pair_exclude_types()),
            sorted(self.expect_pair_exclude_types_3),
        )

    def test_virtual_scale_consistence(self) -> None:
        np.testing.assert_allclose(
            self.spin_3.get_virtual_scale(), self.spin_3_float.get_virtual_scale()
        )
        np.testing.assert_allclose(
            self.spin_3.get_virtual_scale_mask(), self.spin_3_nspin.get_virtual_scale()
        )
        np.testing.assert_allclose(
            self.spin_3.get_virtual_scale_mask(),
            self.spin_3_float.get_virtual_scale_mask(),
        )
        np.testing.assert_allclose(
            self.spin_3.get_virtual_scale_mask(),
            self.spin_3_nspin.get_virtual_scale_mask(),
        )
        self.assertEqual(
            self.spin_3.get_pair_exclude_types(),
            self.spin_3_float.get_pair_exclude_types(),
        )
