# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
    map_pair_exclude_types,
)


class TestTypeIndexMap(unittest.TestCase):
    def test_get_index_between_two_maps(self) -> None:
        tm_1 = [
            "Al",
            "F",
            "N",
            "H",
            "S",
            "O",
            "He",
            "C",
            "Li",
            "Na",
            "Be",
            "Mg",
            "Si",
            "B",
            "Ne",
            "P",
        ]  # 16 elements
        tm_2 = [
            "P",
            "Na",
            "Si",
            "Mg",
            "C",
            "O",
            "Be",
            "B",
            "Li",
            "S",
            "Ne",
            "N",
            "H",
            "Al",
            "F",
            "He",
        ]  # 16 elements
        tm_3 = ["O", "H", "Be", "C", "N", "B", "Li"]  # 7 elements

        # self consistence
        old_tm = tm_1
        new_tm = tm_1
        expected_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        expected_has_new = False
        result_map, result_has_new = get_index_between_two_maps(old_tm, new_tm)
        self.assertEqual(len(result_map), len(new_tm))
        self.assertEqual(expected_map, result_map)
        self.assertEqual(expected_has_new, result_has_new)

        # test resort
        old_tm = tm_1
        new_tm = tm_2
        expected_map = [15, 9, 12, 11, 7, 5, 10, 13, 8, 4, 14, 2, 3, 0, 1, 6]
        expected_has_new = False
        result_map, result_has_new = get_index_between_two_maps(old_tm, new_tm)
        self.assertEqual(len(result_map), len(new_tm))
        self.assertEqual(expected_map, result_map)
        self.assertEqual(expected_has_new, result_has_new)

        # test slim
        old_tm = tm_1
        new_tm = tm_3
        expected_map = [5, 3, 10, 7, 2, 13, 8]
        expected_has_new = False
        result_map, result_has_new = get_index_between_two_maps(old_tm, new_tm)
        self.assertEqual(len(result_map), len(new_tm))
        self.assertEqual(expected_map, result_map)
        self.assertEqual(expected_has_new, result_has_new)

        # test extend
        old_tm = tm_3
        new_tm = tm_1
        expected_map = [-16, -15, 4, 1, -12, 0, -10, 3, 6, -7, 2, -5, -4, 5, -2, -1]
        expected_has_new = True
        result_map, result_has_new = get_index_between_two_maps(old_tm, new_tm)
        self.assertEqual(len(result_map), len(new_tm))
        self.assertEqual(expected_map, result_map)
        self.assertEqual(expected_has_new, result_has_new)

    def test_map_exclude_types(self) -> None:
        old_tm = [
            "Al",
            "F",
            "N",
            "H",
            "S",
            "O",
            "He",
            "C",
            "Li",
            "Na",
            "Be",
            "Mg",
            "Si",
            "B",
            "Ne",
            "P",
        ]  # 16 elements
        new_tm = ["O", "H", "Be", "C", "N", "B", "Li"]  # 7 elements
        remap_index, _ = get_index_between_two_maps(old_tm, new_tm)
        remap_index_reverse, _ = get_index_between_two_maps(new_tm, old_tm)
        aem_1 = [0]
        aem_2 = [0, 5]
        aem_3 = [7, 8, 11]
        pem_1 = [(0, 0), (0, 5)]
        pem_2 = [(0, 0), (0, 5), (5, 8)]
        pem_3 = [(0, 0), (0, 5), (8, 7)]

        # test map_atom_exclude_types
        expected_aem_1 = []
        result_aem_1 = map_atom_exclude_types(aem_1, remap_index)
        self.assertEqual(expected_aem_1, result_aem_1)

        expected_aem_2 = [0]
        result_aem_2 = map_atom_exclude_types(aem_2, remap_index)
        self.assertEqual(expected_aem_2, result_aem_2)

        expected_aem_3 = [3, 6]
        result_aem_3 = map_atom_exclude_types(aem_3, remap_index)
        self.assertEqual(expected_aem_3, result_aem_3)

        expected_aem_1_reverse = [5]
        result_aem_1_reverse = map_atom_exclude_types(aem_1, remap_index_reverse)
        self.assertEqual(expected_aem_1_reverse, result_aem_1_reverse)

        # test map_pair_exclude_types
        expected_pem_1 = []
        result_pem_1 = map_pair_exclude_types(pem_1, remap_index)
        self.assertEqual(expected_pem_1, result_pem_1)

        expected_pem_2 = [(0, 6)]
        result_pem_2 = map_pair_exclude_types(pem_2, remap_index)
        self.assertEqual(expected_pem_2, result_pem_2)

        expected_pem_3 = [(6, 3)]
        result_pem_3 = map_pair_exclude_types(pem_3, remap_index)
        self.assertEqual(expected_pem_3, result_pem_3)

        expected_pem_1_reverse = [(5, 5), (5, 13)]
        result_pem_1_reverse = map_pair_exclude_types(pem_1, remap_index_reverse)
        self.assertEqual(expected_pem_1_reverse, result_pem_1_reverse)
