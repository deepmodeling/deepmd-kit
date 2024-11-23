# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.utils.econf_embd import (
    electronic_configuration_embedding,
    make_econf_embedding,
    normalized_electronic_configuration_embedding,
    transform_to_spin_rep,
)


class TestEConfEmbd(unittest.TestCase):
    def test_fe(self) -> None:
        res = make_econf_embedding(["Fe"], flatten=False)["Fe"]
        expected_res = {
            (1, "s"): [2],
            (2, "s"): [2],
            (2, "p"): [2, 2, 2],
            (3, "s"): [2],
            (3, "p"): [2, 2, 2],
            (3, "d"): [2, 1, 1, 1, 1],
            (4, "s"): [2],
            (4, "p"): [0, 0, 0],
            (4, "d"): [0, 0, 0, 0, 0],
            (4, "f"): [0, 0, 0, 0, 0, 0, 0],
            (5, "s"): [0],
            (5, "p"): [0, 0, 0],
            (5, "d"): [0, 0, 0, 0, 0],
            (5, "f"): [0, 0, 0, 0, 0, 0, 0],
            (6, "s"): [0],
            (6, "p"): [0, 0, 0],
            (6, "d"): [0, 0, 0, 0, 0],
            (7, "s"): [0],
            (7, "p"): [0, 0, 0],
        }
        self.assertDictEqual({kk: list(vv) for kk, vv in res.items()}, expected_res)

    def test_fe_flatten(self) -> None:
        res = make_econf_embedding(["Fe"], flatten=True)["Fe"]
        # fmt: off
        expected_res = [2,2,2,2,2,2,2,2,2,2,1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # fmt: on
        self.assertEqual(list(res), expected_res)

    def test_fe_spin(self) -> None:
        res = make_econf_embedding(["Fe"], flatten=True)
        res = transform_to_spin_rep(res)["Fe"]
        # fmt: off
        expected_res = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        # fmt: on
        self.assertEqual(list(res), expected_res)

    def test_dict(self) -> None:
        res = electronic_configuration_embedding["Fe"]
        # fmt: off
        expected_res = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        # fmt: on
        self.assertEqual(list(res), expected_res)
        res = normalized_electronic_configuration_embedding["Fe"]
        self.assertEqual(
            list(res), [ii / len(expected_res) ** 0.5 for ii in expected_res]
        )
