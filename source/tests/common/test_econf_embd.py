# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.utils.econf_embd import (
    electronic_configuration_embedding,
    make_econf_embedding,
)

try:
    import mendeleev  # noqa: F401

    has_mendeleev = True
except ImportError:
    has_mendeleev = False


@unittest.skipIf(not has_mendeleev, "does not have mendeleev installed, skip the UTs.")
class TestEConfEmbd(unittest.TestCase):
    def test_fe(self):
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

    def test_fe_flatten(self):
        res = make_econf_embedding(["Fe"], flatten=True)["Fe"]
        # fmt: off
        expected_res = [2,2,2,2,2,2,2,2,2,2,1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # fmt: on
        self.assertEqual(list(res), expected_res)

    def test_dict(self):
        res = electronic_configuration_embedding["Fe"]
        # fmt: off
        expected_res = [2,2,2,2,2,2,2,2,2,2,1,1,1,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # fmt: on
        self.assertEqual(list(res), expected_res)
