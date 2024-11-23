# SPDX-License-Identifier: LGPL-3.0-or-later
import random
import unittest
from unittest.mock import (
    patch,
)

from deepmd.tf.entrypoints.train import (
    update_sel,
)
from deepmd.tf.utils.update_sel import (
    UpdateSel,
)

from ..seed import (
    GLOBAL_SEED,
)


class TestTrain(unittest.TestCase):
    def setUp(self) -> None:
        self.update_sel = UpdateSel()
        self.mock_min_nbor_dist = random.Random(GLOBAL_SEED).random()
        return super().setUp()

    def test_train_parse_auto_sel(self) -> None:
        self.assertTrue(self.update_sel.parse_auto_sel("auto"))
        self.assertTrue(self.update_sel.parse_auto_sel("auto:12"))
        self.assertTrue(self.update_sel.parse_auto_sel("auto:12:13"))
        self.assertFalse(self.update_sel.parse_auto_sel([1, 2]))
        self.assertFalse(self.update_sel.parse_auto_sel("abc:12:13"))

    def test_train_parse_auto_sel_ratio(self) -> None:
        self.assertEqual(self.update_sel.parse_auto_sel_ratio("auto"), 1.1)
        self.assertEqual(self.update_sel.parse_auto_sel_ratio("auto:1.2"), 1.2)
        with self.assertRaises(RuntimeError):
            self.update_sel.parse_auto_sel_ratio("auto:1.2:1.3")
        with self.assertRaises(RuntimeError):
            self.update_sel.parse_auto_sel_ratio("abc")
        with self.assertRaises(RuntimeError):
            self.update_sel.parse_auto_sel_ratio([1, 2, 3])

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_one_sel(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist, [10, 20]
        get_data_mock.return_value = None
        min_nbor_dist, sel = self.update_sel.update_one_sel(None, None, 6, "auto")
        # self.assertEqual(descriptor['sel'], [11,22])
        self.assertEqual(sel, [12, 24])
        self.assertAlmostEqual(min_nbor_dist, self.mock_min_nbor_dist)
        min_nbor_dist, sel = self.update_sel.update_one_sel(None, None, 6, "auto:1.5")
        # self.assertEqual(descriptor['sel'], [15,30])
        self.assertEqual(sel, [16, 32])
        self.assertAlmostEqual(min_nbor_dist, self.mock_min_nbor_dist)

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_hybrid(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist, [10, 20]
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "descriptor": {
                    "type": "hybrid",
                    "list": [
                        {"type": "se_e2_a", "rcut": 6, "sel": "auto"},
                        {"type": "se_e2_a", "rcut": 6, "sel": "auto:1.5"},
                    ],
                }
            },
            "training": {"training_data": {}},
        }
        expected_out = {
            "model": {
                "descriptor": {
                    "type": "hybrid",
                    "list": [
                        {"type": "se_e2_a", "rcut": 6, "sel": [12, 24]},
                        {"type": "se_e2_a", "rcut": 6, "sel": [16, 32]},
                    ],
                }
            },
            "training": {"training_data": {}},
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist, [10, 20]
        get_data_mock.return_value = None
        jdata = {
            "model": {"descriptor": {"type": "se_e2_a", "rcut": 6, "sel": "auto"}},
            "training": {"training_data": {}},
        }
        expected_out = {
            "model": {"descriptor": {"type": "se_e2_a", "rcut": 6, "sel": [12, 24]}},
            "training": {"training_data": {}},
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_atten_auto(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist, [25]
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": "auto",
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        expected_out = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": 28,
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_atten_int(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist, [25]
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": 30,
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        expected_out = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": 30,
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_update_sel_atten_list(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist, [25]
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": 30,
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        expected_out = {
            "model": {
                "descriptor": {
                    "type": "se_atten",
                    "sel": 30,
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    def test_skip_loc_frame(self, get_data_mock) -> None:
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "descriptor": {
                    "type": "loc_frame",
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        expected_out = {
            "model": {
                "descriptor": {
                    "type": "loc_frame",
                    "rcut": 6,
                }
            },
            "training": {"training_data": {}},
        }
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    def test_skip_frozen(self, get_data_mock) -> None:
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "type": "frozen",
            },
            "training": {"training_data": {}},
        }
        expected_out = jdata.copy()
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    def test_skip_linear_frozen(self, get_data_mock) -> None:
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "type": "linear_ener",
                "models": [
                    {"type": "frozen"},
                    {"type": "frozen"},
                    {"type": "frozen"},
                    {"type": "frozen"},
                ],
            },
            "training": {"training_data": {}},
        }
        expected_out = jdata.copy()
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    @patch("deepmd.tf.entrypoints.train.get_data")
    @patch("deepmd.tf.utils.update_sel.UpdateSel.get_min_nbor_dist")
    def test_pairwise_dprc(self, sel_mock, get_data_mock) -> None:
        sel_mock.return_value = self.mock_min_nbor_dist
        get_data_mock.return_value = None
        jdata = {
            "model": {
                "type": "pairwise_dprc",
                "models": [
                    {"type": "frozen"},
                    {"type": "frozen"},
                    {"type": "frozen"},
                    {"type": "frozen"},
                ],
            },
            "training": {"training_data": {}},
        }
        expected_out = jdata.copy()
        jdata = update_sel(jdata)
        self.assertEqual(jdata, expected_out)

    def test_wrap_up_4(self) -> None:
        self.assertEqual(self.update_sel.wrap_up_4(12), 3 * 4)
        self.assertEqual(self.update_sel.wrap_up_4(13), 4 * 4)
        self.assertEqual(self.update_sel.wrap_up_4(14), 4 * 4)
        self.assertEqual(self.update_sel.wrap_up_4(15), 4 * 4)
        self.assertEqual(self.update_sel.wrap_up_4(16), 4 * 4)
        self.assertEqual(self.update_sel.wrap_up_4(17), 5 * 4)
