# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.utils.weight_avg import (
    weighted_average,
)


def test(all_sys):
    err_coll = []
    for sys_data in all_sys:
        err, find_energy, find_force, find_virial = sys_data
        err_part = {}
        if find_energy == 1:
            err_part["mae_e"] = err["mae_e"]
            err_part["mae_ea"] = err["mae_ea"]
            err_part["rmse_e"] = err["rmse_e"]
            err_part["rmse_ea"] = err["rmse_ea"]
        if find_force == 1:
            if "rmse_f" in err:
                err_part["mae_f"] = err["mae_f"]
                err_part["rmse_f"] = err["rmse_f"]
            else:
                err_part["mae_fr"] = err["mae_fr"]
                err_part["rmse_fr"] = err["rmse_fr"]
                err_part["mae_fm"] = err["mae_fm"]
                err_part["rmse_fm"] = err["rmse_fm"]
        if find_virial == 1:
            err_part["mae_v"] = err["mae_v"]
            err_part["rmse_v"] = err["rmse_v"]
        err_coll.append(err_part)
    avg_err = weighted_average(err_coll)
    return avg_err


def test_ori(all_sys):
    err_coll = []
    for sys_data in all_sys:
        err, _, _, _ = sys_data
        err_coll.append(err)
    avg_err = weighted_average(err_coll)
    return avg_err


class TestWeightedAverage(unittest.TestCase):
    def test_case1_energy_only(self):
        all_sys = [
            (
                {
                    "mae_e": (2, 2),
                    "mae_ea": (4, 2),
                    "rmse_e": (3, 2),
                    "rmse_ea": (5, 2),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                0,
                0,
            ),
            (
                {
                    "mae_e": (4, 3),
                    "mae_ea": (6, 3),
                    "rmse_e": (5, 3),
                    "rmse_ea": (7, 3),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                0,
                0,
            ),
            (
                {
                    "mae_e": (6, 5),
                    "mae_ea": (8, 5),
                    "rmse_e": (7, 5),
                    "rmse_ea": (9, 5),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                0,
                0,
            ),
        ]
        expected_mae_e = (2 * 2 + 4 * 3 + 6 * 5) / (2 + 3 + 5)
        expected_rmse_e = np.sqrt((3**2 * 2 + 5**2 * 3 + 7**2 * 5) / (2 + 3 + 5))
        expected_mae_ea = (4 * 2 + 6 * 3 + 8 * 5) / 10
        expected_rmse_ea = np.sqrt((5**2 * 2 + 7**2 * 3 + 9**2 * 5) / 10)

        avg_err = test(all_sys)
        self.assertAlmostEqual(avg_err["mae_e"], expected_mae_e)
        self.assertAlmostEqual(avg_err["rmse_e"], expected_rmse_e)
        self.assertAlmostEqual(avg_err["mae_ea"], expected_mae_ea)
        self.assertAlmostEqual(avg_err["rmse_ea"], expected_rmse_ea)
        self.assertAlmostEqual(avg_err["mae_f"], 0)
        self.assertAlmostEqual(avg_err["mae_v"], 0)

        avg_err_ori = test_ori(all_sys)
        self.assertAlmostEqual(avg_err["mae_e"], avg_err_ori["mae_e"])
        self.assertNotEqual(avg_err["mae_f"], avg_err_ori["rmse_f"])
        self.assertNotEqual(avg_err["mae_v"], avg_err_ori["rmse_v"])

    def test_case2_energy_force(self):
        all_sys = [
            (
                {
                    "mae_e": (2, 2),
                    "mae_ea": (4, 2),
                    "rmse_e": (3, 2),
                    "rmse_ea": (5, 2),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                1,
                0,
            ),
            (
                {
                    "mae_e": (4, 3),
                    "mae_ea": (6, 3),
                    "rmse_e": (5, 3),
                    "rmse_ea": (7, 3),
                    "mae_f": (1, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                1,
                0,
            ),
            (
                {
                    "mae_e": (6, 5),
                    "mae_ea": (8, 5),
                    "rmse_e": (7, 5),
                    "rmse_ea": (9, 5),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                0,
                0,
            ),
        ]

        avg_err = test(all_sys)
        expected_mae_f = (2 * 3 + 1 * 3) / (3 + 3)
        self.assertAlmostEqual(avg_err["mae_f"], expected_mae_f)

        avg_err_ori = test_ori(all_sys)
        self.assertAlmostEqual(avg_err["mae_e"], avg_err_ori["mae_e"])
        self.assertNotEqual(avg_err["mae_f"], avg_err_ori["mae_f"])
        self.assertNotEqual(avg_err["mae_v"], avg_err_ori["mae_v"])

    def test_case3_all_components(self):
        all_sys = [
            (
                {
                    "mae_e": (2, 2),
                    "mae_ea": (4, 2),
                    "rmse_e": (3, 2),
                    "rmse_ea": (5, 2),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                1,
                1,
            ),
            (
                {
                    "mae_e": (4, 3),
                    "mae_ea": (6, 3),
                    "rmse_e": (5, 3),
                    "rmse_ea": (7, 3),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (1, 5),
                    "rmse_v": (2, 3),
                },
                1,
                1,
                1,
            ),
            (
                {
                    "mae_e": (6, 5),
                    "mae_ea": (8, 5),
                    "rmse_e": (7, 5),
                    "rmse_ea": (9, 5),
                    "mae_f": (2, 3),
                    "rmse_f": (1, 3),
                    "mae_v": (3, 5),
                    "rmse_v": (3, 3),
                },
                1,
                1,
                0,
            ),
        ]

        avg_err = test(all_sys)
        expected_mae_v = (3 * 5 + 1 * 5) / (5 + 5)
        self.assertAlmostEqual(avg_err["mae_v"], expected_mae_v)

        avg_err_ori = test_ori(all_sys)
        self.assertAlmostEqual(avg_err["mae_e"], avg_err_ori["mae_e"])
        self.assertAlmostEqual(avg_err["mae_f"], avg_err_ori["mae_f"])
        self.assertNotEqual(avg_err["mae_v"], avg_err_ori["rmse_v"])


if __name__ == "__main__":
    unittest.main()
