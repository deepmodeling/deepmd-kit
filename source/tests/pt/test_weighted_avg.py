# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.entrypoints.test import (
    ener_err,
    ener_err_ops,
)



class TestEnerErrFunctions(unittest.TestCase):
    def test_test_ener_err(self):
        test_ener_err_params = [
            (
                1,
                0,
                0,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                0,
                1,
                0,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                0,
                0,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                1,
                1,
                0,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                1,
                0,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                0,
                1,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                1,
                1,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_f",
                    "rmse_f",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
        ]
        expected_keys = [
            ["mae_e", "mae_ea", "rmse_e", "rmse_ea"],
            ["mae_f", "rmse_f"],
            ["mae_v", "mae_va", "rmse_v", "rmse_va"],
            ["mae_e", "mae_ea", "rmse_e", "rmse_ea", "mae_f", "rmse_f"],
            [
                "mae_e",
                "mae_ea",
                "rmse_e",
                "rmse_ea",
                "mae_v",
                "mae_va",
                "rmse_v",
                "rmse_va",
            ],
            ["mae_f", "rmse_f", "mae_v", "mae_va", "rmse_v", "rmse_va"],
            [
                "mae_e",
                "mae_ea",
                "rmse_e",
                "rmse_ea",
                "mae_f",
                "rmse_f",
                "mae_v",
                "mae_va",
                "rmse_v",
                "rmse_va",
            ],
        ]

        for (find_energy, find_force, find_virial, _), expected_key in zip(
            test_ener_err_params, expected_keys
        ):
            with self.subTest(
                find_energy=find_energy, find_force=find_force, find_virial=find_virial
            ):
                energy = np.array([1.0] * 10)
                force = np.array([0.1] * 10)
                virial = np.array([0.2] * 10)
                mae_e, mae_ea, mae_f, mae_v, mae_va = 0.1, 0.1, 0.1, 0.1, 0.1
                rmse_e, rmse_ea, rmse_f, rmse_v, rmse_va = 0.2, 0.2, 0.2, 0.2, 0.2

                err = ener_err(
                    find_energy,
                    find_force,
                    find_virial,
                    energy,
                    force,
                    virial,
                    mae_e,
                    mae_ea,
                    mae_f,
                    mae_v,
                    mae_va,
                    rmse_e,
                    rmse_ea,
                    rmse_f,
                    rmse_v,
                    rmse_va,
                )

                self.assertCountEqual(list(err.keys()), expected_key)

    def test_test_ener_err_ops(self):
        test_ener_err_ops_params = [
            (
                1,
                0,
                0,
                0,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                0,
                1,
                1,
                0,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                0,
                0,
                0,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                1,
                1,
                1,
                0,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                1,
                0,
                0,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                0,
                1,
                1,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
            (
                1,
                1,
                1,
                1,
                [
                    "mae_e",
                    "mae_ea",
                    "rmse_e",
                    "rmse_ea",
                    "mae_fr",
                    "rmse_fr",
                    "mae_fm",
                    "rmse_fm",
                    "mae_v",
                    "mae_va",
                    "rmse_v",
                    "rmse_va",
                ],
            ),
        ]
        expected_keys = [
            ["mae_e", "mae_ea", "rmse_e", "rmse_ea"],
            ["mae_fr", "rmse_fr", "mae_fm", "rmse_fm"],
            ["mae_v", "mae_va", "rmse_v", "rmse_va"],
            [
                "mae_e",
                "mae_ea",
                "rmse_e",
                "rmse_ea",
                "mae_fr",
                "rmse_fr",
                "mae_fm",
                "rmse_fm",
            ],
            [
                "mae_e",
                "mae_ea",
                "rmse_e",
                "rmse_ea",
                "mae_v",
                "mae_va",
                "rmse_v",
                "rmse_va",
            ],
            [
                "mae_fr",
                "rmse_fr",
                "mae_fm",
                "rmse_fm",
                "mae_v",
                "mae_va",
                "rmse_v",
                "rmse_va",
            ],
            [
                "mae_e",
                "mae_ea",
                "rmse_e",
                "rmse_ea",
                "mae_fr",
                "rmse_fr",
                "mae_fm",
                "rmse_fm",
                "mae_v",
                "mae_va",
                "rmse_v",
                "rmse_va",
            ],
        ]

        for (
            find_energy,
            find_force_r,
            find_force_m,
            find_virial,
            _,
        ), expected_key in zip(test_ener_err_ops_params, expected_keys):
            with self.subTest(
                find_energy=find_energy,
                find_force_r=find_force_r,
                find_force_m=find_force_m,
                find_virial=find_virial,
            ):
                energy = np.array([1.0] * 10)
                force_r = np.array([0.1] * 10)
                force_m = np.array([0.1] * 10)
                virial = np.array([0.2] * 10)
                mae_e, mae_ea, mae_fr, mae_fm, mae_v, mae_va = (
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                )
                rmse_e, rmse_ea, rmse_fr, rmse_fm, rmse_v, rmse_va = (
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                    0.2,
                )
                err = ener_err_ops(
                    find_energy,
                    find_force_r,
                    find_force_m,
                    find_virial,
                    energy,
                    force_r,
                    force_m,
                    virial,
                    mae_e,
                    mae_ea,
                    mae_fr,
                    mae_fm,
                    mae_v,
                    mae_va,
                    rmse_e,
                    rmse_ea,
                    rmse_fr,
                    rmse_fm,
                    rmse_v,
                    rmse_va,
                )

                self.assertCountEqual(list(err.keys()), expected_key)


if __name__ == "__main__":
    unittest.main()
