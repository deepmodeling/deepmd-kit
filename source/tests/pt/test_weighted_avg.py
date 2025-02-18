# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from deepmd.entrypoints.test import (
    test,
)


class TestDeepPotModel(unittest.TestCase):
    @patch("deepmd.entrypoints.test.DeepEval")
    @patch("deepmd.entrypoints.test.DeepmdData")
    @patch("deepmd.entrypoints.test.test_ener")
    @patch("deepmd.entrypoints.test.weighted_average")
    @patch("builtins.open")
    def test_deep_pot(
        self,
        mock_open,
        mock_weighted_avg,
        mock_test_ener,
        mock_deepmd_data,
        mock_deep_eval,
    ):
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "mock_system_1\nmock_system_2"
        )

        mock_deep_eval_instance = MagicMock()
        mock_deep_eval.return_value = mock_deep_eval_instance
        mock_deep_eval_instance.get_type_map.return_value = "mock_type_map"

        mock_deepmd_data_instance = MagicMock()
        mock_deepmd_data.return_value = mock_deepmd_data_instance
        base_data = [
            {
                "mae_e": (2.0, 5),
                "mae_ea": (1.5, 5),
                "rmse_e": (2.5, 5),
                "rmse_ea": (2.0, 5),
                "mae_f": (0.3, 15),
                "rmse_f": (0.4, 15),
                "mae_v": (1.2, 5),
                "rmse_v": (1.5, 5),
                "mae_va": (0.8, 5),
                "rmse_va": (1.0, 5),
            },
            {
                "mae_e": (3.0, 10),
                "mae_ea": (2.5, 10),
                "rmse_e": (3.5, 10),
                "rmse_ea": (3.0, 10),
                "mae_f": (0.5, 30),
                "rmse_f": (0.6, 30),
                "mae_v": (2.0, 10),
                "rmse_v": (2.5, 10),
                "mae_va": (1.5, 10),
                "rmse_va": (2.0, 10),
            },
            {
                "mae_e": (4.0, 15),
                "mae_ea": (3.5, 15),
                "rmse_e": (4.5, 15),
                "rmse_ea": (4.0, 15),
                "mae_f": (0.7, 45),
                "rmse_f": (0.8, 45),
                "mae_v": (3.0, 15),
                "rmse_v": (3.5, 15),
                "mae_va": (2.5, 15),
                "rmse_va": (3.0, 15),
            },
        ]

        mock_test_ener.return_value = (
            base_data[0],
            1,
            1,
            1,
        )

        test(
            model="mock_model_path",
            system="mock_system_path",
            datafile="mock_datafile.txt",
            numb_test=10,
            rand_seed=None,
            shuffle_test=True,
            detail_file="mock_detail.txt",
            atomic=True,
        )

        mock_deep_eval.assert_called_once_with("mock_model_path", head=None)
        mock_deepmd_data.assert_called_once_with(
            "mock_system_path",
            set_prefix="set",
            shuffle_test=True,
            type_map="mock_type_map",
            sort_atoms=False,
        )
        mock_test_ener.assert_called_once()
        mock_weighted_avg.assert_called_once()

        mock_open.assert_called_once_with("mock_datafile.txt", "r")

        self.assertEqual(mock_weighted_avg.return_value["mae_e"], 0.7)
        self.assertEqual(mock_weighted_avg.return_value["rmse_e"], 0.4)


if __name__ == "__main__":
    unittest.main()
