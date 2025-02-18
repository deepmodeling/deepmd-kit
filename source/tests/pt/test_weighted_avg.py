# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from deepmd.entrypoints.test import test  # Import the test function


class TestDeepPotModel(unittest.TestCase):
    @patch("deepmd.entrypoints.test.DeepEval")  # Mock DeepEval class
    @patch("deepmd.entrypoints.test.DeepmdData")  # Mock DeepmdData class
    @patch("deepmd.entrypoints.test.test_ener")  # Mock test_ener function
    @patch("deepmd.entrypoints.test.weighted_average")  # Mock weighted_average function
    @patch("builtins.open")  # Mock the open function to avoid FileNotFoundError
    def test_deep_pot(
        self,
        mock_open,
        mock_weighted_avg,
        mock_test_ener,
        mock_deepmd_data,
        mock_deep_eval,
    ):
        # Mock the file reading behavior to return mock data instead
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "mock_system_1\nmock_system_2"
        )

        # Setup mock return values
        mock_deep_eval_instance = MagicMock()
        mock_deep_eval.return_value = mock_deep_eval_instance
        mock_deep_eval_instance.get_type_map.return_value = "mock_type_map"

        mock_deepmd_data_instance = MagicMock()
        mock_deepmd_data.return_value = mock_deepmd_data_instance

        # Define the base_data to simulate the test_ener output
        base_data = [
            {  # System 1
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
            {  # System 2
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
            {  # System 3
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

        # Simulate err values for each system, adding the (1, 1, 1) triplet
        mock_test_ener.return_value = (
            base_data[0],  # Using the first system's base data
            1,  # find_energy
            1,  # find_force
            1,  # find_virial
        )

        # Call the function with mock data
        test(
            model="mock_model_path",
            system="mock_system_path",
            datafile="mock_datafile.txt",  # Still passing mock file name
            numb_test=10,
            rand_seed=None,
            shuffle_test=True,
            detail_file="mock_detail.txt",
            atomic=True,
        )

        # Check if mocks are called as expected
        mock_deep_eval.assert_called_once_with("mock_model_path", head=None)
        mock_deepmd_data.assert_called_once_with(
            "mock_system_path",
            set_prefix="set",
            shuffle_test=True,
            type_map="mock_type_map",
            sort_atoms=False,
        )
        mock_test_ener.assert_called_once()  # Check if test_ener was called for DeepPot
        mock_weighted_avg.assert_called_once()

        # Check if the file was opened (mocked)
        mock_open.assert_called_once_with("mock_datafile.txt", "r")

        # Check results
        self.assertEqual(mock_weighted_avg.return_value["mae_e"], 0.7)
        self.assertEqual(mock_weighted_avg.return_value["rmse_e"], 0.4)


if __name__ == "__main__":
    unittest.main()
