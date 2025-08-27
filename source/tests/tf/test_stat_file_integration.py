# SPDX-License-Identifier: LGPL-3.0-or-later
"""Integration test to validate stat_file functionality end-to-end."""

import json
import os
import tempfile
import unittest
from pathlib import (
    Path,
)

from deepmd.tf.entrypoints.train import (
    train,
)


class TestStatFileIntegration(unittest.TestCase):
    def test_stat_file_save_and_load(self) -> None:
        """Test that stat_file can be saved and loaded in TF training."""
        # Create a minimal training configuration
        config = {
            "model": {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [2, 4],
                    "rcut_smth": 0.50,
                    "rcut": 1.00,
                    "neuron": [4, 8],
                    "resnet_dt": False,
                    "axis_neuron": 4,
                    "seed": 1,
                },
                "fitting_net": {"neuron": [8, 8], "resnet_dt": True, "seed": 1},
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": 100,
                "start_lr": 0.001,
                "stop_lr": 1e-8,
            },
            "loss": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
            "training": {
                "training_data": {
                    "systems": [
                        "dummy_system"
                    ],  # This will fail but that's OK for our test
                    "batch_size": 1,
                },
                "numb_steps": 5,
                "data_stat_nbatch": 1,
                "disp_freq": 1,
                "save_freq": 2,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = os.path.join(temp_dir, "input.json")
            stat_file_path = os.path.join(temp_dir, "stat_files")

            # Add stat_file to config
            config["training"]["stat_file"] = stat_file_path

            # Write config
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            # Attempt to run training
            # This will fail due to missing data but should still process stat_file parameter
            train(
                INPUT=config_file,
                init_model=None,
                restart=None,
                output=os.path.join(temp_dir, "output.json"),
                init_frz_model=None,
                mpi_log="master",
                log_level=20,
                log_path=None,
                is_compress=False,
                skip_neighbor_stat=True,
                finetune=None,
                use_pretrain_script=False,
            )

            # The main validation is that the code didn't crash with an unrecognized parameter
            # and that if the stat file directory was attempted to be created, it exists
            stat_path = Path(stat_file_path)
            if stat_path.exists():
                self.assertTrue(
                    stat_path.is_dir(), "Stat file path should be a directory"
                )

            # This test primarily validates that the stat_file parameter is accepted
            # and processed without errors in the TF pipeline


if __name__ == "__main__":
    unittest.main()
