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
from deepmd.tf.env import (
    tf,
)

# Get the test data directory
tests_path = Path(__file__).parent.parent.parent.parent / "examples"


class TestStatFileIntegration(unittest.TestCase):
    def setUp(self) -> None:
        tf.reset_default_graph()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_stat_file_path_is_accepted_and_created(self) -> None:
        """Test that TF training accepts training.stat_file and creates its directory."""
        # Create a minimal training configuration
        config = {
            "model": {
                "type_map": ["O", "H"],
                "data_stat_nbatch": 1,
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
                        str(tests_path / "water" / "data" / "data_0")
                    ],  # Use actual test data
                    "batch_size": 1,
                },
                "numb_steps": 2,  # Very short training
                "disp_freq": 1,
                "save_freq": 1,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create config file
            config_file = os.path.join(temp_dir, "input.json")
            stat_file_path = os.path.join(temp_dir, "stat_files")

            # Add stat_file to config
            config["training"]["stat_file"] = stat_file_path
            config["training"]["disp_file"] = os.path.join(temp_dir, "lcurve.out")
            config["training"]["save_ckpt"] = os.path.join(temp_dir, "model.ckpt")

            # Write config
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            # Run a short training and verify stat_file is accepted by the TF pipeline.
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
            # and that the stat file directory was created.
            stat_path = Path(stat_file_path)
            self.assertTrue(stat_path.exists(), "Stat file path should be created")
            self.assertTrue(stat_path.is_dir(), "Stat file path should be a directory")
            type_path = stat_path / "O H"
            self.assertTrue(type_path.is_dir(), "Type-map stat directory should exist")
            self.assertTrue(
                any(child.is_dir() for child in type_path.iterdir()),
                "Descriptor stat hash directory should be created",
            )


if __name__ == "__main__":
    unittest.main()
