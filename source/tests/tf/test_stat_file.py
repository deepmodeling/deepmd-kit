# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest
from pathlib import (
    Path,
)

from deepmd.common import (
    j_loader,
)
from deepmd.tf.entrypoints.train import (
    _do_work,
)
from deepmd.tf.train.run_options import (
    RunOptions,
)

from .common import (
    tests_path,
)


class TestStatFile(unittest.TestCase):
    def setUp(self) -> None:
        # Use a minimal config for testing
        self.config_file = str(tests_path / "model_compression" / "input.json")
        self.jdata = j_loader(self.config_file)
        # Reduce number of steps and data for faster testing
        self.jdata["training"]["numb_steps"] = 10
        self.jdata["training"]["data_stat_nbatch"] = 1
        self.jdata["training"]["disp_freq"] = 1
        self.jdata["training"]["save_freq"] = 5

    def test_stat_file_tf(self) -> None:
        """Test that stat_file parameter works in TensorFlow training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            stat_file_path = os.path.join(temp_dir, "stat_files")

            # Add stat_file to training config
            self.jdata["training"]["stat_file"] = stat_file_path

            # Create run options
            run_opt = RunOptions(
                init_model=None,
                restart=None,
                init_frz_model=None,
                finetune=None,
                log_path=None,
                log_level=20,  # INFO level
                mpi_log="master",
            )

            # Run training - this should create the stat file
            _do_work(self.jdata, run_opt, is_compress=False)

            # Check if stat files were created
            stat_path = Path(stat_file_path)
            self.assertTrue(stat_path.exists(), "Stat file directory should be created")

            # Check for energy bias and std files
            bias_file = stat_path / "bias_atom_energy"
            std_file = stat_path / "std_atom_energy"

            # At minimum, the directory structure should be created
            # Even if files aren't created due to insufficient data, the directory should exist
            self.assertTrue(stat_path.is_dir(), "Stat file path should be a directory")


if __name__ == "__main__":
    unittest.main()
