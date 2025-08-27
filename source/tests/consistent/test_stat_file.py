# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test consistency of stat file generation between TensorFlow and PyTorch backends."""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from .common import (
    INSTALLED_PT,
    INSTALLED_TF,
)


class TestStatFileConsistency(unittest.TestCase):
    """Test that TensorFlow and PyTorch produce identical stat files."""

    def setUp(self) -> None:
        """Set up test data and configuration."""
        # Use a minimal but realistic configuration
        self.config_base = {
            "model": {
                "type_map": ["O", "H"],
                "data_stat_nbatch": 5,  # Small for testing
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [2, 4],
                    "rcut_smth": 0.50,
                    "rcut": 1.00,
                    "neuron": [4, 8],
                    "resnet_dt": False,
                    "axis_neuron": 4,
                    "seed": 42,
                },
                "fitting_net": {
                    "neuron": [8, 8],
                    "resnet_dt": True,
                    "seed": 42,
                },
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": 10,
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
                    "systems": [],  # Will be filled with test data
                    "batch_size": 1,
                },
                "numb_steps": 1,  # Minimal training to just generate stat files
                "disp_freq": 1,
                "save_freq": 1,
            },
        }

        # Find the test data directory
        examples_path = Path(__file__).parent.parent.parent.parent / "examples"
        self.test_data_path = examples_path / "water" / "data" / "data_0"

        # Skip if test data not available
        if not self.test_data_path.exists():
            self.skipTest("Test data not available")

    def _run_training_with_stat_file(
        self, backend: str, config: dict, temp_dir: str, stat_dir: str
    ) -> None:
        """Run training with specified backend to generate stat files.

        Parameters
        ----------
        backend : str
            Backend to use ('tf' or 'pt')
        config : dict
            Training configuration
        temp_dir : str
            Temporary directory for output
        stat_dir : str
            Directory for stat files
        """
        config_copy = config.copy()
        config_copy["training"]["stat_file"] = stat_dir
        config_copy["training"]["training_data"]["systems"] = [str(self.test_data_path)]

        config_file = os.path.join(temp_dir, f"input_{backend}.json")

        with open(config_file, "w") as f:
            json.dump(config_copy, f, indent=2)

        # Run training with specified backend using subprocess
        env = os.environ.copy()
        cmd = ["python", "-m", "deepmd.main", "train", config_file]
        if backend == "pt":
            cmd = ["python", "-m", "deepmd.main", "--pt", "train", config_file]

        cmd.extend(["--skip-neighbor-stat", "--log-level", "WARNING"])

        result = subprocess.run(
            cmd, cwd=temp_dir, capture_output=True, text=True, env=env
        )

        if result.returncode != 0:
            self.fail(
                f"Training failed for {backend} backend:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

    def _compare_stat_directories(self, tf_stat_dir: str, pt_stat_dir: str) -> None:
        """Compare stat file directories between TensorFlow and PyTorch.

        Parameters
        ----------
        tf_stat_dir : str
            TensorFlow stat file directory
        pt_stat_dir : str
            PyTorch stat file directory
        """
        tf_path = Path(tf_stat_dir)
        pt_path = Path(pt_stat_dir)

        # Both directories should exist
        self.assertTrue(tf_path.exists(), "TensorFlow stat directory should exist")
        self.assertTrue(pt_path.exists(), "PyTorch stat directory should exist")

        # Both should be directories
        self.assertTrue(tf_path.is_dir(), "TensorFlow stat path should be a directory")
        self.assertTrue(pt_path.is_dir(), "PyTorch stat path should be a directory")

        # Get type map subdirectories
        tf_subdirs = sorted([d.name for d in tf_path.iterdir() if d.is_dir()])
        pt_subdirs = sorted([d.name for d in pt_path.iterdir() if d.is_dir()])

        self.assertEqual(
            tf_subdirs, pt_subdirs, "Both backends should create same subdirectories"
        )

        # Compare files in each subdirectory
        for subdir in tf_subdirs:
            tf_subdir = tf_path / subdir
            pt_subdir = pt_path / subdir

            tf_files = sorted([f.name for f in tf_subdir.iterdir() if f.is_file()])
            pt_files = sorted([f.name for f in pt_subdir.iterdir() if f.is_file()])

            self.assertEqual(
                tf_files, pt_files, f"Files in {subdir} should be identical"
            )

            # Compare file contents
            for filename in tf_files:
                tf_file = tf_subdir / filename
                pt_file = pt_subdir / filename

                tf_data = np.loadtxt(tf_file)
                pt_data = np.loadtxt(pt_file)

                self.assertEqual(
                    tf_data.shape,
                    pt_data.shape,
                    f"Shape mismatch in {subdir}/{filename}",
                )

                # Values should be very close (allow for small numerical differences)
                np.testing.assert_allclose(
                    tf_data,
                    pt_data,
                    rtol=1e-4,
                    atol=1e-6,
                    err_msg=f"Values differ in {subdir}/{filename}",
                )

    @unittest.skipUnless(
        INSTALLED_TF and INSTALLED_PT, "TensorFlow and PyTorch required"
    )
    def test_stat_file_consistency_basic(self) -> None:
        """Test basic stat file consistency between TF and PT."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tf_stat_dir = os.path.join(temp_dir, "tf_stat")
            pt_stat_dir = os.path.join(temp_dir, "pt_stat")

            # Run TensorFlow training
            self._run_training_with_stat_file(
                "tf", self.config_base, temp_dir, tf_stat_dir
            )

            # Run PyTorch training
            self._run_training_with_stat_file(
                "pt", self.config_base, temp_dir, pt_stat_dir
            )

            # Compare the generated stat files
            self._compare_stat_directories(tf_stat_dir, pt_stat_dir)

    @unittest.skipUnless(
        INSTALLED_TF and INSTALLED_PT, "TensorFlow and PyTorch required"
    )
    def test_stat_file_consistency_different_batch_sizes(self) -> None:
        """Test stat file consistency with different data_stat_nbatch values."""
        for nbatch in [1, 3, 10]:
            with self.subTest(nbatch=nbatch):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config = self.config_base.copy()
                    config["model"]["data_stat_nbatch"] = nbatch

                    tf_stat_dir = os.path.join(temp_dir, "tf_stat")
                    pt_stat_dir = os.path.join(temp_dir, "pt_stat")

                    # Run TensorFlow training
                    self._run_training_with_stat_file(
                        "tf", config, temp_dir, tf_stat_dir
                    )

                    # Run PyTorch training
                    self._run_training_with_stat_file(
                        "pt", config, temp_dir, pt_stat_dir
                    )

                    # Compare the generated stat files
                    self._compare_stat_directories(tf_stat_dir, pt_stat_dir)

    @unittest.skipUnless(
        INSTALLED_TF and INSTALLED_PT, "TensorFlow and PyTorch required"
    )
    def test_stat_file_consistency_different_seeds(self) -> None:
        """Test stat file consistency with different random seeds."""
        for seed in [1, 42, 123]:
            with self.subTest(seed=seed):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config = self.config_base.copy()
                    config["model"]["descriptor"]["seed"] = seed
                    config["model"]["fitting_net"]["seed"] = seed

                    tf_stat_dir = os.path.join(temp_dir, "tf_stat")
                    pt_stat_dir = os.path.join(temp_dir, "pt_stat")

                    # Run TensorFlow training
                    self._run_training_with_stat_file(
                        "tf", config, temp_dir, tf_stat_dir
                    )

                    # Run PyTorch training
                    self._run_training_with_stat_file(
                        "pt", config, temp_dir, pt_stat_dir
                    )

                    # Compare the generated stat files
                    self._compare_stat_directories(tf_stat_dir, pt_stat_dir)

    @unittest.skipUnless(
        INSTALLED_TF and INSTALLED_PT, "TensorFlow and PyTorch required"
    )
    def test_stat_file_consistency_different_type_maps(self) -> None:
        """Test stat file consistency with different type maps."""
        type_maps = [
            ["O", "H"],
            ["H", "O"],  # Different order
            ["X", "Y"],  # Different names
        ]

        for type_map in type_maps:
            with self.subTest(type_map=type_map):
                with tempfile.TemporaryDirectory() as temp_dir:
                    config = self.config_base.copy()
                    config["model"]["type_map"] = type_map

                    tf_stat_dir = os.path.join(temp_dir, "tf_stat")
                    pt_stat_dir = os.path.join(temp_dir, "pt_stat")

                    # Run TensorFlow training
                    self._run_training_with_stat_file(
                        "tf", config, temp_dir, tf_stat_dir
                    )

                    # Run PyTorch training
                    self._run_training_with_stat_file(
                        "pt", config, temp_dir, pt_stat_dir
                    )

                    # Compare the generated stat files
                    self._compare_stat_directories(tf_stat_dir, pt_stat_dir)

    def tearDown(self) -> None:
        """Clean up any temporary files."""
        # Clean up any leftover files
        for path in ["checkpoint", "lcurve.out", "model.ckpt"]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
