# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test consistency of stat file generation between TensorFlow and PyTorch backends."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import (
    deepcopy,
)
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
                "data_stat_nbatch": 80,  # Cover the whole test set for deterministic stats
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
                "seed": 42,
                "numb_steps": 1,  # Minimal training to just generate stat files
                "disp_freq": 1,
                "save_freq": 1,
            },
        }

        # Find the test data directory
        examples_path = Path(__file__).parent.parent.parent.parent / "examples"
        self.test_data_path = examples_path / "water" / "data" / "data_0"
        self.unequal_frame_data_paths = [
            examples_path / "water" / "data" / "data_0",
            examples_path / "water" / "data" / "data_1",
        ]

        # Skip if test data not available
        if not self.test_data_path.exists():
            self.skipTest("Test data not available")
        if any(not path.exists() for path in self.unequal_frame_data_paths):
            self.skipTest("Unequal-frame test data not available")

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
        config_copy = deepcopy(config)
        config_copy["training"]["stat_file"] = stat_dir
        if not config_copy["training"]["training_data"]["systems"]:
            config_copy["training"]["training_data"]["systems"] = [
                str(self.test_data_path)
            ]

        config_file = os.path.join(temp_dir, f"input_{backend}.json")

        with open(config_file, "w") as f:
            json.dump(config_copy, f, indent=2)

        # Run training with specified backend using subprocess
        env = os.environ.copy()
        dp_cmd = Path(sys.executable).with_name("dp")
        base_cmd = (
            [str(dp_cmd)]
            if dp_cmd.exists()
            else [sys.executable, "-c", "from deepmd.main import main; main()"]
        )
        cmd = [*base_cmd, "train", config_file]
        if backend == "pt":
            cmd = [*base_cmd, "--pt", "train", config_file]

        cmd.extend(["--log-level", "WARNING"])

        result = subprocess.run(
            cmd,
            cwd=temp_dir,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

        if result.returncode != 0:
            self.fail(
                f"Training failed for {backend} backend:\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

    def _compare_stat_directories(
        self,
        tf_stat_dir: str,
        pt_stat_dir: str,
        selected_names: set[str] | None = None,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> None:
        """Compare stat file directories between TensorFlow and PyTorch.

        Parameters
        ----------
        tf_stat_dir : str
            TensorFlow stat file directory
        pt_stat_dir : str
            PyTorch stat file directory
        selected_names : set[str], optional
            Basenames of stat files to compare. When omitted, compare every file.
        rtol : float
            Relative tolerance for numeric stat file comparisons.
        atol : float
            Absolute tolerance for numeric stat file comparisons.
        """
        tf_path = Path(tf_stat_dir)
        pt_path = Path(pt_stat_dir)

        # Both directories should exist
        self.assertTrue(tf_path.exists(), "TensorFlow stat directory should exist")
        self.assertTrue(pt_path.exists(), "PyTorch stat directory should exist")

        # Both should be directories
        self.assertTrue(tf_path.is_dir(), "TensorFlow stat path should be a directory")
        self.assertTrue(pt_path.is_dir(), "PyTorch stat path should be a directory")

        tf_files = sorted(
            ff.relative_to(tf_path) for ff in tf_path.rglob("*") if ff.is_file()
        )
        pt_files = sorted(
            ff.relative_to(pt_path) for ff in pt_path.rglob("*") if ff.is_file()
        )
        if selected_names is not None:
            tf_files = [ff for ff in tf_files if ff.name in selected_names]
            pt_files = [ff for ff in pt_files if ff.name in selected_names]
            self.assertEqual(
                {ff.name for ff in tf_files},
                selected_names,
                "TensorFlow should create the selected stat files",
            )
            self.assertEqual(
                {ff.name for ff in pt_files},
                selected_names,
                "PyTorch should create the selected stat files",
            )

        self.assertEqual(tf_files, pt_files, "Both backends should create same files")
        if selected_names is None:
            self.assertTrue(
                any(len(ff.parts) > 2 for ff in tf_files),
                "Descriptor stat files should be saved under their hash directory",
            )

        for filename in tf_files:
            tf_file = tf_path / filename
            pt_file = pt_path / filename

            tf_data = np.load(tf_file)
            pt_data = np.load(pt_file)

            self.assertEqual(
                tf_data.shape,
                pt_data.shape,
                f"Shape mismatch in {filename}",
            )

            if np.issubdtype(tf_data.dtype, np.number):
                np.testing.assert_allclose(
                    tf_data,
                    pt_data,
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"Values differ in {filename}",
                )
            else:
                np.testing.assert_array_equal(
                    tf_data,
                    pt_data,
                    err_msg=f"Values differ in {filename}",
                )

    @unittest.skipUnless(
        INSTALLED_TF and INSTALLED_PT, "TensorFlow and PyTorch required"
    )
    def test_stat_file_consistency_basic(self) -> None:
        """Test basic stat file consistency between TensorFlow and PyTorch backends."""
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

            # Compare the generated stat files with tight fp64 tolerances.
            self._compare_stat_directories(tf_stat_dir, pt_stat_dir)

    @unittest.skipUnless(
        INSTALLED_TF and INSTALLED_PT, "TensorFlow and PyTorch required"
    )
    def test_output_stat_file_consistency_unequal_frame_systems(self) -> None:
        """Test TF/PT output-stat consistency with unequal frame counts."""
        config = deepcopy(self.config_base)
        config["training"]["training_data"]["systems"] = [
            str(path) for path in self.unequal_frame_data_paths
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            tf_stat_dir = os.path.join(temp_dir, "tf_stat")
            pt_stat_dir = os.path.join(temp_dir, "pt_stat")

            # This case catches the per-system vs per-frame output-bias regression
            # distinction: the legacy TF path weighted each system equally, whereas
            # the shared stat implementation used by stat files weights frames
            # consistently with the PyTorch backend. Descriptor input statistics are
            # collected by backend-specific pipelines, so compare only the shared
            # output-stat file that determines the restored energy bias. The
            # backends may store different auxiliary standard deviations, but
            # the shared stat file must agree on the bias consumed by TF/PT
            # initialization and stat-file reloads.
            self._run_training_with_stat_file("tf", config, temp_dir, tf_stat_dir)
            self._run_training_with_stat_file("pt", config, temp_dir, pt_stat_dir)

            self._compare_stat_directories(
                tf_stat_dir,
                pt_stat_dir,
                selected_names={"bias_atom_energy"},
                # This regression check runs through the full TF/PT CLI paths.
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
