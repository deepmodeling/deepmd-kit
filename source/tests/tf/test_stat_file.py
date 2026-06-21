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
from deepmd.tf.descriptor.stat import (
    load_or_compute_se_input_stats,
)
from deepmd.tf.entrypoints.train import (
    _do_work,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.train.run_options import (
    RunOptions,
)
from deepmd.tf.utils.argcheck import (
    normalize,
)
from deepmd.tf.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.path import (
    DPPath,
)

from .common import (
    tests_path,
)


class _FakeSeADescriptor:
    rcut_smth = 0.5

    def get_sel(self) -> list[int]:
        return [2, 4]

    def get_ntypes(self) -> int:
        return 2

    def get_rcut(self) -> float:
        return 4.0


class _FakeSeRDescriptor:
    rcut_r_smth = 0.5

    def __init__(self) -> None:
        self.sel_r = [2, 4]

    def get_ntypes(self) -> int:
        return 2

    def get_rcut(self) -> float:
        return 4.0


class _FakeSeAttenDescriptor(_FakeSeADescriptor):
    pass


class TestStatFile(unittest.TestCase):
    def setUp(self) -> None:
        tf.reset_default_graph()
        # Use a minimal config for testing
        self.config_file = str(tests_path / "model_compression" / "input.json")
        self.jdata = j_loader(self.config_file)
        # Add missing type field for fitting_net
        self.jdata["model"]["fitting_net"]["type"] = "ener"
        # Move data_stat_nbatch to model section
        self.jdata["model"]["data_stat_nbatch"] = 1
        # Fix the data path to be absolute
        data_path = str(tests_path / "model_compression" / "data")
        self.jdata["training"]["training_data"]["systems"] = [data_path]
        self.jdata["training"]["validation_data"]["systems"] = [data_path]
        # Reduce number of steps and data for faster testing
        self.jdata["training"]["numb_steps"] = 10
        self.jdata["training"]["disp_freq"] = 1
        self.jdata["training"]["save_freq"] = 5
        self.jdata = normalize(update_deepmd_input(self.jdata, warning=False))

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_stat_file_tf(self) -> None:
        """Test that stat_file parameter works in TensorFlow training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            stat_file_path = os.path.join(temp_dir, "stat_files")

            # Add stat_file to training config
            self.jdata["training"]["stat_file"] = stat_file_path
            self.jdata["training"]["disp_file"] = os.path.join(temp_dir, "lcurve.out")
            self.jdata["training"]["save_ckpt"] = os.path.join(temp_dir, "model.ckpt")

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
            self.assertTrue(stat_path.is_dir(), "Stat file path should be a directory")
            type_path = stat_path / "O H"
            self.assertTrue(type_path.is_dir(), "Type-map stat directory should exist")
            self.assertTrue((type_path / "bias_atom_energy").is_file())
            self.assertTrue((type_path / "std_atom_energy").is_file())
            self.assertTrue(
                any(child.is_dir() for child in type_path.iterdir()),
                "Descriptor stat hash directory should be created",
            )


class TestDescriptorStatFile(unittest.TestCase):
    def _assert_load_round_trip(
        self,
        descrpt,
        stat_dict: dict,
        last_dim: int,
        mixed_types: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            stat_path = DPPath(str(Path(temp_dir)), "a")

            saved = load_or_compute_se_input_stats(
                descrpt,
                stat_path,
                last_dim=last_dim,
                compute=lambda: stat_dict,
                mixed_types=mixed_types,
            )

            def fail_if_recomputed() -> dict:
                self.fail("Descriptor statistics should have been loaded from file")

            loaded = load_or_compute_se_input_stats(
                descrpt,
                stat_path,
                last_dim=last_dim,
                compute=fail_if_recomputed,
                mixed_types=mixed_types,
            )

            self.assertEqual(saved, stat_dict)
            self.assertEqual(loaded, stat_dict)

    def test_se_a_descriptor_stats_reload_from_file(self) -> None:
        """Exercise the angular descriptor-stat save/load path."""
        self._assert_load_round_trip(
            _FakeSeADescriptor(),
            {
                "sumr": [[1.0, 2.0]],
                "sumn": [[3.0, 4.0]],
                "sumr2": [[5.0, 6.0]],
                "suma": [[7.0, 8.0]],
                "suma2": [[9.0, 10.0]],
            },
            last_dim=4,
        )

    def test_se_r_descriptor_stats_reload_from_file(self) -> None:
        """Exercise the radial-only descriptor-stat save/load path."""
        self._assert_load_round_trip(
            _FakeSeRDescriptor(),
            {
                "sumr": [[1.0, 2.0]],
                "sumn": [[3.0, 4.0]],
                "sumr2": [[5.0, 6.0]],
            },
            last_dim=1,
        )

    def test_se_atten_descriptor_stats_reload_from_mixed_type_hash(self) -> None:
        """Exercise the mixed-type descriptor-stat hash branch used by se_atten."""
        self._assert_load_round_trip(
            _FakeSeAttenDescriptor(),
            {
                "sumr": [[1.0, 2.0]],
                "sumn": [[3.0, 4.0]],
                "sumr2": [[5.0, 6.0]],
                "suma": [[7.0, 8.0]],
                "suma2": [[9.0, 10.0]],
            },
            last_dim=4,
            mixed_types=True,
        )


if __name__ == "__main__":
    unittest.main()
