# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    MagicMock,
    patch,
)

from deepmd.tf.entrypoints.change_bias import (
    change_bias,
)


class TestChangeBias(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_change_bias_frozen_model_not_implemented(self):
        """Test that frozen model support raises NotImplementedError."""
        fake_pb = self.temp_path / "model.pb"
        fake_pb.write_text("fake model content")

        with self.assertRaises(NotImplementedError) as cm:
            change_bias(
                INPUT=str(fake_pb),
                mode="change",
                system=".",
            )

        self.assertIn("Bias changing for frozen models", str(cm.exception))
        self.assertIn(".pb/.pbtxt", str(cm.exception))

    def test_change_bias_invalid_model_type(self):
        """Test that invalid model types raise RuntimeError."""
        fake_model = self.temp_path / "model.xyz"
        fake_model.write_text("fake model content")

        with self.assertRaises(RuntimeError) as cm:
            change_bias(
                INPUT=str(fake_model),
                mode="change",
                system=".",
            )

        self.assertIn("checkpoint directory or frozen model file", str(cm.exception))

    def test_change_bias_no_checkpoint_in_directory(self):
        """Test that missing checkpoint in directory raises RuntimeError."""
        fake_dir = self.temp_path / "fake_checkpoint"
        fake_dir.mkdir()

        # Create a fake data system for the test
        fake_data_dir = self.temp_path / "fake_data"
        fake_data_dir.mkdir()
        fake_set_dir = fake_data_dir / "set.000"
        fake_set_dir.mkdir()

        with self.assertRaises(RuntimeError) as cm:
            change_bias(
                INPUT=str(fake_dir),
                mode="change",
                system=str(fake_data_dir),
            )

        self.assertIn("No valid checkpoint found", str(cm.exception))

    def test_change_bias_user_defined_not_implemented(self):
        """Test that user-defined bias raises NotImplementedError."""
        fake_dir = self.temp_path / "fake_checkpoint"
        fake_dir.mkdir()
        (fake_dir / "checkpoint").write_text("fake checkpoint")

        with self.assertRaises(NotImplementedError) as cm:
            change_bias(
                INPUT=str(fake_dir),
                mode="change",
                bias_value=[1.0, 2.0],
                system=".",
            )

        self.assertIn(
            "User-defined bias setting is not yet implemented", str(cm.exception)
        )

    def test_change_bias_successful_execution(self):
        """Test successful bias changing execution path."""
        # Create fake checkpoint directory with required files
        fake_checkpoint_dir = self.temp_path / "checkpoint"
        fake_checkpoint_dir.mkdir()
        (fake_checkpoint_dir / "checkpoint").write_text("fake checkpoint content")
        (fake_checkpoint_dir / "input.json").write_text('{"model": {}}')

        # Create fake data system
        fake_data_dir = self.temp_path / "data_system"
        fake_data_dir.mkdir()
        fake_set_dir = fake_data_dir / "set.000"
        fake_set_dir.mkdir()

        # Import the module properly
        import sys

        change_bias_module = sys.modules["deepmd.tf.entrypoints.change_bias"]

        with (
            patch.object(
                change_bias_module, "expand_sys_str", return_value=[str(fake_data_dir)]
            ),
            patch.object(change_bias_module, "j_loader", return_value={"model": {}}),
            patch.object(
                change_bias_module, "update_deepmd_input", return_value={"model": {}}
            ),
            patch.object(change_bias_module, "normalize", return_value={"model": {}}),
            patch.object(change_bias_module, "DeepmdDataSystem") as mock_data_system,
            patch.object(change_bias_module, "DPTrainer") as mock_trainer_class,
            patch.object(change_bias_module, "shutil"),
        ):
            # Mock the data system
            mock_data_instance = MagicMock()
            mock_data_instance.get_type_map.return_value = ["H", "O"]
            mock_data_system.return_value = mock_data_instance

            # Mock the trainer
            mock_trainer_instance = MagicMock()
            mock_model = MagicMock()
            mock_model.get_type_map.return_value = ["H", "O"]
            mock_trainer_instance.model = mock_model
            mock_trainer_instance._change_energy_bias = MagicMock()
            mock_trainer_instance.save_checkpoint = MagicMock()
            mock_trainer_class.return_value = mock_trainer_instance

            # Call change_bias function
            change_bias(
                INPUT=str(fake_checkpoint_dir),
                mode="change",
                system=str(fake_data_dir),
                output=str(self.temp_path / "output"),
            )

            # Verify that the trainer's change_energy_bias was called
            mock_trainer_instance._change_energy_bias.assert_called_once()

    def test_change_bias_with_data_type_map(self):
        """Test bias changing when data system has its own type_map."""
        # Create fake checkpoint directory with required files
        fake_checkpoint_dir = self.temp_path / "checkpoint"
        fake_checkpoint_dir.mkdir()
        (fake_checkpoint_dir / "checkpoint").write_text("fake checkpoint content")
        (fake_checkpoint_dir / "input.json").write_text('{"model": {}}')

        # Create fake data system
        fake_data_dir = self.temp_path / "data_system"
        fake_data_dir.mkdir()
        fake_set_dir = fake_data_dir / "set.000"
        fake_set_dir.mkdir()

        # Import the module properly
        import sys

        change_bias_module = sys.modules["deepmd.tf.entrypoints.change_bias"]

        with (
            patch.object(
                change_bias_module, "expand_sys_str", return_value=[str(fake_data_dir)]
            ),
            patch.object(change_bias_module, "j_loader", return_value={"model": {}}),
            patch.object(
                change_bias_module, "update_deepmd_input", return_value={"model": {}}
            ),
            patch.object(change_bias_module, "normalize", return_value={"model": {}}),
            patch.object(change_bias_module, "DeepmdDataSystem") as mock_data_system,
            patch.object(change_bias_module, "DPTrainer") as mock_trainer_class,
            patch.object(change_bias_module, "shutil"),
        ):
            # Mock the data system with type_map
            mock_data_instance = MagicMock()
            mock_data_instance.get_type_map.return_value = [
                "C",
                "N",
                "O",
            ]  # Data has type_map
            mock_data_system.return_value = mock_data_instance

            # Mock the trainer
            mock_trainer_instance = MagicMock()
            mock_model = MagicMock()
            mock_model.get_type_map.return_value = [
                "H",
                "O",
            ]  # Model has different type_map
            mock_trainer_instance.model = mock_model
            mock_trainer_instance._change_energy_bias = MagicMock()
            mock_trainer_instance.save_checkpoint = MagicMock()
            mock_trainer_class.return_value = mock_trainer_instance

            # Call change_bias function
            change_bias(
                INPUT=str(fake_checkpoint_dir),
                mode="change",
                system=str(fake_data_dir),
            )

            # Verify that data's type_map was used (not model's)
            mock_trainer_instance._change_energy_bias.assert_called_once()
            args, kwargs = mock_trainer_instance._change_energy_bias.call_args
            # The third argument should be the type_map from data
            self.assertEqual(args[2], ["C", "N", "O"])


if __name__ == "__main__":
    unittest.main()
