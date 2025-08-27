# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
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


if __name__ == "__main__":
    unittest.main()
