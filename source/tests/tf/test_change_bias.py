# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import tempfile
import unittest
from pathlib import (
    Path,
)

from deepmd.tf.entrypoints.change_bias import (
    change_bias,
)
from deepmd.tf.train.run_options import (
    RunOptions,
)
from deepmd.tf.train.trainer import (
    DPTrainer,
)
from deepmd.tf.utils.argcheck import (
    normalize,
)
from deepmd.tf.utils.compat import (
    update_deepmd_input,
)

from .common import (
    j_loader,
    run_dp,
    tests_path,
)


class TestChangeBias(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_change_bias_frozen_model_partial_support(self):
        """Test that frozen model support has limitations but provides helpful error."""
        fake_pb = self.temp_path / "model.pb"
        fake_pb.write_text("fake model content")

        # Without bias_value, should suggest using bias_value or checkpoint
        with self.assertRaises(NotImplementedError) as cm:
            change_bias(
                INPUT=str(fake_pb),
                mode="change",
                system=".",
            )

        self.assertIn(
            "Data-based bias changing for frozen models is not yet implemented",
            str(cm.exception),
        )
        self.assertIn("bias-value option", str(cm.exception))

        # With bias_value, should provide implementation guidance
        with self.assertRaises(NotImplementedError) as cm:
            change_bias(
                INPUT=str(fake_pb),
                mode="change",
                bias_value=[1.0, 2.0],
                system=".",
            )

        self.assertIn(
            "Bias modification for frozen models (.pb) is not yet fully implemented",
            str(cm.exception),
        )
        self.assertIn("checkpoint_dir", str(cm.exception))

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

        self.assertIn(
            "checkpoint file or frozen model file (.pb)",
            str(cm.exception),
        )

    def test_change_bias_no_checkpoint_in_directory(self):
        """Test that checkpoint files need proper checkpoint structure."""
        fake_ckpt = self.temp_path / "model.ckpt"
        fake_ckpt.write_text("fake checkpoint content")

        # Create a fake data system for the test
        fake_data_dir = self.temp_path / "fake_data"
        fake_data_dir.mkdir()
        fake_set_dir = fake_data_dir / "set.000"
        fake_set_dir.mkdir()

        with self.assertRaises(RuntimeError) as cm:
            change_bias(
                INPUT=str(fake_ckpt),
                mode="change",
                system=str(fake_data_dir),
            )

        self.assertIn("No valid checkpoint found", str(cm.exception))

    def test_change_bias_user_defined_requires_real_model(self):
        """Test that user-defined bias requires a real model with proper structure."""
        fake_ckpt_dir = self.temp_path / "fake_checkpoint"
        fake_ckpt_dir.mkdir()
        fake_ckpt = fake_ckpt_dir / "model.ckpt"
        fake_ckpt.write_text("fake checkpoint content")
        (fake_ckpt_dir / "checkpoint").write_text("fake checkpoint")
        # Create a minimal but complete input.json
        minimal_config = {
            "model": {"type_map": ["H", "O"]},
            "training": {"systems": ["."], "validation_data": {"systems": ["."]}},
        }

        (fake_ckpt_dir / "input.json").write_text(json.dumps(minimal_config))

        # Should fail because there's no real model structure, but with different error
        with self.assertRaises((RuntimeError, FileNotFoundError, Exception)) as cm:
            change_bias(
                INPUT=str(fake_ckpt),
                mode="change",
                bias_value=[1.0, 2.0],
                system=".",
            )

        # The error should be about model loading, not about NotImplementedError
        self.assertNotIn("not yet implemented", str(cm.exception))

    def test_change_bias_with_real_model(self):
        """Test change_bias with a real trained model and verify output."""
        # Create temporary directories for training and output
        train_dir = self.temp_path / "train"
        train_dir.mkdir()
        checkpoint_dir = train_dir / "checkpoint"
        output_file = self.temp_path / "output_model.pb"

        # Use existing test data and configuration
        data_dir = tests_path / "init_frz_model" / "data"
        config_file = tests_path / "init_frz_model" / "input.json"

        # Load and modify configuration for quick training
        jdata = j_loader(str(config_file))
        jdata["training"]["training_data"]["systems"] = [str(data_dir)]
        jdata["training"]["validation_data"]["systems"] = [str(data_dir)]
        jdata["training"]["numb_steps"] = 2  # Minimal training for testing
        jdata["training"]["save_freq"] = 1
        jdata["training"]["save_ckpt"] = str(checkpoint_dir / "model.ckpt")

        # Write modified config
        input_json_path = train_dir / "input.json"
        with open(input_json_path, "w") as f:
            json.dump(jdata, f, indent=4)

        # Train the model using run_dp
        ret = run_dp(f"dp train {input_json_path}")
        self.assertEqual(ret, 0, "DP train failed!")

        # Verify checkpoint was created
        self.assertTrue(checkpoint_dir.exists())
        checkpoint_files = list(checkpoint_dir.glob("*"))
        self.assertGreater(len(checkpoint_files), 0, "No checkpoint files created")

        # Find the actual checkpoint file
        checkpoint_file = checkpoint_dir / "model.ckpt"

        # Create a frozen model from the checkpoint for testing
        frozen_model_path = train_dir / "frozen_model.pb"
        ret = run_dp(f"dp freeze -c {checkpoint_dir} -o {frozen_model_path}")
        self.assertEqual(ret, 0, "DP freeze failed!")
        self.assertTrue(frozen_model_path.exists())

        # Test change_bias function - this should provide implementation guidance for frozen models
        with self.assertRaises(NotImplementedError) as cm:
            change_bias(
                INPUT=str(frozen_model_path),
                mode="change",
                system=str(data_dir),
                output=str(output_file),
            )
        self.assertIn(
            "Data-based bias changing for frozen models is not yet implemented",
            str(cm.exception),
        )

        # Now test change_bias on the real checkpoint file (this is the real test)
        change_bias(
            INPUT=str(checkpoint_file),
            mode="change",
            system=str(data_dir),
            output=str(output_file),
        )

        # Verify that output model file was created
        self.assertTrue(output_file.exists())
        self.assertTrue(output_file.stat().st_size > 0, "Output model file is empty")

        # Load original model to verify structure
        original_run_opt = RunOptions(init_model=str(checkpoint_dir), log_level=20)

        # Load the configuration again for creating trainers
        jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
        jdata = normalize(jdata)

        original_trainer = DPTrainer(jdata, run_opt=original_run_opt)

        # Verify original model loads successfully
        self.assertIsNotNone(original_trainer.model)

        # Verify the original model has the expected structure
        original_type_map = original_trainer.model.get_type_map()
        self.assertGreater(len(original_type_map), 0, "Model should have a type_map")

        # Clean up training artifacts
        for artifact in ["lcurve.out", "input_v2_compat.json"]:
            if os.path.exists(artifact):
                os.remove(artifact)


if __name__ == "__main__":
    unittest.main()
