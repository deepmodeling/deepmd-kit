# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for model summary display functions."""

import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

import torch


class TestLogModelSummary(unittest.TestCase):
    """Test _log_model_summary method behavior."""

    def _create_mock_trainer(self, multi_task: bool = False):
        """Create a mock Trainer instance for testing."""
        trainer = MagicMock()
        trainer.multi_task = multi_task
        trainer.rank = 0
        return trainer

    def _create_mock_model_with_descriptor(self, desc_type: str):
        """Create a mock model with get_descriptor method."""
        mock_descriptor = MagicMock()
        mock_descriptor.serialize.return_value = {"type": desc_type}

        mock_model = MagicMock()
        mock_model.get_descriptor.return_value = mock_descriptor
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.randn(10, 5, device="cpu"))]
        )
        return mock_model

    def _create_mock_zbl_model(self, desc_type: str):
        """Create a mock ZBL model using serialize() API."""
        # Use spec as list to only allow specific attributes (no get_descriptor)
        mock_model = MagicMock()
        # Delete get_descriptor so hasattr returns False
        del mock_model.get_descriptor
        mock_model.serialize.return_value = {
            "type": "zbl",
            "models": [
                {"descriptor": {"type": desc_type}},
                {"type": "pairtab"},
            ],
        }
        mock_model.parameters.return_value = iter(
            [torch.nn.Parameter(torch.randn(10, 5, device="cpu"))]
        )
        return mock_model

    @patch("deepmd.pt.train.training.log")
    def test_standard_model_log_output(self, mock_log):
        """Test log output for standard models."""
        from deepmd.pt.train.training import (
            Trainer,
        )

        trainer = self._create_mock_trainer(multi_task=False)
        trainer.model = self._create_mock_model_with_descriptor("se_e2_a")

        # Call the actual method
        Trainer._log_model_summary(trainer)

        # Verify log.info was called with expected descriptor type
        calls = [str(call) for call in mock_log.info.call_args_list]
        self.assertTrue(any("SE_E2_A" in call for call in calls))
        self.assertTrue(any("Model Params" in call for call in calls))

    @patch("deepmd.pt.train.training.log")
    def test_zbl_model_log_output(self, mock_log):
        """Test log output for ZBL models."""
        from deepmd.pt.train.training import (
            Trainer,
        )

        trainer = self._create_mock_trainer(multi_task=False)
        trainer.model = self._create_mock_zbl_model("dpa1")

        # Call the actual method
        Trainer._log_model_summary(trainer)

        # Verify log.info was called with expected descriptor type
        calls = [str(call) for call in mock_log.info.call_args_list]
        self.assertTrue(any("DPA1 (with ZBL)" in call for call in calls))

    @patch("deepmd.pt.train.training.log")
    def test_multi_task_log_output(self, mock_log):
        """Test log output for multi-task models."""
        from deepmd.pt.train.training import (
            Trainer,
        )

        trainer = self._create_mock_trainer(multi_task=True)
        trainer.model_keys = ["task1", "task2"]
        trainer.model = {
            "task1": self._create_mock_model_with_descriptor("dpa2"),
            "task2": self._create_mock_model_with_descriptor("se_atten"),
        }

        # Call the actual method
        Trainer._log_model_summary(trainer)

        # Verify log.info was called for each task
        calls = [str(call) for call in mock_log.info.call_args_list]
        self.assertTrue(any("task1" in call for call in calls))
        self.assertTrue(any("task2" in call for call in calls))
        self.assertTrue(any("DPA2" in call for call in calls))
        self.assertTrue(any("SE_ATTEN" in call for call in calls))

    @patch("deepmd.pt.train.training.log")
    def test_unknown_model_structure(self, mock_log):
        """Test handling of unknown model structure."""
        from deepmd.pt.train.training import (
            Trainer,
        )

        trainer = self._create_mock_trainer(multi_task=False)
        # Model without get_descriptor and without serialize returning valid type
        mock_model = MagicMock()
        del mock_model.get_descriptor
        mock_model.serialize.return_value = {"other_key": "value"}
        mock_model.parameters.return_value = iter([])
        trainer.model = mock_model

        # Call the actual method
        Trainer._log_model_summary(trainer)

        # Verify "UNKNOWN" appears in output
        calls = [str(call) for call in mock_log.info.call_args_list]
        self.assertTrue(any("UNKNOWN" in call for call in calls))

    @patch("deepmd.pt.train.training.log")
    def test_none_descriptor(self, mock_log):
        """Test handling when get_descriptor returns None."""
        from deepmd.pt.train.training import (
            Trainer,
        )

        trainer = self._create_mock_trainer(multi_task=False)
        mock_model = MagicMock()
        mock_model.get_descriptor.return_value = None
        mock_model.serialize.return_value = {"other_key": "value"}
        mock_model.parameters.return_value = iter([])
        trainer.model = mock_model

        # Call the actual method - should not raise AttributeError
        Trainer._log_model_summary(trainer)

        # Verify "UNKNOWN" appears in output
        calls = [str(call) for call in mock_log.info.call_args_list]
        self.assertTrue(any("UNKNOWN" in call for call in calls))


class TestCountParameters(unittest.TestCase):
    """Test parameter counting behavior through _log_model_summary."""

    @patch("deepmd.pt.train.training.log")
    def test_parameter_count_in_log(self, mock_log):
        """Test that parameter count is correctly logged."""
        from deepmd.pt.train.training import (
            Trainer,
        )

        trainer = MagicMock()
        trainer.multi_task = False
        trainer.rank = 0

        # Create model with known parameter count
        real_model = torch.nn.Linear(10, 5, device="cpu")  # 10*5 + 5 = 55 parameters

        # Add mock methods
        mock_descriptor = MagicMock()
        mock_descriptor.serialize.return_value = {"type": "test"}
        real_model.get_descriptor = MagicMock(return_value=mock_descriptor)

        trainer.model = real_model

        # Call the actual method
        Trainer._log_model_summary(trainer)

        # Verify parameter count is logged (55 params = 0.000055 M)
        calls = [str(call) for call in mock_log.info.call_args_list]
        self.assertTrue(any("0.000 M" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
