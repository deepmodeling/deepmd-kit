# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for model summary display functions."""

import unittest
from unittest.mock import (
    MagicMock,
)

import torch


class TestGetDescriptorType(unittest.TestCase):
    """Test get_descriptor_type helper function."""

    @staticmethod
    def get_descriptor_type(model):
        """Replicate the logic from training.py for testing."""
        # Standard models have get_descriptor method
        if hasattr(model, "get_descriptor"):
            descriptor = model.get_descriptor()
            serialized = descriptor.serialize()
            if isinstance(serialized, dict) and "type" in serialized:
                return serialized["type"].upper()
        # ZBL models: descriptor is in atomic_model.models[0]
        if hasattr(model, "atomic_model") and hasattr(model.atomic_model, "models"):
            models = model.atomic_model.models
            if models:  # Check non-empty
                dp_model = models[0]
                if hasattr(dp_model, "descriptor"):
                    serialized = dp_model.descriptor.serialize()
                    if isinstance(serialized, dict) and "type" in serialized:
                        return serialized["type"].upper() + " (with ZBL)"
        return "UNKNOWN"

    def test_standard_model(self):
        """Test descriptor type detection for standard models."""
        mock_descriptor = MagicMock()
        mock_descriptor.serialize.return_value = {"type": "se_e2_a"}

        mock_model = MagicMock()
        mock_model.get_descriptor.return_value = mock_descriptor

        result = self.get_descriptor_type(mock_model)
        self.assertEqual(result, "SE_E2_A")

    def test_zbl_model(self):
        """Test descriptor type detection for ZBL models."""
        mock_descriptor = MagicMock()
        mock_descriptor.serialize.return_value = {"type": "dpa1"}

        mock_dp_model = MagicMock()
        mock_dp_model.descriptor = mock_descriptor

        mock_atomic_model = MagicMock()
        mock_atomic_model.models = [mock_dp_model]

        mock_model = MagicMock(spec=[])  # No get_descriptor
        mock_model.atomic_model = mock_atomic_model

        result = self.get_descriptor_type(mock_model)
        self.assertEqual(result, "DPA1 (with ZBL)")

    def test_empty_models_list(self):
        """Test handling of empty models list in ZBL model."""
        mock_atomic_model = MagicMock()
        mock_atomic_model.models = []

        mock_model = MagicMock(spec=[])
        mock_model.atomic_model = mock_atomic_model

        result = self.get_descriptor_type(mock_model)
        self.assertEqual(result, "UNKNOWN")

    def test_missing_type_key(self):
        """Test handling of serialize() without 'type' key."""
        mock_descriptor = MagicMock()
        mock_descriptor.serialize.return_value = {"other_key": "value"}

        mock_model = MagicMock()
        mock_model.get_descriptor.return_value = mock_descriptor

        result = self.get_descriptor_type(mock_model)
        self.assertEqual(result, "UNKNOWN")

    def test_serialize_returns_non_dict(self):
        """Test handling of serialize() returning non-dict."""
        mock_descriptor = MagicMock()
        mock_descriptor.serialize.return_value = "not_a_dict"

        mock_model = MagicMock()
        mock_model.get_descriptor.return_value = mock_descriptor

        result = self.get_descriptor_type(mock_model)
        self.assertEqual(result, "UNKNOWN")

    def test_unknown_model_structure(self):
        """Test handling of unknown model structure."""
        mock_model = MagicMock(spec=[])  # No get_descriptor, no atomic_model
        result = self.get_descriptor_type(mock_model)
        self.assertEqual(result, "UNKNOWN")


class TestCountParameters(unittest.TestCase):
    """Test count_parameters helper function."""

    @staticmethod
    def count_parameters(model):
        """Replicate the logic from training.py for testing."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_all_trainable(self):
        """Test counting when all parameters are trainable."""
        with torch.device("cpu"):
            model = torch.nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        result = self.count_parameters(model)
        self.assertEqual(result, 55)

    def test_mixed_trainable(self):
        """Test counting with some frozen parameters."""
        with torch.device("cpu"):
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),  # 55 params
                torch.nn.Linear(5, 3),  # 18 params
            )
        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        result = self.count_parameters(model)
        self.assertEqual(result, 18)  # Only second layer

    def test_all_frozen(self):
        """Test counting when all parameters are frozen."""
        with torch.device("cpu"):
            model = torch.nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False

        result = self.count_parameters(model)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
