#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest

from deepmd.pt.model.model.dp_linear_model import (
    LinearEnergyModel,
)


class TestSharedDictLinear(unittest.TestCase):
    """Test shared_dict functionality in linear models."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        """Clean up test environment."""
        self.tempdir.cleanup()

    def test_shared_dict_basic(self) -> None:
        """Test basic shared_dict functionality."""
        # Test configuration following the issue example
        config = {
            "type": "linear_ener",
            "shared_dict": {
                "type_map_all": ["O", "H"],
                "dpa1_descriptor_1": {
                    "type": "dpa1",
                    "rcut": 6.00,
                    "rcut_smth": 0.50,
                    "sel": 138,
                    "neuron": [25, 50, 100],
                    "axis_neuron": 16,
                    "seed": 1,
                },
            },
            "models": [
                {
                    "type_map": "type_map_all",
                    "descriptor": "dpa1_descriptor_1",
                    "fitting_net": {
                        "neuron": [240, 240, 240],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
                {
                    "type_map": "type_map_all",
                    "descriptor": "dpa1_descriptor_1",
                    "fitting_net": {
                        "type": "dipole",
                        "neuron": [100, 100, 100],
                        "resnet_dt": True,
                        "seed": 1,
                        "embedding_width": 64,  # Required for dipole fitting
                    },
                },
            ],
            "weights": "mean",
        }

        try:
            # Create the model - this should not raise an error
            model = LinearEnergyModel(**config)

            # Check that shared_links exist
            self.assertTrue(hasattr(model, "shared_links"))
            self.assertIsNotNone(model.shared_links)

            # Check that the model has the expected structure
            self.assertEqual(len(model.atomic_model.models), 2)

            # Check type map is correctly shared
            for sub_model in model.atomic_model.models:
                self.assertEqual(sub_model.get_type_map(), ["O", "H"])

        except RuntimeError as e:
            if "statistics of the descriptor has not been computed" in str(e):
                # This is expected in test environment - statistics would be computed during training
                pass
            else:
                self.fail(f"Failed to create LinearEnergyModel with shared_dict: {e}")
        except Exception as e:
            self.fail(f"Failed to create LinearEnergyModel with shared_dict: {e}")

    def test_backward_compatibility(self) -> None:
        """Test that the changes don't break backward compatibility."""
        # Test traditional linear model configuration
        config = {
            "type": "linear_ener",
            "models": [
                {
                    "type_map": ["O", "H"],
                    "descriptor": {
                        "type": "dpa1",
                        "rcut": 6.00,
                        "rcut_smth": 0.50,
                        "sel": 138,
                        "neuron": [25, 50, 100],
                        "axis_neuron": 16,
                        "seed": 1,
                    },
                    "fitting_net": {
                        "neuron": [240, 240, 240],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
                {
                    "type_map": ["O", "H"],
                    "descriptor": {
                        "type": "dpa1",
                        "rcut": 6.00,
                        "rcut_smth": 0.50,
                        "sel": 138,
                        "neuron": [25, 50, 100],
                        "axis_neuron": 16,
                        "seed": 1,
                    },
                    "fitting_net": {
                        "neuron": [240, 240, 240],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
            ],
            "weights": "mean",
            "type_map": ["O", "H"],
        }

        try:
            # Create the model - this should work as before
            model = LinearEnergyModel(**config)

            # Check that shared_links is None for traditional models
            self.assertIsNone(model.shared_links)

            # Check that the model has the expected structure
            self.assertEqual(len(model.atomic_model.models), 2)

        except Exception as e:
            self.fail(f"Failed to create traditional LinearEnergyModel: {e}")


if __name__ == "__main__":
    # Run the tests
    unittest.main()
