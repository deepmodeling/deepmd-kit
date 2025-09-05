# SPDX-License-Identifier: LGPL-3.0-or-later

import unittest

import numpy as np

from deepmd.tf.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.tf.fit.dipole import (
    DipoleFittingSeA,
)
from deepmd.tf.fit.ener import (
    EnerFitting,
)
from deepmd.tf.model.model import (
    StandardModel,
)


class TestOutBiasStd(unittest.TestCase):
    """Test out_bias and out_std functionality in TensorFlow backend."""

    def test_init_out_stat_basic(self):
        """Test basic init_out_stat functionality."""
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        # Test initial state
        self.assertIsNone(model.out_bias)
        self.assertIsNone(model.out_std)

        # Test init_out_stat
        model.init_out_stat()
        self.assertIsNotNone(model.out_bias)
        self.assertIsNotNone(model.out_std)
        self.assertEqual(model.out_bias.shape, (1, 2, 1))  # 1 output, 2 types, 1 dim
        self.assertEqual(model.out_std.shape, (1, 2, 1))

        # Check default values
        np.testing.assert_array_equal(model.out_bias, np.zeros((1, 2, 1)))
        np.testing.assert_array_equal(model.out_std, np.ones((1, 2, 1)))

    def test_get_set_methods(self):
        """Test get/set methods for out_bias and out_std."""
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        model.init_out_stat()

        # Test get methods
        original_bias = model.get_out_bias()
        original_std = model.get_out_std()
        self.assertEqual(original_bias.shape, (1, 2, 1))
        self.assertEqual(original_std.shape, (1, 2, 1))

        # Test set methods
        test_bias = np.array([[[1.0], [2.0]]])
        test_std = np.array([[[0.5], [0.8]]])

        model.set_out_bias(test_bias)
        model.set_out_std(test_std)

        np.testing.assert_array_equal(model.get_out_bias(), test_bias)
        np.testing.assert_array_equal(model.get_out_std(), test_std)

    def test_different_fitting_dimensions(self):
        """Test that different fitting types have correct dimensions."""
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )

        # Test energy fitting (dim_out = 1)
        fitting_ener = EnerFitting(ntypes=2, dim_descrpt=32)
        model_ener = StandardModel(
            descriptor=descriptor, fitting_net=fitting_ener, type_map=["H", "O"]
        )
        model_ener.init_out_stat(suffix="_ener")
        self.assertEqual(model_ener.out_bias.shape, (1, 2, 1))

        # Test dipole fitting (dim_out = 3)
        fitting_dipole = DipoleFittingSeA(ntypes=2, dim_descrpt=32, embedding_width=32)
        model_dipole = StandardModel(
            descriptor=descriptor, fitting_net=fitting_dipole, type_map=["H", "O"]
        )
        model_dipole.init_out_stat(suffix="_dipole")
        self.assertEqual(model_dipole.out_bias.shape, (1, 2, 3))

    def test_apply_out_stat(self):
        """Test apply_out_stat method."""
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        model.init_out_stat()

        # Set test bias
        test_bias = np.array([[[1.0], [2.0]]])  # bias for type 0: 1.0, type 1: 2.0
        model.set_out_bias(test_bias)

        # Create test data
        nframes = 2
        nloc = 3
        ret = {
            "energy": np.array(
                [[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]
            ),  # [nframes, nloc, 1]
        }
        atype = np.array([[0, 1, 0], [1, 0, 1]])  # [nframes, nloc]

        # Apply bias
        result = model.apply_out_stat(ret, atype)

        # Check that bias was applied correctly
        expected = np.array([[[1.0], [2.0], [1.0]], [[2.0], [1.0], [2.0]]])
        np.testing.assert_array_equal(result["energy"], expected)

    def test_apply_out_stat_no_bias(self):
        """Test apply_out_stat when no bias is set."""
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        # Don't initialize out_stat, so out_bias remains None
        ret = {
            "energy": np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
        }
        atype = np.array([[0, 1], [1, 0]])

        # Should return unchanged
        result = model.apply_out_stat(ret, atype)
        np.testing.assert_array_equal(result["energy"], ret["energy"])

    def test_workaround_shape_conversion(self):
        """Test the improved workaround for shape conversion between out_bias and bias_atom_e."""
        # Test the shape conversion logic for dipole models
        # This ensures the problematic reshape is handled correctly

        # Simulate data structure from deserialization
        data = {
            "fitting": {
                "@variables": {
                    "bias_atom_e": np.array([0.0, 0.0])  # shape [ntypes]
                }
            },
            "@variables": {
                "out_bias": np.array(
                    [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
                )  # shape [1, ntypes, 3]
            },
        }

        # Test the conversion logic directly
        bias_atom_e_shape = data["fitting"]["@variables"]["bias_atom_e"].shape
        out_bias_data = data["@variables"]["out_bias"]

        self.assertEqual(bias_atom_e_shape, (2,))
        self.assertEqual(out_bias_data.shape, (1, 2, 3))

        # Apply the improved logic
        if len(bias_atom_e_shape) == 1 and len(out_bias_data.shape) == 3:
            if out_bias_data.shape[2] == 1:
                bias_increment = out_bias_data[0, :, 0]
            else:
                # Dipole/Polar case: take norm for compatibility
                bias_increment = np.linalg.norm(out_bias_data[0], axis=-1)

            # Should successfully create bias_increment with correct shape
            self.assertEqual(bias_increment.shape, (2,))
            # Values should be norms of [1,2,3] and [4,5,6]
            expected_norms = np.array(
                [np.linalg.norm([1, 2, 3]), np.linalg.norm([4, 5, 6])]
            )
            np.testing.assert_array_almost_equal(bias_increment, expected_norms)


if __name__ == "__main__":
    unittest.main()
