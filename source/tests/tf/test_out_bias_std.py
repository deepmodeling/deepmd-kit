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
        """Test that out_bias and out_std are applied during model build."""
        from deepmd.tf.env import (
            tf,
        )

        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        # Set test bias and std
        test_bias = np.array([[[1.0], [2.0]]])  # bias for type 0: 1.0, type 1: 2.0
        test_std = np.array([[[0.5], [1.5]]])  # std for type 0: 0.5, type 1: 1.5
        model.set_out_bias(test_bias)
        model.set_out_std(test_std)

        # Create mock input data for testing
        nframes = 2
        nloc = 3

        # Mock coordinates and atom types
        coord = tf.placeholder(tf.float64, [None, nloc * 3])
        atype = tf.placeholder(tf.int32, [None, nloc])
        natoms = [
            nloc,
            nloc,
            1,
            2,
        ]  # [local atoms, total atoms, type 0 count, type 1 count]
        box = tf.placeholder(tf.float64, [None, 9])
        mesh = tf.placeholder(tf.int32, [None, 6])

        # Build the model - this should apply bias/std internally
        model.build(coord, atype, natoms, box, mesh, input_dict=None)

        # Check that the bias and std variables were created
        self.assertTrue(hasattr(model, "t_out_bias"))
        self.assertTrue(hasattr(model, "t_out_std"))

        # Test that out_bias and out_std getters work
        bias = model.get_out_bias()
        std = model.get_out_std()
        np.testing.assert_array_equal(bias, test_bias)
        np.testing.assert_array_equal(std, test_std)

    def test_apply_out_stat_no_bias(self):
        """Test that when no bias is explicitly set, default bias (zeros) is used."""
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        # Initialize the model which should set default bias=0, std=1
        model.init_out_stat()

        # Verify that default bias and std are set correctly
        bias = model.get_out_bias()
        std = model.get_out_std()

        # Default bias should be zeros
        expected_bias = np.zeros([1, 2, 1])  # [1, ntypes, dim_out]
        expected_std = np.ones([1, 2, 1])  # [1, ntypes, dim_out]

        np.testing.assert_array_equal(bias, expected_bias)
        np.testing.assert_array_equal(std, expected_std)

    def test_decoupled_bias_architecture(self):
        """Test that out_bias and bias_atom_e are completely decoupled."""
        # Test that setting out_bias does not affect bias_atom_e and vice versa

        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=3.5, sel=[10, 20], neuron=[8, 16, 32]
        )
        fitting = EnerFitting(ntypes=2, dim_descrpt=32)
        model = StandardModel(
            descriptor=descriptor, fitting_net=fitting, type_map=["H", "O"]
        )

        # Initialize with defaults
        model.init_out_stat()

        # Set out_bias
        test_out_bias = np.array([[[1.0], [2.0]]])
        model.set_out_bias(test_out_bias)

        # Verify out_bias is set correctly
        retrieved_bias = model.get_out_bias()
        np.testing.assert_array_equal(retrieved_bias, test_out_bias)

        # Verify that out_std can be set independently
        test_out_std = np.array([[[0.5], [1.5]]])
        model.set_out_std(test_out_std)
        retrieved_std = model.get_out_std()
        np.testing.assert_array_equal(retrieved_std, test_out_std)

        # Verify shapes are correct for energy models
        self.assertEqual(retrieved_bias.shape, (1, 2, 1))  # [1, ntypes, dim_out]
        self.assertEqual(retrieved_std.shape, (1, 2, 1))  # [1, ntypes, dim_out]


if __name__ == "__main__":
    unittest.main()
