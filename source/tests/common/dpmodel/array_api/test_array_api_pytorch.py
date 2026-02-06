# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test PyTorch support in array_api.py."""

import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from deepmd.dpmodel.array_api import (
    xp_add_at,
    xp_bincount,
    xp_scatter_sum,
)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch is not installed")
class TestPyTorchArrayAPI(unittest.TestCase):
    """Test PyTorch array API support."""

    def test_xp_scatter_sum(self):
        """Test xp_scatter_sum with PyTorch tensors."""
        input_tensor = torch.zeros(3, 5)
        dim = 0
        index = torch.tensor([[0, 1, 2, 0, 0]])
        src = torch.ones(1, 5)
        result = xp_scatter_sum(input_tensor, dim, index, src)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(result, expected))

    def test_xp_add_at(self):
        """Test xp_add_at with PyTorch tensors."""
        x = torch.zeros(5, 3)
        indices = torch.tensor([0, 1, 1, 3])
        values = torch.ones(4, 3)
        result = xp_add_at(x, indices, values)
        expected = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(result, expected))

    def test_xp_bincount(self):
        """Test xp_bincount with PyTorch tensors."""
        x = torch.tensor([0, 1, 1, 3, 2, 1, 7])
        result = xp_bincount(x)
        expected = torch.tensor([1, 3, 1, 1, 0, 0, 0, 1])
        self.assertTrue(torch.equal(result, expected))

    def test_xp_bincount_with_weights(self):
        """Test xp_bincount with weights using PyTorch tensors."""
        x = torch.tensor([0, 1, 1, 3, 2, 1, 7])
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        result = xp_bincount(x, weights=weights)
        expected = torch.tensor([0.1, 1.1, 0.5, 0.4, 0.0, 0.0, 0.0, 0.7])
        self.assertTrue(torch.allclose(result, expected))

    def test_xp_bincount_with_minlength(self):
        """Test xp_bincount with minlength using PyTorch tensors."""
        x = torch.tensor([0, 1, 1, 3])
        result = xp_bincount(x, minlength=10)
        self.assertEqual(result.shape[0], 10)
        expected = torch.tensor([1, 2, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertTrue(torch.equal(result, expected))


if __name__ == "__main__":
    unittest.main()
