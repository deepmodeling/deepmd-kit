# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.dpmodel.array_api import (
    xp_add_at,
    xp_bincount,
    xp_scatter_sum,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)

from .common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
)

if INSTALLED_PT:
    import torch

if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )

if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict as xp


class TestXpScatterSumConsistent(unittest.TestCase):
    """Test xp_scatter_sum consistency across backends."""

    def setUp(self) -> None:
        # Reference using NumPy (via clone and scatter_add simulation)
        self.input_np = np.zeros((3, 5))
        self.dim = 0
        self.index_np = np.array([[0, 1, 2, 0, 0]])
        self.src_np = np.ones((1, 5))
        # Manually compute reference for scatter_sum
        self.ref = self.input_np.copy()
        for i in range(self.index_np.shape[0]):
            for j in range(self.index_np.shape[1]):
                idx = self.index_np[i, j]
                self.ref[idx, j] += self.src_np[i, j]

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        input_pt = torch.from_numpy(self.input_np)
        index_pt = torch.from_numpy(self.index_np).long()
        src_pt = torch.from_numpy(self.src_np)
        result = xp_scatter_sum(input_pt, self.dim, index_pt, src_pt)
        # Verify original tensor is unchanged (non-mutating)
        np.testing.assert_allclose(self.input_np, to_numpy_array(input_pt), atol=1e-10)
        # Verify result matches reference
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        input_jax = jnp.array(self.input_np)
        index_jax = jnp.array(self.index_np)
        src_jax = jnp.array(self.src_np)
        result = xp_scatter_sum(input_jax, self.dim, index_jax, src_jax)
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)


class TestXpAddAtConsistent(unittest.TestCase):
    """Test xp_add_at consistency across backends."""

    def setUp(self) -> None:
        self.x_np = np.zeros((5, 3))
        self.indices_np = np.array([0, 1, 1, 3])
        self.values_np = np.ones((4, 3))
        # Reference using NumPy
        self.ref = self.x_np.copy()
        np.add.at(self.ref, self.indices_np, self.values_np)

    def test_numpy_consistent_with_ref(self) -> None:
        x = self.x_np.copy()
        result = xp_add_at(x, self.indices_np, self.values_np)
        np.testing.assert_allclose(self.ref, result, atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        x_pt = torch.from_numpy(self.x_np)
        indices_pt = torch.from_numpy(self.indices_np).long()
        values_pt = torch.from_numpy(self.values_np)
        result = xp_add_at(x_pt, indices_pt, values_pt)
        # Verify original tensor is unchanged (non-mutating)
        np.testing.assert_allclose(self.x_np, to_numpy_array(x_pt), atol=1e-10)
        # Verify result matches reference
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        x_jax = jnp.array(self.x_np)
        indices_jax = jnp.array(self.indices_np)
        values_jax = jnp.array(self.values_np)
        result = xp_add_at(x_jax, indices_jax, values_jax)
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)

    @unittest.skipUnless(
        INSTALLED_ARRAY_API_STRICT, "array_api_strict is not installed"
    )
    @unittest.skipUnless(
        sys.version_info >= (3, 9), "array_api_strict doesn't support Python<=3.8"
    )
    def test_array_api_strict_consistent_with_ref(self) -> None:
        x_xp = xp.asarray(self.x_np)
        indices_xp = xp.asarray(self.indices_np)
        values_xp = xp.asarray(self.values_np)
        result = xp_add_at(x_xp, indices_xp, values_xp)
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)


class TestXpBincountConsistent(unittest.TestCase):
    """Test xp_bincount consistency across backends."""

    def setUp(self) -> None:
        self.x_np = np.array([0, 1, 1, 3, 2, 1, 7])
        self.ref = np.bincount(self.x_np)

    def test_numpy_consistent_with_ref(self) -> None:
        result = xp_bincount(self.x_np)
        np.testing.assert_equal(self.ref, result)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        x_pt = torch.from_numpy(self.x_np)
        result = xp_bincount(x_pt)
        np.testing.assert_equal(self.ref, to_numpy_array(result))

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        x_jax = jnp.array(self.x_np)
        result = xp_bincount(x_jax)
        np.testing.assert_equal(self.ref, to_numpy_array(result))


class TestXpBincountWithWeightsConsistent(unittest.TestCase):
    """Test xp_bincount with weights consistency across backends."""

    def setUp(self) -> None:
        self.x_np = np.array([0, 1, 1, 3, 2, 1, 7])
        self.weights_np = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.ref = np.bincount(self.x_np, weights=self.weights_np)

    def test_numpy_consistent_with_ref(self) -> None:
        result = xp_bincount(self.x_np, weights=self.weights_np)
        np.testing.assert_allclose(self.ref, result, atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        x_pt = torch.from_numpy(self.x_np)
        weights_pt = torch.from_numpy(self.weights_np)
        result = xp_bincount(x_pt, weights=weights_pt)
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        x_jax = jnp.array(self.x_np)
        weights_jax = jnp.array(self.weights_np)
        result = xp_bincount(x_jax, weights=weights_jax)
        np.testing.assert_allclose(self.ref, to_numpy_array(result), atol=1e-10)


class TestXpBincountWithMinlengthConsistent(unittest.TestCase):
    """Test xp_bincount with minlength consistency across backends."""

    def setUp(self) -> None:
        self.x_np = np.array([0, 1, 1, 3])
        self.minlength = 10
        self.ref = np.bincount(self.x_np, minlength=self.minlength)

    def test_numpy_consistent_with_ref(self) -> None:
        result = xp_bincount(self.x_np, minlength=self.minlength)
        np.testing.assert_equal(self.ref, result)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        x_pt = torch.from_numpy(self.x_np)
        result = xp_bincount(x_pt, minlength=self.minlength)
        np.testing.assert_equal(self.ref, to_numpy_array(result))

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        x_jax = jnp.array(self.x_np)
        result = xp_bincount(x_jax, minlength=self.minlength)
        np.testing.assert_equal(self.ref, to_numpy_array(result))
