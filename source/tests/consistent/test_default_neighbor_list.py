# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-gated consistency tests for the default neighbor-list builder."""

import unittest
import unittest.mock as mock
from typing import (
    Any,
)

import array_api_compat
import numpy as np

import deepmd.dpmodel.utils.default_neighbor_list as default_nlist
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)

from .common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_TF2,
)

if INSTALLED_ARRAY_API_STRICT:
    import array_api_strict as strict

if INSTALLED_PT:
    import torch

if INSTALLED_JAX:
    import jax
    import jax.numpy as jnp

if INSTALLED_TF2:
    import tensorflow as tf

    from deepmd._vendors import ndtensorflow as ndtf


def _random_system(
    *, nloc: int, dtype: Any = np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one reproducible triclinic frame for backend comparisons."""
    rng = np.random.default_rng(20260721)
    box = np.asarray(
        [[[9.0, 0.0, 0.0], [0.8, 8.5, 0.0], [0.3, 0.6, 9.5]]],
        dtype=dtype,
    )
    coord = np.matmul(rng.random((1, nloc, 3)), box)
    atype = rng.integers(0, 2, size=(1, nloc), dtype=np.int64)
    return coord, atype, box


@unittest.skipUnless(INSTALLED_ARRAY_API_STRICT, "array_api_strict is not installed")
class TestArrayAPIStrictDefaultNeighborList(unittest.TestCase):
    """Validate that the cell-list helper stays within the strict Array API."""

    def test_array_api_strict_cell_list_matches_dense(self) -> None:
        coord, atype, _ = _random_system(nloc=24)
        coord_strict = strict.asarray(coord)
        atype_strict = strict.asarray(atype)
        result = default_nlist._build_neighbor_list_cell(
            strict.reshape(coord_strict, (1, -1)),
            atype_strict,
            24,
            3.2,
            32,
        )
        reference = default_nlist.build_neighbor_list(
            coord.reshape(1, -1),
            atype,
            24,
            3.2,
            [32],
            distinguish_types=False,
        )
        self.assertTrue(array_api_compat.is_array_api_obj(result))
        np.testing.assert_array_equal(np.asarray(result), reference)


@unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
class TestTorchDefaultNeighborList(unittest.TestCase):
    """Validate PyTorch output, gradients, and measured CPU dispatch."""

    def test_torch_namespace_and_gradient(self) -> None:
        coord, atype, box = _random_system(nloc=32)
        coord_t = torch.tensor(coord, dtype=torch.float64, requires_grad=True)
        atype_t = torch.tensor(atype, dtype=torch.int64)
        box_t = torch.tensor(box, dtype=torch.float64)

        with mock.patch.multiple(
            default_nlist,
            _TORCH_CPU_PERIODIC_CELL_LIST_THRESHOLD=10**9,
        ):
            dense = default_nlist.DefaultNeighborList().build(
                coord_t.detach(), atype_t, box_t, 3.2, [24, 24]
            )
        with mock.patch.multiple(
            default_nlist,
            _TORCH_CPU_PERIODIC_CELL_LIST_THRESHOLD=0,
        ):
            cell = default_nlist.DefaultNeighborList().build(
                coord_t, atype_t, box_t, 3.2, [24, 24]
            )
        for dense_value, cell_value in zip(dense, cell, strict=True):
            torch.testing.assert_close(dense_value, cell_value)

        cell[0].sum().backward()
        self.assertIsNotNone(coord_t.grad)

    def test_torch_cpu_threshold(self) -> None:
        coord = torch.zeros((1, 1, 3), dtype=torch.float64)
        self.assertFalse(default_nlist._supports_cell_list(coord, 2047, periodic=False))
        self.assertTrue(default_nlist._supports_cell_list(coord, 2048, periodic=False))


@unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
class TestJAXDefaultNeighborList(unittest.TestCase):
    """Validate eager JAX construction and the traced dense fallback."""

    def test_jax_eager_cell_list_matches_dense(self) -> None:
        previous_x64 = jax.config.read("jax_enable_x64")
        self.addCleanup(jax.config.update, "jax_enable_x64", previous_x64)
        jax.config.update("jax_enable_x64", True)
        coord, atype, _ = _random_system(nloc=256)
        pair_excl = PairExcludeMask(2, [(0, 1)])
        reference = default_nlist.build_neighbor_list(
            coord.reshape(1, -1),
            atype,
            256,
            3.2,
            [32],
            distinguish_types=False,
            pair_excl=pair_excl,
        )
        cpu = jax.devices("cpu")[0]
        coord_jax = jax.device_put(coord, cpu)
        atype_jax = jax.device_put(atype, cpu)

        result = default_nlist.DefaultNeighborList().build(
            coord_jax,
            atype_jax,
            None,
            3.2,
            [32],
            pair_excl=pair_excl,
        )[2]
        result.block_until_ready()
        np.testing.assert_array_equal(np.asarray(result), reference)
        self.assertFalse(
            default_nlist._supports_cell_list(coord_jax, 255, periodic=False)
        )
        self.assertTrue(
            default_nlist._supports_cell_list(coord_jax, 256, periodic=False)
        )
        self.assertTrue(default_nlist._supports_padded_selection(coord_jax))
        traced_support = jax.jit(
            lambda cc: jnp.asarray(
                default_nlist._supports_cell_list(cc, 256, periodic=False)
            )
        )(coord_jax)
        self.assertFalse(bool(np.asarray(traced_support)))
        traced_padded_support = jax.jit(
            lambda cc: jnp.asarray(default_nlist._supports_padded_selection(cc))
        )(coord_jax)
        self.assertFalse(bool(np.asarray(traced_padded_support)))


@unittest.skipUnless(INSTALLED_TF2, "TF2 backend is not installed")
class TestTF2DefaultNeighborList(unittest.TestCase):
    """Validate the Array API neighbor search in the opt-in TF2 test job."""

    def test_tf2_cpu_threshold(self) -> None:
        """TensorFlow keeps sub-4096-atom CPU systems on dense search."""
        with tf.device("/CPU:0"):
            coord = ndtf.asarray(tf.zeros((1, 1, 3), dtype=tf.float64))
        self.assertFalse(default_nlist._supports_cell_list(coord, 4095, periodic=False))
        self.assertTrue(default_nlist._supports_cell_list(coord, 4096, periodic=False))

    def test_tf2_function_periodic_cell_list_matches_dense(self) -> None:
        """A dynamic-batch TF graph preserves the dense periodic result."""
        coord, atype, box = _random_system(nloc=24)

        # Force opposite public paths so the comparison does not depend on
        # benchmark-derived dispatch thresholds or the available TF2 device.
        with mock.patch.multiple(
            default_nlist,
            _NUMPY_CPU_PERIODIC_CELL_LIST_THRESHOLD=10**9,
            _TF_CPU_PERIODIC_CELL_LIST_THRESHOLD=0,
            _TF_CUDA_PERIODIC_CELL_LIST_THRESHOLD=0,
        ):
            dense = default_nlist.DefaultNeighborList().build(
                coord, atype, box, 3.2, [24, 24]
            )

            # The TF1 consistency suite can leave legacy Dimension objects
            # enabled process-wide.  Reproduce that shape representation here,
            # then restore the TF2 job's original setting after graph tracing.
            v2_shapes_enabled = tf.TensorSpec([None], tf.float64).shape[0] is None
            tf.compat.v1.disable_v2_tensorshape()
            try:

                @tf.function(
                    autograph=False,
                    input_signature=[
                        tf.TensorSpec([None, 24, 3], tf.float64),
                        tf.TensorSpec([None, 24], tf.int64),
                        tf.TensorSpec([None, 3, 3], tf.float64),
                    ],
                )
                def build_graph(cc: Any, aa: Any, bb: Any) -> tuple[Any, Any, Any, Any]:
                    result = default_nlist.DefaultNeighborList().build(
                        ndtf.asarray(cc),
                        ndtf.asarray(aa),
                        ndtf.asarray(bb),
                        3.2,
                        [24, 24],
                    )
                    return tuple(value.unwrap() for value in result)

                cell = build_graph(coord, atype, box)
            finally:
                if v2_shapes_enabled:
                    tf.compat.v1.enable_v2_tensorshape()

        for dense_value, cell_value in zip(dense, cell, strict=True):
            np.testing.assert_allclose(dense_value, cell_value.numpy())


if __name__ == "__main__":
    unittest.main()
