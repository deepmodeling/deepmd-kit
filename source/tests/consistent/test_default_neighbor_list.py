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
    coord = np.matmul(rng.random((1, nloc, 3), dtype=dtype), box)
    atype = rng.integers(0, 2, size=(1, nloc), dtype=np.int64)
    return coord, atype, box


class _CellListBackendMixin:
    """Run one shared geometric/exclusion case through each Array API backend."""

    def _backend_devices(self) -> list[Any]:
        """Return validated devices, including an accelerator when available."""
        return [None]

    def _backend_arrays(
        self, coord: np.ndarray, atype: np.ndarray, device: Any
    ) -> tuple[Any, Any]:
        """Convert the shared NumPy case to one backend device under test."""
        raise NotImplementedError

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Synchronize and convert one backend result for exact comparison."""
        return np.asarray(value)

    def test_backend_cell_list_matches_dense(self) -> None:
        """All backends preserve common periodic and nonperiodic references."""
        nloc = 24
        rcut = 3.2
        nsel = 32
        coord, atype, box = _random_system(nloc=nloc, dtype=np.float32)
        pair_excl = PairExcludeMask(2, [(0, 1)])
        for periodic in (False, True):
            if periodic:
                search_coord, search_atype, _ = default_nlist.extend_coord_with_ghosts(
                    coord,
                    atype,
                    box,
                    rcut,
                )
            else:
                search_coord = coord.reshape(1, -1)
                search_atype = atype
            reference = default_nlist.build_neighbor_list(
                search_coord,
                search_atype,
                nloc,
                rcut,
                [nsel],
                distinguish_types=False,
                pair_excl=pair_excl,
            )
            for device in self._backend_devices():
                with self.subTest(periodic=periodic, device=str(device)):
                    coord_backend, atype_backend = self._backend_arrays(
                        search_coord, search_atype, device
                    )
                    # Exercise the periodic boundary-shell compaction route on
                    # every backend/device where it is supported.  Production
                    # thresholds are benchmark-derived and otherwise exceed the
                    # deliberately small cross-backend fixture.
                    with mock.patch.multiple(
                        default_nlist,
                        _NUMPY_CPU_PERIODIC_COMPACTION_THRESHOLD=0,
                        _TORCH_CPU_PERIODIC_COMPACTION_THRESHOLD=0,
                        _JAX_CPU_PERIODIC_COMPACTION_THRESHOLD=0,
                        _TF_CPU_PERIODIC_COMPACTION_THRESHOLD=0,
                        _TORCH_CUDA_PERIODIC_COMPACTION_THRESHOLD=0,
                    ):
                        result = default_nlist._build_neighbor_list_cell(
                            coord_backend,
                            atype_backend,
                            nloc,
                            rcut,
                            nsel,
                            pair_excl=pair_excl,
                        )
                    assert array_api_compat.is_array_api_obj(result)
                    np.testing.assert_array_equal(self._to_numpy(result), reference)

    def test_backend_virtual_outlier_does_not_expand_grid(self) -> None:
        """Virtual placeholder coordinates do not affect real-atom cells."""
        coord = np.asarray(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [1.0e12, -1.0e12, 1.0e12],
                ],
                [
                    [1.0e12, 0.0, 0.0],
                    [0.0, -1.0e12, 0.0],
                    [0.0, 0.0, 1.0e12],
                    [-1.0e12, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        )
        atype = np.asarray([[0, 1, 0, -1], [-1, -1, -1, -1]], dtype=np.int64)
        reference = default_nlist.build_neighbor_list(
            coord,
            atype,
            nloc=4,
            rcut=1.0,
            sel=[8],
            distinguish_types=False,
        )
        for device in self._backend_devices():
            with self.subTest(device=str(device)):
                coord_backend, atype_backend = self._backend_arrays(
                    coord, atype, device
                )
                result = default_nlist._build_neighbor_list_cell(
                    coord_backend,
                    atype_backend,
                    nloc=4,
                    rcut=1.0,
                    nsel=8,
                )
                np.testing.assert_array_equal(self._to_numpy(result), reference)


@unittest.skipUnless(INSTALLED_ARRAY_API_STRICT, "array_api_strict is not installed")
class TestArrayAPIStrictDefaultNeighborList(_CellListBackendMixin, unittest.TestCase):
    """Validate strict Array API execution and conservative public dispatch."""

    def _backend_arrays(
        self, coord: np.ndarray, atype: np.ndarray, device: Any
    ) -> tuple[Any, Any]:
        return strict.asarray(coord), strict.asarray(atype)

    def test_unknown_array_namespace_uses_dense_path(self) -> None:
        """Unbenchmarked Array API namespaces do not inherit another threshold."""
        coord = strict.zeros((1, 1, 3), dtype=strict.float32)
        self.assertFalse(
            default_nlist._supports_cell_list(coord, 10**6, periodic=False)
        )


@unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
class TestTorchDefaultNeighborList(_CellListBackendMixin, unittest.TestCase):
    """Validate PyTorch gradients, dispatch thresholds, and supported devices."""

    def _backend_devices(self) -> list[Any]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

    def _backend_arrays(
        self, coord: np.ndarray, atype: np.ndarray, device: Any
    ) -> tuple[Any, Any]:
        return torch.as_tensor(coord, device=device), torch.as_tensor(
            atype, device=device
        )

    def _to_numpy(self, value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()

    def test_torch_namespace_and_gradient(self) -> None:
        coord, atype, box = _random_system(nloc=32)
        coord_t = torch.tensor(coord, dtype=torch.float64, requires_grad=True)
        atype_t = torch.tensor(atype, dtype=torch.int64)
        box_t = torch.tensor(box, dtype=torch.float64)

        with mock.patch.multiple(
            default_nlist,
            _TORCH_CPU_PERIODIC_CELL_LIST_THRESHOLD=0,
        ):
            result = default_nlist.DefaultNeighborList().build(
                coord_t, atype_t, box_t, 3.2, [24, 24]
            )
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in result))
        result[0].sum().backward()
        self.assertIsNotNone(coord_t.grad)

    def test_torch_cpu_threshold(self) -> None:
        coord = torch.zeros((1, 1, 3), dtype=torch.float64)
        self.assertFalse(default_nlist._supports_cell_list(coord, 2047, periodic=False))
        self.assertTrue(default_nlist._supports_cell_list(coord, 2048, periodic=False))

    def test_torch_unvalidated_device_uses_dense_path(self) -> None:
        """Non-CPU/CUDA torch devices retain the conservative dense path."""
        coord = torch.zeros((1, 1, 3), dtype=torch.float64)
        with mock.patch.object(
            default_nlist.array_api_compat,
            "device",
            return_value=mock.Mock(type="mps"),
        ):
            self.assertFalse(
                default_nlist._supports_cell_list(coord, 10**6, periodic=False)
            )

    def test_torch_without_compiler_api_uses_compact_selection(self) -> None:
        """PyTorch variants without compiler state avoid dynamic row padding."""
        coord = torch.zeros((1, 1, 3), dtype=torch.float64)
        with mock.patch.object(torch, "compiler", None):
            self.assertFalse(default_nlist._supports_padded_selection(coord))


@unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
class TestJAXDefaultNeighborList(_CellListBackendMixin, unittest.TestCase):
    """Validate eager JAX construction and traced/unknown-device fallbacks."""

    def _backend_devices(self) -> list[Any]:
        devices = [jax.devices("cpu")[0]]
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        if gpu_devices:
            devices.append(gpu_devices[0])
        return devices

    def _backend_arrays(
        self, coord: np.ndarray, atype: np.ndarray, device: Any
    ) -> tuple[Any, Any]:
        return jax.device_put(coord, device), jax.device_put(atype, device)

    def test_jax_dispatch_and_tracing(self) -> None:
        coord, atype, _ = _random_system(nloc=256, dtype=np.float32)
        coord_jax, atype_jax = self._backend_arrays(coord, atype, jax.devices("cpu")[0])

        result = default_nlist.DefaultNeighborList().build(
            coord_jax,
            atype_jax,
            None,
            3.2,
            [32],
        )[2]
        result.block_until_ready()
        self.assertEqual(result.shape, (1, 256, 32))
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

    def test_jax_unvalidated_device_uses_dense_path(self) -> None:
        coord, _, _ = _random_system(nloc=1, dtype=np.float32)
        coord_jax, _ = self._backend_arrays(
            coord,
            np.zeros((1, 1), dtype=np.int64),
            jax.devices("cpu")[0],
        )
        with mock.patch.object(
            default_nlist.array_api_compat,
            "device",
            return_value=mock.Mock(platform="tpu"),
        ):
            self.assertFalse(
                default_nlist._supports_cell_list(coord_jax, 10**6, periodic=False)
            )


@unittest.skipUnless(INSTALLED_TF2, "TF2 backend is not installed")
class TestTF2DefaultNeighborList(_CellListBackendMixin, unittest.TestCase):
    """Validate the Array API neighbor search in the opt-in TF2 test job."""

    def _backend_devices(self) -> list[Any]:
        devices = ["/CPU:0"]
        if tf.config.list_logical_devices("GPU"):
            devices.append("/GPU:0")
        return devices

    def _backend_arrays(
        self, coord: np.ndarray, atype: np.ndarray, device: Any
    ) -> tuple[Any, Any]:
        with tf.device(device):
            return ndtf.asarray(tf.convert_to_tensor(coord)), ndtf.asarray(
                tf.convert_to_tensor(atype)
            )

    def _to_numpy(self, value: Any) -> np.ndarray:
        return value.unwrap().numpy()

    def test_tf2_cpu_threshold(self) -> None:
        """TensorFlow keeps sub-4096-atom CPU systems on dense search."""
        with tf.device("/CPU:0"):
            coord = ndtf.asarray(tf.zeros((1, 1, 3), dtype=tf.float64))
        self.assertFalse(default_nlist._supports_cell_list(coord, 4095, periodic=False))
        self.assertTrue(default_nlist._supports_cell_list(coord, 4096, periodic=False))

    def test_tf2_unvalidated_device_uses_dense_path(self) -> None:
        with tf.device("/CPU:0"):
            coord = ndtf.asarray(tf.zeros((1, 1, 3), dtype=tf.float32))
        with mock.patch.object(
            default_nlist.array_api_compat,
            "device",
            return_value="TPU:0",
        ):
            self.assertFalse(
                default_nlist._supports_cell_list(coord, 10**6, periodic=False)
            )

    def test_tf2_unplaced_device_uses_later_threshold(self) -> None:
        """An unplaced TF graph is conservative across validated CPU/GPU paths."""
        with tf.device("/CPU:0"):
            coord = ndtf.asarray(tf.zeros((1, 1, 3), dtype=tf.float32))
        with (
            mock.patch.object(
                default_nlist.array_api_compat,
                "device",
                return_value="",
            ),
            mock.patch.multiple(
                default_nlist,
                _TF_CPU_NONPERIODIC_CELL_LIST_THRESHOLD=1024,
                _TF_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD=2048,
            ),
        ):
            self.assertFalse(
                default_nlist._supports_cell_list(coord, 2047, periodic=False)
            )
            self.assertTrue(
                default_nlist._supports_cell_list(coord, 2048, periodic=False)
            )

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
