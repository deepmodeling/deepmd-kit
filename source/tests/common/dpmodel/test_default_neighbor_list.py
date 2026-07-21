# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the adaptive Array API default neighbor-list builder."""

from typing import (
    Any,
)

import array_api_compat
import numpy as np
import pytest

import deepmd.dpmodel.utils.default_neighbor_list as default_nlist
from deepmd.dpmodel.utils.default_neighbor_list import (
    DefaultNeighborList,
)
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)


def _force_search(monkeypatch: pytest.MonkeyPatch, *, cell: bool) -> None:
    """Force one public search path without changing the builder API."""
    threshold = 0 if cell else 10**9
    for name in (
        "_NUMPY_CPU_PERIODIC_CELL_LIST_THRESHOLD",
        "_NUMPY_CPU_NONPERIODIC_CELL_LIST_THRESHOLD",
        "_TORCH_CPU_PERIODIC_CELL_LIST_THRESHOLD",
        "_TORCH_CPU_NONPERIODIC_CELL_LIST_THRESHOLD",
        "_JAX_CPU_PERIODIC_CELL_LIST_THRESHOLD",
        "_JAX_CPU_NONPERIODIC_CELL_LIST_THRESHOLD",
        "_TF_CPU_PERIODIC_CELL_LIST_THRESHOLD",
        "_TF_CPU_NONPERIODIC_CELL_LIST_THRESHOLD",
        "_TORCH_CUDA_PERIODIC_CELL_LIST_THRESHOLD",
        "_TORCH_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD",
        "_JAX_CUDA_PERIODIC_CELL_LIST_THRESHOLD",
        "_JAX_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD",
        "_TF_CUDA_PERIODIC_CELL_LIST_THRESHOLD",
        "_TF_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD",
    ):
        monkeypatch.setattr(default_nlist, name, threshold)


def _random_system(
    *, nframes: int = 2, nloc: int = 48, dtype: Any = np.float64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create reproducible triclinic frames without distance degeneracies."""
    rng = np.random.default_rng(20260721)
    box = np.asarray(
        [
            [[9.0, 0.0, 0.0], [0.8, 8.5, 0.0], [0.3, 0.6, 9.5]],
            [[8.5, 0.0, 0.0], [-0.5, 9.2, 0.0], [0.4, -0.2, 8.8]],
        ][:nframes],
        dtype=dtype,
    )
    fractional = rng.random((nframes, nloc, 3))
    coord = np.matmul(fractional, box)
    atype = rng.integers(0, 2, size=(nframes, nloc), dtype=np.int64)
    return coord, atype, box


def _build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cell: bool,
    coord: Any,
    atype: Any,
    box: Any,
    rcut: float,
    sel: list[int],
    pair_excl: PairExcludeMask | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Build with a forced dense or cell-list implementation."""
    _force_search(monkeypatch, cell=cell)
    return DefaultNeighborList().build(
        coord, atype, box, rcut, sel, pair_excl=pair_excl
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("periodic", [False, True])
def test_cell_list_matches_dense(
    monkeypatch: pytest.MonkeyPatch, periodic: bool, dtype: Any
) -> None:
    """The sparse candidate search preserves the dense quartet exactly."""
    coord, atype, box = _random_system(dtype=dtype)
    box_arg = box if periodic else None
    dense = _build(
        monkeypatch,
        cell=False,
        coord=coord,
        atype=atype,
        box=box_arg,
        rcut=3.2,
        sel=[24, 24],
    )
    cell = _build(
        monkeypatch,
        cell=True,
        coord=coord,
        atype=atype,
        box=box_arg,
        rcut=3.2,
        sel=[24, 24],
    )
    for dense_value, cell_value in zip(dense, cell, strict=True):
        np.testing.assert_allclose(dense_value, cell_value)


def test_cell_list_exact_cutoff_virtual_and_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boundary neighbors, virtual atoms, and exclusion holes match dense search."""
    coord = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    atype = np.asarray([[0, 1, 0, -1]], dtype=np.int64)
    pair_excl = PairExcludeMask(2, [(0, 1)])
    dense = _build(
        monkeypatch,
        cell=False,
        coord=coord,
        atype=atype,
        box=None,
        rcut=1.0,
        sel=[4, 4],
        pair_excl=pair_excl,
    )
    cell = _build(
        monkeypatch,
        cell=True,
        coord=coord,
        atype=atype,
        box=None,
        rcut=1.0,
        sel=[4, 4],
        pair_excl=pair_excl,
    )
    for dense_value, cell_value in zip(dense, cell, strict=True):
        np.testing.assert_array_equal(dense_value, cell_value)


def test_cell_list_preserves_equal_distance_image_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable ordering of symmetry-equivalent periodic images matches dense search."""
    coord = np.asarray([[[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]], dtype=np.float64)
    atype = np.asarray([[0, 0]], dtype=np.int64)
    box = 2.0 * np.eye(3, dtype=np.float64)[None, :, :]
    dense = _build(
        monkeypatch,
        cell=False,
        coord=coord,
        atype=atype,
        box=box,
        rcut=2.1,
        sel=[100],
    )
    cell = _build(
        monkeypatch,
        cell=True,
        coord=coord,
        atype=atype,
        box=box,
        rcut=2.1,
        sel=[100],
    )
    for dense_value, cell_value in zip(dense, cell, strict=True):
        np.testing.assert_array_equal(dense_value, cell_value)


def test_cell_list_array_api_strict() -> None:
    """The low-complexity helper uses operations from the strict Array API."""
    strict = pytest.importorskip("array_api_strict")
    coord, atype, _ = _random_system(nframes=1, nloc=24)
    coord_strict = strict.asarray(coord)
    atype_strict = strict.asarray(atype)
    result = default_nlist._build_neighbor_list_cell(
        strict.reshape(coord_strict, (1, -1)),
        atype_strict,
        24,
        3.2,
        32,
    )

    # Compare against the historical NumPy implementation as the oracle.
    reference = default_nlist.build_neighbor_list(
        coord.reshape(1, -1),
        atype,
        24,
        3.2,
        [32],
        distinguish_types=False,
    )
    assert array_api_compat.is_array_api_obj(result)
    np.testing.assert_array_equal(np.asarray(result), reference)


def test_cell_list_torch_namespace_and_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch receives tensor outputs while extended-coordinate gradients survive."""
    torch = pytest.importorskip("torch")
    coord, atype, box = _random_system(nframes=1, nloc=32)
    coord_t = torch.tensor(coord, dtype=torch.float64, requires_grad=True)
    atype_t = torch.tensor(atype, dtype=torch.int64)
    box_t = torch.tensor(box, dtype=torch.float64)

    dense = _build(
        monkeypatch,
        cell=False,
        coord=coord_t.detach(),
        atype=atype_t,
        box=box_t,
        rcut=3.2,
        sel=[24, 24],
    )
    cell = _build(
        monkeypatch,
        cell=True,
        coord=coord_t,
        atype=atype_t,
        box=box_t,
        rcut=3.2,
        sel=[24, 24],
    )
    for dense_value, cell_value in zip(dense, cell, strict=True):
        torch.testing.assert_close(dense_value, cell_value)

    cell[0].sum().backward()
    assert coord_t.grad is not None


def test_torch_cpu_threshold() -> None:
    """PyTorch CPU stays dense until the rcut=6 crossover."""
    torch = pytest.importorskip("torch")
    coord = torch.zeros((1, 1, 3), dtype=torch.float64)
    assert not default_nlist._supports_cell_list(coord, 2047, periodic=False)
    assert default_nlist._supports_cell_list(coord, 2048, periodic=False)


def test_cell_list_jax_eager() -> None:
    """Eager JAX builds the dynamic candidate list exactly outside ``jit``."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = pytest.importorskip("jax.numpy")
    coord, atype, _ = _random_system(nframes=1, nloc=256)
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

    result = DefaultNeighborList().build(
        jnp.asarray(coord),
        jnp.asarray(atype),
        None,
        3.2,
        [32],
        pair_excl=pair_excl,
    )[2]
    result.block_until_ready()
    np.testing.assert_array_equal(np.asarray(result), reference)
    assert not default_nlist._supports_cell_list(
        jnp.asarray(coord), 255, periodic=False
    )
    assert default_nlist._supports_cell_list(jnp.asarray(coord), 256, periodic=False)
    traced_support = jax.jit(
        lambda cc: jnp.asarray(
            default_nlist._supports_cell_list(cc, 256, periodic=False)
        )
    )(jnp.asarray(coord))
    assert not bool(np.asarray(traced_support))


def test_cell_list_tensorflow_function_periodic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorFlow graph mode supports dynamic candidates and periodic ghosts."""
    tf = pytest.importorskip("tensorflow")
    ndtf = pytest.importorskip("deepmd._vendors.ndtensorflow")
    coord, atype, box = _random_system(nframes=1, nloc=24)
    dense = _build(
        monkeypatch,
        cell=False,
        coord=coord,
        atype=atype,
        box=box,
        rcut=3.2,
        sel=[24, 24],
    )
    _force_search(monkeypatch, cell=True)

    @tf.function(
        autograph=False,
        input_signature=[
            tf.TensorSpec([None, 24, 3], tf.float64),
            tf.TensorSpec([None, 24], tf.int64),
            tf.TensorSpec([None, 3, 3], tf.float64),
        ],
    )
    def build_graph(cc: Any, aa: Any, bb: Any) -> tuple[Any, Any, Any, Any]:
        result = DefaultNeighborList().build(
            ndtf.asarray(cc), ndtf.asarray(aa), ndtf.asarray(bb), 3.2, [24, 24]
        )
        return tuple(value.unwrap() for value in result)

    cell = build_graph(coord, atype, box)
    for dense_value, cell_value in zip(dense, cell, strict=True):
        np.testing.assert_allclose(dense_value, cell_value.numpy())


def test_tensorflow_cpu_threshold() -> None:
    """TensorFlow keeps sub-4096-atom CPU systems on dense search."""
    tf = pytest.importorskip("tensorflow")
    ndtf = pytest.importorskip("deepmd._vendors.ndtensorflow")
    tf_coord = ndtf.asarray(tf.zeros((1, 1, 3), dtype=tf.float64))
    assert not default_nlist._supports_cell_list(tf_coord, 4095, periodic=False)
    assert default_nlist._supports_cell_list(tf_coord, 4096, periodic=False)


def test_automatic_cpu_thresholds() -> None:
    """Measured CPU crossovers keep small systems on the dense fast path."""
    assert not default_nlist._supports_cell_list(
        np.zeros((1, 31, 3)),
        31,
        periodic=True,
    )
    assert default_nlist._supports_cell_list(
        np.zeros((1, 32, 3)),
        32,
        periodic=True,
    )
    assert not default_nlist._supports_cell_list(
        np.zeros((1, 255, 3)),
        255,
        periodic=False,
    )
    assert default_nlist._supports_cell_list(
        np.zeros((1, 256, 3)),
        256,
        periodic=False,
    )
