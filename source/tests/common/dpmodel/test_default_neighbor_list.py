# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the adaptive Array API default neighbor-list builder."""

from typing import (
    Any,
)

import numpy as np
import pytest

import deepmd.dpmodel.utils.default_neighbor_list as default_nlist
from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)


def _force_search(monkeypatch: pytest.MonkeyPatch, *, cell: bool) -> None:
    """Force one public search path without changing the builder API."""
    threshold = 0 if cell else 10**9
    for name in (
        "_NUMPY_CPU_PERIODIC_CELL_LIST_THRESHOLD",
        "_NUMPY_CPU_NONPERIODIC_CELL_LIST_THRESHOLD",
    ):
        monkeypatch.setattr(default_nlist, name, threshold)


def _random_system(
    *, nframes: int = 2, nloc: int = 64, dtype: Any = np.float64
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
    fractional = rng.random((nframes, nloc, 3), dtype=dtype)
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
    return default_nlist.DefaultNeighborList().build(
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
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0e12, -1.0e12, 1.0e12],
            ]
        ],
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


def test_padded_selection_orders_rows_and_pads() -> None:
    """Row-wise selection preserves distance/index order and empty slots."""
    center = np.asarray([0, 0, 0, 1, 1], dtype=np.int64)
    neighbor = np.asarray([3, 2, 1, 4, 0], dtype=np.int64)
    distance = np.asarray([1.0, 1.0, 0.5, 2.0, 1.0], dtype=np.float64)
    result = default_nlist._select_nearest_padded(
        center, neighbor, distance, ncenters=3, nsel=4
    )
    np.testing.assert_array_equal(
        result,
        np.asarray(
            [[1, 2, 3, -1], [0, 4, -1, -1], [-1, -1, -1, -1]],
            dtype=np.int64,
        ),
    )


def test_padded_selection_rejects_excessive_imbalance() -> None:
    """A single wide row falls back before allocating a mostly empty matrix."""
    center = np.asarray([0] * 9 + [1], dtype=np.int64)
    neighbor = np.arange(10, dtype=np.int64)
    distance = np.arange(10, dtype=np.float64)
    assert (
        default_nlist._select_nearest_padded(
            center, neighbor, distance, ncenters=10, nsel=4
        )
        is None
    )


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
