# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the backend-neutral DPA4 edge-cache acceleration seams."""

import numpy as np

from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
    _edge_cache_from_arrays,
    edge_cache_to_dtype,
)


def test_fused_builders_replace_reference_and_initialize_step_cache() -> None:
    calls: dict[str, int] = {"radial": 0, "wigner": 0}
    keep_seen: list[np.ndarray] = []

    def unexpected_reference(_: np.ndarray) -> np.ndarray:
        raise AssertionError("the reference builder must not run")

    def fused_radial(
        edge_len: np.ndarray, edge_keep: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        calls["radial"] += 1
        keep_seen.append(edge_keep.copy())
        edge_env = 2.0 * edge_len * edge_keep
        edge_rbf = np.concatenate([edge_len, edge_len * edge_len], axis=-1)
        return edge_env, edge_rbf * edge_keep

    def fused_wigner(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls["wigner"] += 1
        marker = np.arange(quaternion.shape[0], dtype=quaternion.dtype)[:, None, None]
        return marker + 1.0, -(marker + 1.0)

    cache = _edge_cache_from_arrays(
        type_ebed=np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]]),
        edge_index=np.array([[1, 2, 0], [0, 0, 1]], dtype=np.int64),
        edge_vec=np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 2.0]]),
        edge_mask=np.array([True, False, True]),
        compute_dtype=np.float64,
        eps=1.0e-12,
        deg_norm_floor=1.0,
        inner_clamp=None,
        bridging_switch=None,
        edge_envelope=unexpected_reference,
        radial_basis=unexpected_reference,
        random_gamma=False,
        wigner_calc=unexpected_reference,
        fused_radial=fused_radial,
        fused_wigner=fused_wigner,
    )

    assert calls == {"radial": 1, "wigner": 1}
    np.testing.assert_array_equal(keep_seen[0], np.array([[1.0], [0.0], [1.0]]))
    np.testing.assert_allclose(cache.edge_env, np.array([[6.0], [0.0], [4.0]]))
    np.testing.assert_allclose(
        cache.edge_rbf,
        np.array([[3.0, 9.0], [0.0, 0.0], [2.0, 4.0]]),
    )
    np.testing.assert_allclose(cache.D_full[:, 0, 0], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(cache.Dt_full[:, 0, 0], np.array([-1.0, -2.0, -3.0]))
    assert cache.csr_cache == {}

    cache.csr_cache["dst"] = (np.array([0, 2, 1]), np.array([0, 2, 3, 3]))
    converted = edge_cache_to_dtype(cache, np.float32)
    assert converted.csr_cache is not cache.csr_cache
    assert converted.csr_cache["dst"] is cache.csr_cache["dst"]

    cache.csr_cache = None
    assert edge_cache_to_dtype(cache, np.float32).csr_cache is None


def test_edge_cache_normalizes_graph_csr_to_int64() -> None:
    def wigner_calc(quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        block = np.ones((quaternion.shape[0], 1, 1), dtype=quaternion.dtype)
        return block, block

    cache = _edge_cache_from_arrays(
        type_ebed=np.array([[1.0], [2.0]]),
        edge_index=np.array([[1, 0], [0, 1]], dtype=np.int32),
        edge_vec=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        edge_mask=np.array([True, True]),
        compute_dtype=np.float64,
        eps=1.0e-12,
        deg_norm_floor=1.0,
        inner_clamp=None,
        bridging_switch=None,
        edge_envelope=np.ones_like,
        radial_basis=np.ones_like,
        random_gamma=False,
        wigner_calc=wigner_calc,
        csr_cache={
            "dst": (
                np.array([0, 1], dtype=np.int32),
                np.array([0, 1, 2], dtype=np.int32),
            ),
            "src": (
                np.array([1, 0], dtype=np.int32),
                np.array([0, 1, 2], dtype=np.int32),
            ),
        },
    )

    assert cache.src.dtype == np.int64
    assert cache.dst.dtype == np.int64
    assert cache.csr_cache is not None
    for order, row_ptr in cache.csr_cache.values():
        assert order.dtype == np.int64
        assert row_ptr.dtype == np.int64
