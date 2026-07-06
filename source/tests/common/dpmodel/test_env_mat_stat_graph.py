# SPDX-License-Identifier: LGPL-3.0-or-later
"""Piece B: the input stat's env matrix computed via the NeighborGraph
(``from_dense_quartet`` -> ``edge_env_mat``) is BIT-IDENTICAL to the dense
``EnvMat`` path.

This is the same machinery the dpa1 graph forward uses. ``from_dense_quartet``
reuses the same neighbor set and padding, ``edge_env_mat`` mirrors
``EnvMat.call``, and the row-major ``(frame, center, slot)`` edge order maps 1:1
back to the dense ``(nf, nloc, nsel, 4)`` env-matrix tensor -- so the stored
``davg``/``dstd`` are unchanged. The parity must hold with no exclusion, with
model-level ``pair_exclude_types`` (folded into the nlist at BUILD, Piece A),
and with descriptor-level ``exclude_types``.
"""

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
    DescrptSeA,
)
from deepmd.dpmodel.utils.env_mat_stat import (
    EnvMatStatSe,
)


def _sample() -> dict:
    rng = np.random.default_rng(0)
    nf, nloc = 2, 6
    coord = rng.normal(size=(nf, nloc, 3)) * 2.0
    atype = np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.int64)
    box = np.tile((np.eye(3) * 12.0).reshape(1, 9), (nf, 1))
    return {"coord": coord, "atype": atype, "box": box}


def _stats(exclude_types, pair_exclude_types, use_graph):
    ds = DescrptSeA(6.0, 0.5, [10, 10], exclude_types=exclude_types)
    sample = _sample()
    if pair_exclude_types is not None:
        sample["pair_exclude_types"] = pair_exclude_types
    st = EnvMatStatSe(ds, use_graph=use_graph)
    st.load_or_compute_stats([sample], None)
    mean, stddev = st()
    return np.asarray(mean), np.asarray(stddev)


def _assert_graph_equals_dense(exclude_types, pair_exclude_types) -> None:
    m_dense, s_dense = _stats(exclude_types, pair_exclude_types, use_graph=False)
    m_graph, s_graph = _stats(exclude_types, pair_exclude_types, use_graph=True)
    np.testing.assert_allclose(m_graph, m_dense, rtol=1e-15, atol=1e-15)
    np.testing.assert_allclose(s_graph, s_dense, rtol=1e-15, atol=1e-15)


def test_graph_equals_dense_no_exclusion() -> None:
    _assert_graph_equals_dense(exclude_types=[], pair_exclude_types=None)


def test_graph_equals_dense_model_level_exclusion() -> None:
    _assert_graph_equals_dense(exclude_types=[], pair_exclude_types=[[0, 1]])


def test_graph_equals_dense_descriptor_level_exclusion() -> None:
    _assert_graph_equals_dense(exclude_types=[[0, 1]], pair_exclude_types=None)


def _dpa1_block_stats(pair_exclude_types, use_graph):
    """Stats for the actual dpa1 (mixed_types) block, the wired descriptor."""
    ds = DescrptDPA1(6.0, 0.5, 20, ntypes=2, attn_layer=0)
    block = ds.se_atten
    sample = _sample()
    if pair_exclude_types is not None:
        sample["pair_exclude_types"] = pair_exclude_types
    st = EnvMatStatSe(block, use_graph=use_graph)
    st.load_or_compute_stats([sample], None)
    mean, stddev = st()
    return np.asarray(mean), np.asarray(stddev)


def test_dpa1_block_graph_equals_dense() -> None:
    """The wired dpa1 block (use_graph=True) is bit-identical to the dense path."""
    for pair_exclude_types in (None, [[0, 1]]):
        m_dense, s_dense = _dpa1_block_stats(pair_exclude_types, use_graph=False)
        m_graph, s_graph = _dpa1_block_stats(pair_exclude_types, use_graph=True)
        np.testing.assert_allclose(m_graph, m_dense, rtol=1e-15, atol=1e-15)
        np.testing.assert_allclose(s_graph, s_dense, rtol=1e-15, atol=1e-15)
