# SPDX-License-Identifier: LGPL-3.0-or-later
"""Layout-agnostic edge-aggregation regression test for the dpmodel DPA4.

The standard ``call`` path consumes a padded edge cache (``E = n_nodes * nnei``
with ``dst == repeat(arange(n_nodes), nnei)``), while ``call_with_edges``
consumes an arbitrary sparse edge list. The destination-wise aggregations
(geometric initial embedding, environment initial embedding, and the attention
softmax) are scatter reductions over ``dst``, so both layouts must yield the
same descriptor for the same physical edges. These tests feed the sparse path
the padded path's valid edges -- once in row-major order and once permuted into
an arbitrary order with a non-uniform per-node degree -- and assert the two
descriptors agree. They are the regression guard for the scatter-by-``dst``
aggregation.
"""

import numpy as np

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)


def build_neighbor_list_np(coord, rcut, nnei):
    """Build a padded, distance-sorted gas-phase neighbor list (no PBC).

    Parameters
    ----------
    coord
        Coordinates with shape (nf, nloc, 3).
    rcut
        Cutoff radius.
    nnei
        Number of neighbor slots; pads with -1.

    Returns
    -------
    np.ndarray
        Neighbor list with shape (nf, nloc, nnei) holding local indices.
    """
    nf, nloc, _ = coord.shape
    nlist = -np.ones((nf, nloc, nnei), dtype=np.int64)
    for f in range(nf):
        dist = np.linalg.norm(coord[f][:, None, :] - coord[f][None, :, :], axis=-1)
        for i in range(nloc):
            neighbors = [
                (dist[i, j], j) for j in range(nloc) if j != i and dist[i, j] < rcut
            ]
            neighbors.sort()
            for slot, (_, j) in enumerate(neighbors[:nnei]):
                nlist[f, i, slot] = j
    return nlist


def build_sparse_edges_from_nlist(coord, nlist):
    """Extract the valid physical edges of a padded neighbor list.

    The padded layout keeps one slot per neighbor (``-1`` marks padding). The
    sparse contract for :meth:`DescrptDPA4.call_with_edges` is one explicit edge
    per kept slot, indexing the flattened frame-major node axis
    (``node = f * nloc + i``). The edge vector points from the center toward the
    neighbor, matching the padded path's ``r_j - r_i``.

    Parameters
    ----------
    coord
        Coordinates with shape (nf, nloc, 3).
    nlist
        Neighbor list with shape (nf, nloc, nnei); -1 marks padding.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``edge_index`` with shape (2, E) (rows are src, dst) and ``edge_vec``
        with shape (E, 3), aligned on the same edge axis in row-major
        ``(frame, center, slot)`` order.
    """
    nf, nloc, nnei = nlist.shape
    src, dst, vec = [], [], []
    for f in range(nf):
        for i in range(nloc):
            for s in range(nnei):
                j = int(nlist[f, i, s])
                if j < 0:
                    continue
                src.append(f * nloc + j)
                dst.append(f * nloc + i)
                vec.append(coord[f, j] - coord[f, i])
    edge_index = np.asarray([src, dst], dtype=np.int64)  # (2, E)
    edge_vec = np.asarray(vec, dtype=np.float64)  # (E, 3)
    return edge_index, edge_vec


def make_descriptor() -> DescrptDPA4:
    return DescrptDPA4(
        ntypes=3,
        sel=8,
        rcut=4.0,
        channels=16,
        n_radial=8,
        lmax=2,
        mmax=1,
        n_blocks=2,
        precision="float64",
        seed=7,
        random_gamma=False,
    )


def make_inputs(seed=7, nf=2, nloc=6, rcut=4.0, nnei=8, ntypes=3):
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 3.5, size=(nf, nloc, 3))
    atype = rng.integers(0, ntypes, size=(nf, nloc))
    nlist = build_neighbor_list_np(coord, rcut, nnei)
    return coord, atype, nlist


def _run_sparse(dd, coord, atype, edge_index, edge_vec):
    nf = atype.shape[0]
    edge_mask = np.ones(edge_index.shape[1], dtype=bool)
    return np.asarray(
        dd.call_with_edges(
            coord_ext=coord,
            atype_ext=atype,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
        )[0]
    )


def test_sparse_edges_match_padded_rowmajor() -> None:
    # Row-major sparse edges reproduce the padded scatter order exactly: the
    # masked padding slots of the padded path contribute zero, so dropping them
    # leaves the destination accumulation order unchanged.
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape

    out_pad = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    edge_index, edge_vec = build_sparse_edges_from_nlist(coord, nlist)
    out_sparse = _run_sparse(dd, coord, atype, edge_index, edge_vec)

    assert out_sparse.shape == out_pad.shape == (nf, nloc, dd.get_dim_out())
    assert np.isfinite(out_sparse).all()
    np.testing.assert_allclose(out_sparse, out_pad, rtol=1e-10, atol=1e-12)


def test_sparse_edges_match_padded_permuted() -> None:
    # Permuted sparse edges exercise an arbitrary (non-row-major) ``dst`` order
    # and a non-uniform per-node degree. The destination scatter reductions are
    # order-agnostic, so the descriptor must still match the padded path within
    # float64 reassociation tolerance.
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape

    out_pad = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])

    edge_index, edge_vec = build_sparse_edges_from_nlist(coord, nlist)
    perm = np.random.default_rng(31).permutation(edge_index.shape[1])
    edge_index = edge_index[:, perm]
    edge_vec = edge_vec[perm]
    out_sparse = _run_sparse(dd, coord, atype, edge_index, edge_vec)

    assert out_sparse.shape == out_pad.shape == (nf, nloc, dd.get_dim_out())
    assert np.isfinite(out_sparse).all()
    np.testing.assert_allclose(out_sparse, out_pad, rtol=1e-10, atol=1e-12)
