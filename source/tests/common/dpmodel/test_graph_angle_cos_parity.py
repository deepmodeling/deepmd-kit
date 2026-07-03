# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for graph_angle_cos vs dpa3 dense cosine_ij and se_t dot form."""

from __future__ import annotations

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    attach_angles,
    build_neighbor_graph,
    graph_angle_cos,
)


# ---------------------------------------------------------------------------
# Step 1/4: explicit geometry sanity
# ---------------------------------------------------------------------------


def test_graph_angle_cos_matches_normalized_dot():
    """Center at origin, neighbors along x and y: cos=0 (perpendicular).

    Use rcut=1.1 so neighbors (distance 1.0 from center) are included but
    the neighbor-to-neighbor distance (sqrt(2)~1.41) is NOT, making atom 0
    the ONLY center with multiple neighbors => exactly ONE unordered angle.
    """
    coord = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]]])
    atype = np.array([[0, 0, 0]])
    ng = attach_angles(build_neighbor_graph(coord, atype, None, 1.1), a_rcut=1.1)
    cos = graph_angle_cos(ng.angle_index, ng.edge_vec)
    real = np.asarray(ng.angle_mask)
    vals = np.asarray(cos)[real]
    # only atom 0 has 2 neighbors within rcut => exactly ONE unordered angle
    assert int(real.sum()) == 1
    # perpendicular vectors: dot = 0, (1-eps) scaling => val ~0.0 to atol 1e-6
    np.testing.assert_allclose(vals, [0.0], atol=1e-6)


def test_graph_angle_cos_parallel():
    """Two neighbors in the SAME direction as seen from center: cos ~(1-eps).

    Use rcut=1.5 so only center→neighbor1 (dist=1) and center→neighbor2 (dist=2)
    but atom1→atom2 (dist=1) is NOT seen from atom2 as a second neighbor.
    Actually place center at 1.0, so neighbors at 0.0 and 2.0 are collinear.
    rcut=1.1 => center (at 1.0) sees atoms at 0.0 (dist=1.0) and 2.0 (dist=1.0),
    but those two atoms don't see each other (dist=2.0 > 1.1).
    """
    coord = np.array([[[1.0, 0, 0], [0.0, 0, 0], [2.0, 0, 0]]])
    atype = np.array([[0, 0, 0]])
    ng = attach_angles(build_neighbor_graph(coord, atype, None, 1.1), a_rcut=1.1)
    cos = graph_angle_cos(ng.angle_index, ng.edge_vec)
    real = np.asarray(ng.angle_mask)
    vals = np.asarray(cos)[real]
    assert int(real.sum()) == 1
    # edge_vec = neighbor - center; both neighbors are in opposite x-directions
    # from the center: edge to atom1=(0,0,0) is (-1,0,0); edge to atom2=(2,0,0) is (+1,0,0)
    # For unit-norm vectors (norm=1.0):
    #   na = va / (1 + eps),  nb = vb / (1 + eps)
    #   dot(na, nb) = -1 / (1+eps)^2
    #   cos = -1 / (1+eps)^2 * (1 - eps)
    eps = 1e-6
    expected = -1.0 / (1 + eps) ** 2 * (1 - eps)
    np.testing.assert_allclose(vals, [expected], rtol=1e-6)


def test_graph_angle_cos_no_self_angles():
    """No graph angle should have both edge slots pointing to the same edge."""
    coord = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]]])
    atype = np.array([[0, 0, 0, 0]])
    ng = attach_angles(build_neighbor_graph(coord, atype, None, 3.0), a_rcut=3.0)
    ai = np.asarray(ng.angle_index)
    am = np.asarray(ng.angle_mask)
    # For every VALID angle, the two edge indices must differ
    real_angles = [(int(ai[0, p]), int(ai[1, p])) for p in range(am.shape[0]) if am[p]]
    for a, b in real_angles:
        assert a != b, f"Self-angle found: edge {a}"


# ---------------------------------------------------------------------------
# dpa3 dense parity
# ---------------------------------------------------------------------------


def _dense_cosine_ij(coord_ext, nlist, a_rcut, a_sel):
    """Faithful numpy transcription of repflows.py:597-649.

    Returns cosine_ij of shape (nf, nloc, a_sel, a_sel).
    a_diff convention: coord_r - coord_l = neighbor - center.
    """
    nf, nloc, nnei = nlist.shape
    # coord_ext: (nf, nall, 3)
    # a_nlist: truncate to a_sel and mask beyond a_rcut
    diff_full = np.zeros((nf, nloc, nnei, 3), dtype=np.float64)
    for f in range(nf):
        for i in range(nloc):
            for k in range(nnei):
                j = nlist[f, i, k]
                if j >= 0:
                    diff_full[f, i, k] = coord_ext[f, j] - coord_ext[f, i]
    # a_rcut gate — clip a_sel to actual nnei columns available
    nnei = nlist.shape[2]
    eff_a_sel = min(a_sel, nnei)
    a_dist_mask = (np.linalg.norm(diff_full, axis=-1) < a_rcut)[:, :, :eff_a_sel]
    a_nlist = nlist[:, :, :eff_a_sel].copy()
    a_nlist = np.where(a_dist_mask, a_nlist, np.full_like(a_nlist, -1))
    # a_diff: shape (nf, nloc, eff_a_sel, 3)
    a_diff = np.zeros((nf, nloc, eff_a_sel, 3), dtype=np.float64)
    for f in range(nf):
        for i in range(nloc):
            for k in range(eff_a_sel):
                j = a_nlist[f, i, k]
                if j >= 0:
                    a_diff[f, i, k] = coord_ext[f, j] - coord_ext[f, i]
    # normalized_diff_i: (nf, nloc, eff_a_sel, 3)
    norm = np.linalg.norm(a_diff, axis=-1, keepdims=True)
    normalized_diff_i = a_diff / (norm + 1e-6)
    # cosine_ij: (nf, nloc, eff_a_sel, eff_a_sel)
    cosine_ij = np.matmul(normalized_diff_i, np.swapaxes(normalized_diff_i, -2, -1))
    cosine_ij = cosine_ij * (1 - 1e-6)
    return cosine_ij, a_nlist


def test_graph_angle_cos_parity_vs_dpa3_dense():
    """Graph unordered/no-self cos values must match dense OFF-DIAGONAL cosine_ij.

    Uses a small 4-atom system with a large a_sel (non-binding) so that the
    graph and dense see the same neighbor set.

    - Tolerance: rtol=1e-12, atol=1e-12 (CPU fp64 same-math, identical eps).
    - The graph is UNORDERED (no duplicates) and NO-SELF.
    - Dense diagonal (j==k, cos≈1) must NOT appear in graph angle set.
    - Dense off-diagonal (j!=k) are collected as unordered {j,k} pairs and
      matched against graph angles by (center_node, unordered edge-src pair).
    """
    rng = np.random.default_rng(42)
    # 4 atoms in a box; no PBC => set box to None
    # Use a single frame, 4 atoms
    nf, nloc = 1, 4
    coord = rng.uniform(-1, 1, (nf, nloc, 3))
    atype = np.zeros((nf, nloc), dtype=np.int32)
    rcut = 3.0
    a_rcut = 2.5
    # Choose a_sel equal to nloc-1 (max neighbors) => non-binding
    a_sel = nloc - 1  # =3; each atom has at most 3 neighbors => non-binding

    # --- graph side ---
    ng = attach_angles(build_neighbor_graph(coord, atype, None, rcut), a_rcut=a_rcut)
    cos_graph = np.asarray(graph_angle_cos(ng.angle_index, ng.edge_vec))
    am = np.asarray(ng.angle_mask)
    ai = np.asarray(ng.angle_index)
    ei = np.asarray(ng.edge_index)  # (2, E): [src, dst]
    ev = np.asarray(ng.edge_vec)

    # No self-angles
    for p in range(am.shape[0]):
        if am[p]:
            assert int(ai[0, p]) != int(ai[1, p]), "Self-angle found in graph"

    # --- dense side ---
    # Build a dense nlist from the same coord (no PBC, full nlist)
    # We construct nlist by brute force
    coord3 = coord[0]  # (nloc, 3)
    # For the dense side, coord_ext = coord (nloc=nall, no ghosts)
    coord_ext = coord  # (1, nloc, 3)
    # Build dense nlist: shape (1, nloc, nnei_max)
    # For 4 atoms, each atom has at most nloc-1=3 neighbors
    nnei = nloc - 1  # max possible neighbors (no self)
    dense_nlist = np.full((nf, nloc, nnei), -1, dtype=np.int64)
    for i in range(nloc):
        k = 0
        for j in range(nloc):
            d = np.linalg.norm(coord3[j] - coord3[i])
            if d < rcut and i != j:
                dense_nlist[0, i, k] = j
                k += 1

    cosine_ij, a_nlist = _dense_cosine_ij(coord_ext, dense_nlist, a_rcut, a_sel)
    # cosine_ij: (1, nloc, a_sel, a_sel)

    # --- match graph angles to dense off-diagonal ---
    # graph edge_index: src=edge_index[0], dst=edge_index[1] (center=dst)
    # For each valid graph angle p: edge_a=ai[0,p], edge_b=ai[1,p]
    # center = ei[1, edge_a] = ei[1, edge_b] (shared center)
    # neighbor_a = ei[0, edge_a], neighbor_b = ei[0, edge_b]

    # Build dense lookup: (center, na, nb) -> cos for off-diagonal
    dense_cos_lookup = {}  # (center, unordered frozenset(na, nb)) -> cos
    for i in range(nloc):
        for j_idx in range(a_sel):
            na = int(a_nlist[0, i, j_idx])
            if na < 0:
                continue
            for k_idx in range(a_sel):
                nb = int(a_nlist[0, i, k_idx])
                if nb < 0:
                    continue
                if j_idx == k_idx:  # skip diagonal (self-angles = edge channel)
                    continue
                cos_val = float(cosine_ij[0, i, j_idx, k_idx])
                key = (i, frozenset([na, nb]))
                # unordered: both (j,k) and (k,j) map to same pair
                # store the value for frozenset key (they should be equal by symmetry
                # of cos, but we verify)
                if key not in dense_cos_lookup:
                    dense_cos_lookup[key] = cos_val
                else:
                    # cosine is symmetric: both directions should be equal
                    np.testing.assert_allclose(
                        dense_cos_lookup[key],
                        cos_val,
                        atol=1e-14,
                        err_msg=f"Dense cosine not symmetric for ({i},{na},{nb})",
                    )

    # Now compare each valid graph angle
    for p in range(am.shape[0]):
        if not am[p]:
            continue
        ea = int(ai[0, p])
        eb = int(ai[1, p])
        center = int(ei[1, ea])
        assert center == int(ei[1, eb]), "Angle edges don't share center"
        na = int(ei[0, ea])
        nb = int(ei[0, eb])
        key = (center, frozenset([na, nb]))
        assert key in dense_cos_lookup, (
            f"Graph angle (center={center}, na={na}, nb={nb}) not in dense"
        )
        cos_g = float(cos_graph[p])
        cos_d = dense_cos_lookup[key]
        np.testing.assert_allclose(
            cos_g,
            cos_d,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"cos mismatch at (center={center}, na={na}, nb={nb})",
        )


# ---------------------------------------------------------------------------
# se_t dot-product cross-check
# ---------------------------------------------------------------------------


def test_matches_se_t_dot_form():
    """Cos * |va| * |vb| ≈ va·vb (the se_t unnormalized inner product).

    se_t.py:428-437 uses `rr_i·rr_j` (unnormalized 3D vectors = edge_vec
    without smoothing weights).  graph_angle_cos(eps=1e-6) satisfies:

        cos ≈ (va/‖va‖) · (vb/‖vb‖) * (1-eps)

    So:
        va·vb ≈ cos * (‖va‖ + eps) * (‖vb‖ + eps) / (1 - eps)

    We verify this identity up to the eps rounding (atol~1e-6 * ‖va‖*‖vb‖).
    """
    coord = np.array([[[0.0, 0, 0], [1.0, 0.5, 0.3], [0.2, 1.0, 0.8]]])
    atype = np.array([[0, 0, 0]])
    ng = attach_angles(build_neighbor_graph(coord, atype, None, 5.0), a_rcut=5.0)
    ai = np.asarray(ng.angle_index)
    am = np.asarray(ng.angle_mask)
    ev = np.asarray(ng.edge_vec)
    cos = np.asarray(graph_angle_cos(ng.angle_index, ng.edge_vec))

    eps = 1e-6
    for p in range(am.shape[0]):
        if not am[p]:
            continue
        ea = int(ai[0, p])
        eb = int(ai[1, p])
        va = ev[ea]
        vb = ev[eb]
        # unnormalized dot product (se_t convention)
        dot_se_t = float(np.dot(va, vb))
        # reconstruct from graph cos
        na_norm = np.linalg.norm(va)
        nb_norm = np.linalg.norm(vb)
        dot_reconstructed = (
            float(cos[p]) * (na_norm + eps) * (nb_norm + eps) / (1 - eps)
        )
        # tolerance: eps * ‖va‖*‖vb‖ dominates
        tol = eps * max(na_norm * nb_norm, 1e-12)
        np.testing.assert_allclose(
            dot_reconstructed,
            dot_se_t,
            atol=tol,
            err_msg=f"se_t dot mismatch for angle {p}: va={va}, vb={vb}",
        )


# ---------------------------------------------------------------------------
# torch namespace smoke test (TID253)
# ---------------------------------------------------------------------------


def test_graph_angle_cos_torch_matches_numpy():
    """graph_angle_cos on torch tensors matches numpy output (array-API compat)."""
    import torch

    coord = np.array([[[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0], [1.0, 1.0, 0]]])
    atype = np.array([[0, 0, 0, 0]])
    ng = attach_angles(build_neighbor_graph(coord, atype, None, 3.0), a_rcut=3.0)
    angle_index_np = np.asarray(ng.angle_index)
    edge_vec_np = np.asarray(ng.edge_vec)

    cos_np = np.asarray(graph_angle_cos(angle_index_np, edge_vec_np))

    angle_index_t = torch.from_numpy(angle_index_np)
    edge_vec_t = torch.from_numpy(edge_vec_np)
    cos_t = graph_angle_cos(angle_index_t, edge_vec_t)
    cos_t_np = cos_t.numpy()

    np.testing.assert_allclose(cos_t_np, cos_np, rtol=1e-14, atol=1e-14)
