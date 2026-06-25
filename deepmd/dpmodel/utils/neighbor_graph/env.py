# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-edge environment-matrix 4-vector, the graph-native analogue of
EnvMat.call (deepmd/dpmodel/utils/env_mat.py).

Computes, per edge, [1/r, dx/r^2, dy/r^2, dz/r^2] * smooth_weight, then
normalizes by (davg, dstd) indexed by the edge's CENTER (dst) atom type.
Stats are (ntypes, 4) — slot-independent — which is valid because
EnvMatStatSe tiles a single per-type vector across all nnei slots
(``np.tile(davgunit, [nsel, 1])``), so the slot axis carries no information.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from deepmd.dpmodel.utils.env_mat import (
    compute_smooth_weight,
)
from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )


def edge_env_mat(
    edge_vec: Array,
    center_type: Array,
    davg: Array,
    dstd: Array,
    rcut: float,
    rcut_smth: float,
    protection: float = 0.0,
) -> Array:
    """Compute the per-edge environment-matrix 4-vector.

    Mirrors the math in ``_make_env_mat`` / ``EnvMat.call`` (env_mat.py)
    for a single edge batch instead of a dense (nf, nloc, nnei) tensor.

    Parameters
    ----------
    edge_vec
        (E, 3) displacement vectors r_src - r_dst (neighbor minus center);
        padding edges must have ``edge_vec = 0``.
    center_type
        (E,) int — atom type of the center (dst) atom for each edge.
    davg
        (ntypes, 4) per-center-type mean (slot-independent).
    dstd
        (ntypes, 4) per-center-type inverse-std (slot-independent).
    rcut
        Outer cutoff radius.
    rcut_smth
        Inner radius where the smooth switch begins.
    protection
        Small additive offset to avoid exact division-by-zero on
        atoms that are numerically at the same position (default 0).

    Returns
    -------
    Array
        (E, 4) normalized environment-matrix vectors.
        Padding edges (``edge_vec = 0``) produce nonzero values but are
        masked by ``NeighborGraph.edge_mask`` downstream.
    """
    xp = array_api_compat.array_namespace(edge_vec)
    dev = array_api_compat.device(edge_vec)

    # ── geometry ───────────────────────────────────────────────────────────
    # (E, 1) lengths; safe_for_vector_norm returns 0 for zero vectors
    length = safe_for_vector_norm(edge_vec, axis=-1, keepdims=True)

    # Guard against exact zero to avoid 1/0 (happens on padding edges where
    # edge_vec = 0).  Real edges always have length > 0.
    safe_len = xp.where(length < 1e-10, xp.ones_like(length), length)

    denom = safe_len + protection  # (E, 1)
    t0 = 1.0 / denom  # (E, 1)  — radial component
    t1 = edge_vec / (denom**2)  # (E, 3) — angular components

    # ── smooth switch (same polynomial as compute_smooth_weight) ───────────
    # length has shape (E, 1); compute_smooth_weight broadcasts over any shape
    sw = compute_smooth_weight(length, rcut_smth, rcut)  # (E, 1)

    # ── raw (unnormalized) env-mat ─────────────────────────────────────────
    em = xp.concat([t0, t1], axis=-1) * sw  # (E, 4)

    # ── per-type normalization (indexed by center-atom type) ───────────────
    # davg/dstd must be asarray'd to ensure device placement when called with
    # numpy stats on a torch/jax edge_vec.
    avg = xp.take(xp.asarray(davg, device=dev), center_type, axis=0)  # (E, 4)
    std = xp.take(xp.asarray(dstd, device=dev), center_type, axis=0)  # (E, 4)

    return (em - avg) / std
