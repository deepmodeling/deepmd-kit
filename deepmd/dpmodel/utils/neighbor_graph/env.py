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
    edge_mask: Array | None = None,
    return_sw: bool = False,
) -> Array | tuple[Array, Array]:
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
    edge_mask
        (E,) boolean valid-edge mask. When provided, the length of INVALID
        (padding) edges has 1 added to it before adding ``protection`` ---
        matching the dense ``_make_env_mat`` (``length = length + ~mask``),
        which guards padding by mask rather than by a length threshold.
        When ``None``, fall back to the ``length < 1e-10`` zero-guard
        (back-compat for callers without a mask).

    Returns
    -------
    Array
        (E, 4) normalized environment-matrix vectors.
        Padding edges (``edge_vec = 0``) produce nonzero values but are
        masked by ``NeighborGraph.edge_mask`` downstream.
        When ``return_sw`` is True, returns ``(em, sw)`` where ``sw`` is the
        (E, 1) smooth switch, zeroed on padding edges (mirrors the dense
        ``_make_env_mat`` mask; consumed by the smooth attention branch).
    """
    xp = array_api_compat.array_namespace(edge_vec)
    dev = array_api_compat.device(edge_vec)

    # ── geometry ───────────────────────────────────────────────────────────
    # (E, 1) lengths; safe_for_vector_norm returns 0 for zero vectors
    length = safe_for_vector_norm(edge_vec, axis=-1, keepdims=True)

    # Guard against 1/0 on padding edges.  When an edge_mask is provided,
    # match the dense _make_env_mat exactly: add 1 to the length of INVALID
    # (padding) edges by mask (not by a length threshold), so a real edge and
    # a padding edge never share the same protection arithmetic.  Otherwise
    # fall back to the length<1e-10 zero-guard (back-compat).
    if edge_mask is not None:
        length = length + xp.astype(xp.logical_not(edge_mask)[:, None], length.dtype)
    else:
        length = xp.where(length < 1e-10, xp.ones_like(length), length)

    denom = length + protection  # (E, 1)
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

    if return_sw:
        # per-edge switch, zeroed on padding edges — mirrors the dense
        # ``_make_env_mat`` (``weight = weight * mask``); used by the smooth
        # attention branch.
        if edge_mask is not None:
            sw_out = sw * xp.astype(edge_mask[:, None], sw.dtype)
        else:
            sw_out = sw
        return (em - avg) / std, sw_out
    return (em - avg) / std
