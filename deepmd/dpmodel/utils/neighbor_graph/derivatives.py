# SPDX-License-Identifier: LGPL-3.0-or-later
"""Assemble per-node force and virial from a per-edge gradient g_e = dE/d(edge_vec).

The autograd that produces g_e (grad(E, edge_vec)) is wired in the torch/jax
backend later; this pure-array-API assembly is shared by all backends.

Conventions (see the unified edge-nlist design discussion, wanghan-iapcm/deepmd-kit#4):
  edge_vec_e = r_src - r_dst ;  F_k = sum_{dst=k} g - sum_{src=k} g
  per-edge virial w_e = -g_e (x) edge_vec_e
  atom virial attributed FULL-TO-src (canonical TF==pt-legacy convention)
  per-frame virial = sum over the edges of that frame of w_e  (DeePMD virials
  are per frame; a multi-frame NeighborGraph must NOT collapse frames)
Padding/guard edges (edge_mask == 0) are zeroed before any scatter.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from .segment import (
    segment_sum,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )


def edge_force_virial(
    g_e: Array,
    edge_vec: Array,
    edge_index: Array,
    edge_mask: Array,
    n_node: Array,
    node_capacity: int | None = None,
) -> tuple[Array, Array, Array]:
    """Assemble per-node force/atom-virial and PER-FRAME virial from ``g_e``.

    Handles the fully general layout: multi-frame, RAGGED (different per-frame
    node and edge counts), padding/guard EDGES (``edge_mask == 0``), and a padded
    NODE axis (``node_capacity`` > ``sum(n_node)``).

    Parameters
    ----------
    g_e
        (E, 3) per-edge gradient ``dE/d(edge_vec)``.
    edge_vec
        (E, 3) per-edge displacement ``r_src - r_dst``; padding edges are zero.
    edge_index
        (2, E) ``[src, dst]`` node endpoints of each edge.
    edge_mask
        (E,) boolean valid-edge mask; padding/guard edges (``False``) are zeroed
        before any scatter.
    n_node
        (nf,) per-frame REAL node counts. Real nodes occupy the compact prefix
        ``[0, sum(n_node))`` frame-major; ``nf = n_node.shape[0]``.
    node_capacity
        Size of the (possibly padded) node axis ``N``. ``None`` => ``sum(n_node)``
        (no node padding — the torch/eager case). When set (jax static ``N_max``),
        force/atom_virial are sized to it; padding nodes (never referenced by an
        edge) get zero. Frame assignment is unaffected (padding nodes are not
        ``dst`` of any real edge).

    Returns
    -------
    force
        (N, 3) per-node force.
    atom_virial
        (N, 3, 3) per-node virial, full-to-``src`` attribution.
    virial
        (nf, 3, 3) PER-FRAME virial. A multi-frame graph keeps each frame's
        virial separate (DeePMD virials are per frame); edges are assigned to a
        frame via the frame of their ``dst`` node.
    """
    xp = array_api_compat.array_namespace(g_e)
    # node-axis size; when a ``node_capacity`` is supplied (the jax/export path)
    # use it AS-IS so we never call int() on the traced ``sum(n_node)`` -- and,
    # crucially, never on ``node_capacity`` itself: under symbolic make_fx /
    # torch.export it is a SymInt (``atype.shape[0]``); ``int(SymInt)`` would
    # SPECIALIZE the node axis to the trace-time sample size, baking a constant
    # ``N`` into the scatter and breaking dynamic-``N`` inference.
    n_out = node_capacity if node_capacity is not None else int(xp.sum(n_node))
    nf = n_node.shape[0]
    # zero padding/guard contributions; cast mask to g's dtype (array-API pure,
    # CLAUDE.md mask-multiply guideline — avoids bool*float under array_api_strict)
    g = g_e * xp.astype(edge_mask[:, None], g_e.dtype)
    # Clamp scatter indices into the valid node range ``[0, n_out)``. Padding/guard
    # edges (``edge_mask == 0``) carry ``g == 0`` above, so ``w_edge == 0`` and a
    # clamped out-of-range index scatters ZERO -- numerically harmless. This keeps
    # the scatter address in-bounds for the CUDA-compiled kernel: under dynamic-edge
    # ``torch.export`` a padding index can reach the ``index_add`` BEFORE the mask
    # zeroes its value, tripping ``tl.device_assert(idx < ks0)`` (a hard device-side
    # assert on CUDA; benign on CPU, which does not bounds-check the address).
    src = xp.clip(edge_index[0], 0, n_out - 1)
    dst = xp.clip(edge_index[1], 0, n_out - 1)
    # force (output sized to the node axis, incl. any padding tail)
    force = segment_sum(g, dst, n_out) - segment_sum(g, src, n_out)
    # per-edge virial w_e[k, j] = -g_e[k] * edge_vec[j]  (broadcast, no einsum)
    w_edge = -(g[:, :, None] * edge_vec[:, None, :])  # (E, 3, 3)
    # atom virial: full-to-src
    atom_virial = segment_sum(w_edge, src, n_out)  # (N, 3, 3)
    # per-frame virial: assign each edge to the frame of its dst node. Node
    # ``k`` belongs to frame ``searchsorted(cumsum(n_node), k, "right")`` because
    # real nodes are compact frame-major (frame f owns a contiguous block).
    boundaries = xp.cumulative_sum(n_node)  # (nf,) per-frame node upper bounds
    edge_frame = xp.astype(
        xp.searchsorted(boundaries, dst, side="right"), xp.int64
    )  # (E,) in [0, nf)
    # searchsorted(side="right") can return ``nf`` for an out-of-range ``dst``
    # (padding/garbage); clamp into ``[0, nf)`` for the same CUDA-bounds reason.
    edge_frame = xp.clip(edge_frame, 0, nf - 1)
    virial = segment_sum(w_edge, edge_frame, nf)  # (nf, 3, 3)
    return force, atom_virial, virial
