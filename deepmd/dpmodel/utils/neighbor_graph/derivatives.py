# SPDX-License-Identifier: LGPL-3.0-or-later
"""Assemble per-node force and virial from a per-edge gradient g_e = dE/d(edge_vec).

The autograd that produces g_e (grad(E, edge_vec)) is wired in the torch/jax
backend later; this pure-array-API assembly is shared by all backends.

Conventions (see memory/spec_unified_edge_nlist.md):
  edge_vec_e = r_src - r_dst ;  F_k = sum_{dst=k} g - sum_{src=k} g
  per-edge virial w_e = -g_e (x) edge_vec_e
  atom virial attributed FULL-TO-src (canonical TF==pt-legacy convention)
  per-frame virial = sum over the edges of that frame of w_e  (DeePMD virials
  are per frame; a multi-frame NeighborGraph must NOT collapse frames)
Padding/guard edges (edge_mask == 0) are zeroed before any scatter.
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)

from .segment import (
    segment_sum,
)


def edge_force_virial(
    g_e: Array,
    edge_vec: Array,
    edge_index: Array,
    edge_mask: Array,
    n_node: Array,
) -> tuple[Array, Array, Array]:
    """Assemble per-node force/atom-virial and PER-FRAME virial from ``g_e``.

    Parameters
    ----------
    n_node
        (nf,) per-frame node counts. The flat node axis is ``N = sum(n_node)``
        (compact, frame-major); ``nf = n_node.shape[0]``.

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
    n_total = int(xp.sum(n_node))  # flat node count N (static shape)
    nf = n_node.shape[0]
    # zero padding/guard contributions; cast mask to g's dtype (array-API pure,
    # CLAUDE.md mask-multiply guideline — avoids bool*float under array_api_strict)
    g = g_e * xp.astype(edge_mask[:, None], g_e.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    # force
    force = segment_sum(g, dst, n_total) - segment_sum(g, src, n_total)
    # per-edge virial w_e[k, j] = -g_e[k] * edge_vec[j]  (broadcast, no einsum)
    w_edge = -(g[:, :, None] * edge_vec[:, None, :])  # (E, 3, 3)
    # atom virial: full-to-src
    atom_virial = segment_sum(w_edge, src, n_total)  # (N, 3, 3)
    # per-frame virial: assign each edge to the frame of its dst node. Node
    # ``k`` belongs to frame ``searchsorted(cumsum(n_node), k, "right")`` because
    # the node axis is compact frame-major (frame f owns a contiguous block).
    boundaries = xp.cumulative_sum(n_node)  # (nf,) per-frame node upper bounds
    edge_frame = xp.astype(
        xp.searchsorted(boundaries, dst, side="right"), xp.int64
    )  # (E,) in [0, nf)
    virial = segment_sum(w_edge, edge_frame, nf)  # (nf, 3, 3)
    return force, atom_virial, virial
