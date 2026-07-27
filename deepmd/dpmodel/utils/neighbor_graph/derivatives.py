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

from .graph import (
    frame_id_from_n_node,
)
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
    # A supplied capacity remains symbolic under export; converting it to int
    # would specialize the dynamic node axis to the trace sample.
    n_out = node_capacity if node_capacity is not None else int(xp.sum(n_node))
    nf = n_node.shape[0]
    if isinstance(n_out, int) and n_out == 0:
        device = array_api_compat.device(g_e)
        return (
            xp.zeros((0, 3), dtype=g_e.dtype, device=device),
            xp.zeros((0, 3, 3), dtype=g_e.dtype, device=device),
            xp.zeros((nf, 3, 3), dtype=g_e.dtype, device=device),
        )
    # Padding edges carry no force or virial contribution.
    g = g_e * xp.astype(edge_mask[:, None], g_e.dtype)
    # Real endpoints are already in range. Modulo leaves them unchanged while
    # bounding masked sentinel endpoints after export removes symbolic shape
    # equalities between the endpoint and output node axes.
    src = edge_index[0] % n_out
    dst = edge_index[1] % n_out
    # force (output sized to the node axis, incl. any padding tail)
    force = segment_sum(g, dst, n_out) - segment_sum(g, src, n_out)
    # per-edge virial w_e[k, j] = -g_e[k] * edge_vec[j]  (broadcast, no einsum)
    w_edge = -(g[:, :, None] * edge_vec[:, None, :])  # (E, 3, 3)
    # atom virial: full-to-src
    atom_virial = segment_sum(w_edge, src, n_out)  # (N, 3, 3)
    # Per-frame virial: reduce the PER-ATOM virial by its node frame (N -> nf)
    # rather than the per-edge virial by its edge frame (E -> nf). Real edges
    # never cross frames (``src`` and ``dst`` share a frame), so summing the
    # full-to-``src`` atom virial over a frame's nodes equals summing ``w_edge``
    # over that frame's edges. Reducing the N nodes instead of the E >> N edges
    # avoids serializing ~E scatter-adds into a single per-frame accumulator --
    # the single-frame (inference) worst case, where every edge targets one slot
    # and the scatter degenerates into a fully contended atomic reduction.
    node_frame = frame_id_from_n_node(n_node, n_total=n_out)  # (N,) frame per node
    virial = segment_sum(atom_virial, node_frame, nf)  # (nf, 3, 3)
    return force, atom_virial, virial
