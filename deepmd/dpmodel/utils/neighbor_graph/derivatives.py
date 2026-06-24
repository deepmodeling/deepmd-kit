# SPDX-License-Identifier: LGPL-3.0-or-later
"""Assemble per-node force and virial from a per-edge gradient g_e = dE/d(edge_vec).

The autograd that produces g_e (grad(E, edge_vec)) is wired in the torch/jax
backend later; this pure-array-API assembly is shared by all backends.

Conventions (see memory/spec_unified_edge_nlist.md):
  edge_vec_e = r_src - r_dst ;  F_k = sum_{dst=k} g - sum_{src=k} g
  per-edge virial w_e = -g_e (x) edge_vec_e
  atom virial attributed FULL-TO-src (canonical TF==pt-legacy convention)
  global virial = sum_e w_e
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
    n_node_total: int,
) -> tuple[Array, Array, Array]:
    """Returns (force (N,3), atom_virial (N,3,3), global_virial (3,3))."""
    xp = array_api_compat.array_namespace(g_e)
    # zero padding/guard contributions; cast mask to g's dtype (array-API pure,
    # CLAUDE.md mask-multiply guideline — avoids bool*float under array_api_strict)
    g = g_e * xp.astype(edge_mask[:, None], g_e.dtype)
    src = edge_index[0]
    dst = edge_index[1]
    # force
    force = segment_sum(g, dst, n_node_total) - segment_sum(g, src, n_node_total)
    # per-edge virial w_e[k, j] = -g_e[k] * edge_vec[j]  (broadcast, no einsum)
    w_edge = -(g[:, :, None] * edge_vec[:, None, :])  # (E, 3, 3)
    # atom virial: full-to-src
    atom_virial = segment_sum(w_edge, src, n_node_total)  # (N, 3, 3)
    global_virial = xp.sum(w_edge, axis=0)  # (3, 3)
    return force, atom_virial, global_virial
