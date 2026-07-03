# SPDX-License-Identifier: LGPL-3.0-or-later
"""3-body angle graph: pairs of edges sharing a center within a_rcut.

Angles reference EDGES (angle_index into [0,E)); edge_vec stays the only
geometry leaf. a_sel is normalization-only (not a truncation). Reuses PR-D's
center_edge_pairs; a_rcut filters the participating edges.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import array_api_compat

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import Array

from .graph import GraphLayout, pad_and_guard_angles
from .pairs import center_edge_pairs


def build_angle_index(
    edge_index: Array,
    edge_vec: Array,
    edge_mask: Array,
    n_total: int,
    a_rcut: float,
    *,
    ordered: bool = False,
    include_self: bool = False,
    layout: GraphLayout | None = None,
) -> tuple[Array, Array]:
    """Build angle index for 3-body terms.

    Parameters
    ----------
    edge_index : Array
        Shape (2, E) [src, dst] SoA edge indices.
    edge_vec : Array
        Shape (E, 3) edge vectors (neighbor - center).
    edge_mask : Array
        Shape (E,) boolean validity mask for edges.
    n_total : int
        Total number of nodes.
    a_rcut : float
        Angle cutoff. Only edges with norm < a_rcut participate in angles.
    ordered : bool, optional
        If True, include both (a, b) and (b, a) pairs (ordered pairs).
    include_self : bool, optional
        If True, include self-angle pairs (a, a).
    layout : GraphLayout or None, optional
        If provided, uses layout.angle_capacity as static padding capacity.

    Returns
    -------
    angle_index : Array
        Shape (2, A) index pairs into the edge list.
    angle_mask : Array
        Shape (A,) boolean mask for valid angles.
    """
    xp = array_api_compat.array_namespace(edge_index)
    # a_rcut edge gate: only edges within a_rcut may participate in an angle
    dist = xp.linalg.vector_norm(edge_vec, axis=-1)  # (E,)
    a_edge_mask = xp.astype(edge_mask, xp.bool) & (dist < a_rcut)
    # compact eager form only (static_nnei not exposed until angle export is
    # needed, PR-G). dst = edge_index[1, :] per the [src, dst] SoA convention.
    q_e, k_e, pair_mask = center_edge_pairs(
        edge_index[1, :],
        a_edge_mask,
        n_total,
        include_self=include_self,
        ordered=ordered,
    )
    # compact form returns all-True pair_mask, but NEVER discard it: the
    # shape-static form keeps filtered pairs and invalidates them only here.
    angle_index = xp.stack([q_e, k_e], axis=0)  # (2, A_real)
    cap = layout.angle_capacity if layout is not None else None
    ai, am = pad_and_guard_angles(angle_index, cap, min_angles=2)
    # fold pair_mask into the real-angle prefix of the padded mask
    pm_padded = xp.concat(
        [
            pair_mask,
            xp.zeros(
                (am.shape[0] - pair_mask.shape[0],),
                dtype=xp.bool,
                device=array_api_compat.device(pair_mask),
            ),
        ],
        axis=0,
    )
    return ai, am & pm_padded
