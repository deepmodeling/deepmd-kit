# SPDX-License-Identifier: LGPL-3.0-or-later
"""Flat-N (ragged-native) graph output transform for the dpmodel backend.

The graph lower produces per-node outputs on the flat ``(N,)`` node axis
(``N = sum(graph.n_node)``); this reduces every reducible fitting output per
frame via ``segment_sum`` over ``frame_id``.  dpmodel is energy-only (no
autograd force on the graph path), so derivative name-holders are ``None`` --
the pt_expt backend (:mod:`deepmd.pt_expt.model.edge_transform_output`) assembles
force/virial from the same ``NeighborGraph`` via ``edge_energy_deriv``.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
)
from deepmd.dpmodel.output_def import (
    get_deriv_name,
    get_reduce_name,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.dpmodel.output_def import (
        FittingOutputDef,
    )
    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )


def node_ownership_mask(n_node: Array, n_local: Array, n_total: int) -> Array:
    """Derive the ``(n_total,)`` owned-node mask from per-frame local counts.

    Owned-prefix layout: frame ``f`` owns the first ``n_local[f]`` of its
    contiguous ``n_node[f]``-node block (the remainder, if any, are ghost/
    ghost nodes owned by another rank). Local helper matching the (not yet
    merged) ``#5758`` name/semantics so a later rebase converges.

    Parameters
    ----------
    n_node
        (nf,) per-frame REAL (local + ghost) node counts.
    n_local
        (nf,) per-frame OWNED node counts, ``n_local[f] <= n_node[f]``.
    n_total
        Size of the flat node axis ``N`` (``== sum(n_node)`` when unpadded).

    Returns
    -------
    mask
        (n_total,) boolean mask, ``True`` for nodes owned by this rank.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
    )

    xp = array_api_compat.array_namespace(n_node, n_local)
    device = array_api_compat.device(n_node)
    node_index = xp.arange(n_total, dtype=n_node.dtype, device=device)
    frame_id = frame_id_from_n_node(n_node, n_total=n_total)
    frame_end = xp.cumulative_sum(n_node)
    frame_start = frame_end - n_node
    index_in_frame = node_index - xp.take(frame_start, frame_id, axis=0)
    local_count = xp.take(n_local, frame_id, axis=0)
    return index_in_frame < local_count


def fit_output_to_model_output_graph(
    fit_ret: dict[str, Array],
    fit_output_def: FittingOutputDef,
    graph: NeighborGraph,
    mask: Array | None = None,
    n_local: Array | None = None,
) -> dict[str, Array]:
    """Flat-N analogue of :func:`~deepmd.dpmodel.model.transform_output.fit_output_to_model_output`.

    Parameters
    ----------
    fit_ret
        the raw per-node fitting dict, each value ``(N, *shape)``.
    fit_output_def
        the fitting output def (drives the per-key reduction).
    graph
        the :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph`; only
        ``graph.n_node`` is used (the node->frame map for the reduction).
    mask
        the ``(N,)`` real-node mask for the intensive-output denominator.
    n_local
        ``(nf,)`` per-frame OWNED node counts for multi-rank ghost exclusion
        (owned-prefix layout, :func:`node_ownership_mask`). When given, every
        reducible per-node value is masked to zero on ghost rows (index
        ``>= n_local[frame]``) BEFORE the per-frame ``segment_sum`` -- each
        ghost atom is owned (and counted) on another rank, so its contribution
        must not double-count here. The per-node output (``<var>``) itself is
        left FULL/unmasked (the C++ caller slices owned rows itself; ghost
        partial forces are reverse-commed by LAMMPS). ``None`` (default):
        unchanged single-rank behavior.

    Returns
    -------
    model_ret
        ``fit_ret`` plus, for each reducible key, ``<var>_redu (nf, *shape)`` via
        ``segment_sum`` over ``frame_id`` (intensive ⇒ divide by the per-frame
        real-node count); derivative name-holders are ``None``.
    """
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
        segment_sum,
    )

    n_node = graph.n_node
    xp = array_api_compat.get_namespace(n_node)
    nf = n_node.shape[0]
    frame_id = frame_id_from_n_node(n_node)
    n_total = next(iter(fit_ret.values())).shape[0]
    owned = (
        node_ownership_mask(n_node, n_local, n_total) if n_local is not None else None
    )
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        if not vdef.reducible:
            continue
        kk_redu = get_reduce_name(kk)
        vv_e = xp.astype(vv, GLOBAL_ENER_FLOAT_PRECISION)
        if owned is not None:
            owned_e = xp.astype(owned, GLOBAL_ENER_FLOAT_PRECISION)
            vv_e = vv_e * xp.reshape(owned_e, (n_total, *([1] * (vv_e.ndim - 1))))
        redu = segment_sum(vv_e, frame_id, nf)  # (nf, *shape)
        if vdef.intensive:
            if mask is not None:
                cnt_mask = xp.astype(mask, GLOBAL_ENER_FLOAT_PRECISION)
                if owned is not None:
                    cnt_mask = cnt_mask * owned_e
                cnt = segment_sum(cnt_mask, frame_id, nf)
            elif owned is not None:
                cnt = segment_sum(owned_e, frame_id, nf)
            else:
                cnt = xp.astype(n_node, GLOBAL_ENER_FLOAT_PRECISION)
            redu = redu / xp.reshape(cnt, (nf, *([1] * (redu.ndim - 1))))
        model_ret[kk_redu] = redu
        if vdef.r_differentiable:
            kk_derv_r, _ = get_deriv_name(kk)
            model_ret[kk_derv_r] = None
        if vdef.c_differentiable:
            _, kk_derv_c = get_deriv_name(kk)
            model_ret[kk_derv_c] = None
    return model_ret
