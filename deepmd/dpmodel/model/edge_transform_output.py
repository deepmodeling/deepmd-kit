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
    RESERVED_PRECISION_DICT,
    get_xp_precision,
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


def fit_output_to_model_output_graph(
    fit_ret: dict[str, Array],
    fit_output_def: FittingOutputDef,
    graph: NeighborGraph,
    mask: Array | None = None,
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
    # The configured energy precision is represented by a NumPy dtype class,
    # which is not accepted as a dtype by every array namespace (notably Torch).
    energy_dtype = get_xp_precision(
        xp, RESERVED_PRECISION_DICT[GLOBAL_ENER_FLOAT_PRECISION]
    )
    nf = n_node.shape[0]
    frame_id = frame_id_from_n_node(n_node)
    model_ret = dict(fit_ret.items())
    for kk, vv in fit_ret.items():
        vdef = fit_output_def[kk]
        if not vdef.reducible:
            continue
        kk_redu = get_reduce_name(kk)
        vv_e = xp.astype(vv, energy_dtype)
        redu = segment_sum(vv_e, frame_id, nf)  # (nf, *shape)
        if vdef.intensive:
            if mask is not None:
                cnt = segment_sum(xp.astype(mask, energy_dtype), frame_id, nf)
            else:
                cnt = xp.astype(n_node, energy_dtype)
            redu = redu / xp.reshape(cnt, (nf, *([1] * (redu.ndim - 1))))
        model_ret[kk_redu] = redu
        if vdef.r_differentiable:
            kk_derv_r, _ = get_deriv_name(kk)
            model_ret[kk_derv_r] = None
        if vdef.c_differentiable:
            _, kk_derv_c = get_deriv_name(kk)
            model_ret[kk_derv_c] = None
    return model_ret
