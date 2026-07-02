# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all NeighborGraph builder backed by vesin.torch (O(N) cell list).

World-2 counterpart of vesin_neighbor_list.py: instead of building the dense
quartet, it returns per-frame local (i, j, S), then delegates to the array-API
``neighbor_graph_from_ijs`` (which recomputes ``edge_vec`` differentiably from
the ORIGINAL grad-carrying coords). torch-only => lives in pt_expt.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import array_api_compat
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    neighbor_graph_from_ijs,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)


def build_neighbor_graph_vesin(
    coord: Any,
    atype: Any,
    box: Any | None,
    rcut: float,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Build a CARRY-ALL NeighborGraph using vesin.torch's O(N) cell list.

    Mirrors :func:`deepmd.dpmodel.utils.neighbor_graph.build_neighbor_graph_ase`
    but runs on the input tensor's device via ``vesin.torch``.
    """
    if not is_vesin_torch_available():
        raise ImportError(
            "build_neighbor_graph_vesin requires vesin[torch]; "
            "install with `pip install vesin[torch]` or use neighbor_graph_method='dense'."
        )
    import vesin.torch

    xp = array_api_compat.array_namespace(coord)
    dev = array_api_compat.device(coord)
    nf = coord.shape[0] if coord.ndim == 3 else 1
    coord = xp.reshape(coord, (nf, -1, 3))
    nloc = coord.shape[1]
    periodic = box is not None
    if periodic:
        box = xp.reshape(box, (nf, 3, 3))

    if nloc == 0:
        empty_i = torch.zeros((0,), dtype=torch.int64, device=dev)
        empty_S = torch.zeros((0, 3), dtype=torch.int64, device=dev)
        empty_nf = torch.zeros((0,), dtype=torch.int64, device=dev)
        return neighbor_graph_from_ijs(
            empty_i, empty_i, empty_S, coord, box, empty_nf, nloc, layout=layout
        )

    i_parts, j_parts, S_parts, nf_parts = [], [], [], []
    nl = vesin.torch.NeighborList(cutoff=float(rcut), full_list=True)
    for f in range(nf):
        pts = coord[f].detach()
        with torch.device(dev):
            ii, jj, ss = nl.compute(
                points=pts,
                box=(
                    box[f].detach()
                    if periodic
                    else torch.zeros((3, 3), dtype=pts.dtype, device=dev)
                ),
                periodic=periodic,
                quantities="ijS",
            )
        ii = ii.to(torch.int64)
        jj = jj.to(torch.int64)
        ss = ss.to(torch.int64).reshape(-1, 3)
        i_parts.append(ii)
        j_parts.append(jj)
        S_parts.append(ss)
        nf_parts.append(torch.full((ii.shape[0],), f, dtype=torch.int64, device=dev))

    i_all = torch.cat(i_parts)
    j_all = torch.cat(j_parts)
    S_all = torch.cat(S_parts)
    nf_all = torch.cat(nf_parts)

    # i = center (dst), j = neighbor (src); pass ORIGINAL coord/box (grad-carrying).
    return neighbor_graph_from_ijs(
        i_all, j_all, S_all, coord, box, nf_all, nloc, layout=layout
    )
