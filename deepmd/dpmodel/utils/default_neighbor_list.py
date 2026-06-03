# SPDX-License-Identifier: LGPL-3.0-or-later
"""Default all-pairs neighbor-list builder (historical deepmd behavior)."""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)

from .neighbor_list import (
    NeighborList,
)
from .nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from .region import (
    normalize_coord,
)


class DefaultNeighborList(NeighborList):
    """All-pairs builder: replicate the cell into periodic images and rank by
    distance (:func:`~deepmd.dpmodel.utils.nlist.extend_coord_with_ghosts` +
    :func:`~deepmd.dpmodel.utils.nlist.build_neighbor_list`).  This is the
    default when no strategy is supplied, so results are unchanged.
    """

    def build(
        self,
        coord: Array,
        atype: Array,
        box: Array | None,
        rcut: float,
        sel: list[int],
    ) -> tuple[Array, Array, Array, Array]:
        xp = array_api_compat.array_namespace(coord, atype)
        nframes, nloc = atype.shape[:2]
        if box is not None:
            coord_normalized = normalize_coord(
                xp.reshape(coord, (nframes, nloc, 3)),
                xp.reshape(box, (nframes, 3, 3)),
            )
        else:
            coord_normalized = xp.reshape(coord, (nframes, nloc, 3))
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype, box, rcut
        )
        # types are distinguished in the lower interface, so keep them merged here
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=False,
        )
        extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
        return extended_coord, extended_atype, nlist, mapping
