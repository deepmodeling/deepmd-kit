# SPDX-License-Identifier: LGPL-3.0-or-later
"""Default all-pairs neighbor-list builder (historical deepmd behavior)."""

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.utils.neighbor_list import (
    EdgeNeighborList,
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

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask


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
        return_mode: str = "extended",
        pair_excl: "PairExcludeMask | None" = None,
    ) -> tuple[Array, Array, Array, Array] | EdgeNeighborList:
        """Build extended coordinates and a candidate neighbor list.

        Parameters
        ----------
        coord : Array
            Local coordinates, shape ``(nf, nloc, 3)`` or ``(nf, nloc*3)``.
        atype : Array
            Local atom types, shape ``(nf, nloc)``.
        box : Array or None
            Simulation cell, shape ``(nf, 3, 3)`` or ``(nf, 9)``; ``None``
            for non-periodic systems.
        rcut : float
            Cutoff radius.
        sel : list[int]
            Number of selected neighbors per type.
        return_mode : str
            Must be ``"extended"`` (the only mode this builder supports).
        pair_excl : PairExcludeMask or None, optional
            When provided, excluded type pairs are erased from the returned
            neighbor list immediately after the geometric search by
            :func:`~deepmd.dpmodel.utils.nlist.build_neighbor_list`.

        Returns
        -------
        tuple[Array, Array, Array, Array]
            ``(extended_coord, extended_atype, nlist, mapping)`` as documented
            in :meth:`~deepmd.dpmodel.utils.neighbor_list.NeighborList.build`.
        """
        if return_mode != "extended":
            raise NotImplementedError(
                "DefaultNeighborList only supports the extended-coordinate contract."
            )
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
        # types are distinguished in the lower interface, so keep them merged here;
        # pair_excl is forwarded so exclusion is applied at build time.
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=False,
            pair_excl=pair_excl,
        )
        extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
        return extended_coord, extended_atype, nlist, mapping
