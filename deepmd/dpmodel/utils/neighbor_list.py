# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pluggable neighbor-list construction strategies.

A :class:`NeighborList` turns local coordinates (and an optional cell) into the
*extended* representation consumed by the model's lower interface.  The default
all-pairs builder lives in :mod:`deepmd.dpmodel.utils.default_neighbor_list`;
backend-specific O(N) builders (e.g. the ``vesin``-based one in
``deepmd.pt_expt.utils.vesin_neighbor_list``) subclass :class:`NeighborList`
and are injected into the model, so the rest of the model is agnostic to how the
neighbor list was built.
"""

from deepmd.dpmodel.array_api import (
    Array,
)


class NeighborList:
    """Strategy that builds the extended neighbor environment from local atoms.

    Implementations turn local coordinates into the extended representation: the
    coordinates and atom types of local-plus-ghost (periodic-image) atoms, a
    candidate neighbor list indexing the extended atoms, and a mapping from each
    extended atom to its local owner.  Implementations are stateless --
    ``rcut``/``sel`` are supplied by the model at call time.
    """

    def build(
        self,
        coord: Array,
        atype: Array,
        box: Array | None,
        rcut: float,
        sel: list[int],
    ) -> tuple[Array, Array, Array, Array]:
        """Build the extended system and a candidate neighbor list.

        Parameters
        ----------
        coord
            local coordinates, shape (nf, nloc, 3) or (nf, nloc*3).
        atype
            local atom types, shape (nf, nloc).
        box
            simulation cell, shape (nf, 3, 3) or (nf, 9); ``None`` for non-periodic.
        rcut
            cutoff radius.
        sel
            number of selected neighbors per type.

        Returns
        -------
        extended_coord
            shape (nf, nall, 3).
        extended_atype
            shape (nf, nall).
        nlist
            shape (nf, nloc, nnei), type-undistinguished candidate neighbors
            indexing the extended atoms (the lower interface re-formats it:
            distance sort, truncate to ``sel``, split by type).
        mapping
            shape (nf, nall), mapping each extended atom to its local owner.
        """
        raise NotImplementedError
