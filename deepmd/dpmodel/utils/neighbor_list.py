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

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Literal,
)

from deepmd.dpmodel.array_api import (
    Array,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask


@dataclass
class EdgeNeighborList:
    """Edge-vector neighbor-list contract.

    The model consumes geometry only through ``edge_vec``.  Builders that own
    periodic-image shifts compute those shifts before constructing this object;
    callers that receive already-shifted ghost coordinates use zero-shift edge
    vectors computed from the provided coordinates.
    """

    coord: Array
    """Coordinates of the scatter domain with shape ``(nf, nscatter, 3)``."""

    atype: Array
    """Local owner atom types with shape ``(nf, nloc)``."""

    edge_index: Array
    """Message-passing edge indices with shape ``(2, nedge)`` in owner space."""

    edge_vec: Array
    """Per-edge displacement vectors with shape ``(nedge, 3)``."""

    edge_scatter_index: Array
    """Force/virial scatter indices with shape ``(2, nedge)`` in scatter space."""

    edge_mask: Array
    """Boolean edge-validity mask with shape ``(nedge,)``."""


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
        return_mode: Literal["extended", "edges"] = "extended",
        pair_excl: "PairExcludeMask | None" = None,
    ) -> tuple[Array, Array, Array, Array] | EdgeNeighborList:
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
        return_mode
            ``"extended"`` returns the historical extended-coordinate quartet.
            ``"edges"`` returns :class:`EdgeNeighborList`, where ``edge_vec`` is
            the only geometric displacement consumed by the model.
        pair_excl : PairExcludeMask or None, optional
            When provided, excluded type pairs are erased from the returned
            neighbor list (entries set to ``-1``) by calling
            :func:`~deepmd.dpmodel.utils.nlist.apply_pair_exclusion_nlist`.
            Implementations that do not override this parameter fall back to
            the default post-build application in the base interface.

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
