# SPDX-License-Identifier: LGPL-3.0-or-later
"""Complete intra-frame pair topology for interactions without a cutoff.

A module whose interaction acts between every two atoms of a system, such as
a non-periodic long-range module, cannot read its geometry from the
short-range :class:`~deepmd.dpmodel.utils.neighbor_graph.NeighborGraph`. It
declares the :data:`ALL_PAIRS` topology instead: every unordered pair of valid
nodes sharing a frame, listed once with ``src < dst``. The pairs are generated
from the per-frame node counts alone, so their storage is the true pair count
``sum_f n_f (n_f - 1) / 2`` with neither a distance search nor padding to the
widest frame. Pairs of types excluded by the model-level pair exclusion are
left out, as they are left out of the neighbor graph. Like the edge axis of
the neighbor graph, the pair axis ends in masked guard pairs, which keep its
length a dynamic dimension of its own however few real pairs a batch holds.

Geometry enters only through :attr:`PairGraph.vec`, which an autograd backend
makes a leaf next to the short-range ``edge_vec``; its gradient is pulled back
to the nodes by :func:`~deepmd.dpmodel.utils.neighbor_graph.derivatives.pair_force_virial`.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import array_api_compat

from .graph import (
    frame_id_from_n_node,
    pad_and_guard_edges,
)
from .segment import (
    segment_sum,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

ALL_PAIRS = "all_pairs"
"""Name of the topology holding every unordered pair of valid nodes of a frame."""


@dataclass
class PairGraph:
    """Unordered pairs of nodes on the flat node axis of a NeighborGraph."""

    index: Array
    """(2, P) int64  [src, dst] with src < dst; both endpoints in one frame."""
    vec: Array
    """(P, 3) float  r_src - r_dst; the geometry leaf of the pair interaction."""
    mask: Array
    """(P,) bool  real (1) vs guard (0) pairs; guards join node 0 to itself."""


def unordered_pair_index(n_node: Array, node_mask: Array | None = None) -> Array:
    """Enumerate the unordered pairs of valid nodes within every frame.

    Within a frame of ``n`` valid nodes the pairs follow the row-major order
    of the strict upper triangle, ``(0, 1), (0, 2), ..., (0, n-1), (1, 2), ...``.
    Row ``i`` starts at rank ``S(i) = i (2n - 1 - i) / 2``; the row of rank
    ``k`` is the floor of the smaller root of ``S(i) = k``, evaluated in
    float64 and corrected by one integer step against rounding.

    The pair count is data dependent, so the enumeration runs eagerly.

    Parameters
    ----------
    n_node
        Per-frame node counts with shape ``(nf,)``.
    node_mask
        Validity of the flat nodes with shape ``(N,)``; ``None`` takes every
        node as valid. Invalid nodes appear in no pair.

    Returns
    -------
    Array
        The ``[src, dst]`` node indices with shape ``(2, P)``, int64.
    """
    xp = array_api_compat.array_namespace(n_node)
    device = array_api_compat.device(n_node)
    n_node = xp.astype(n_node, xp.int64)
    if node_mask is None:
        count = n_node
    else:
        frame_id = frame_id_from_n_node(n_node, n_total=node_mask.shape[0])
        count = segment_sum(xp.astype(node_mask, xp.int64), frame_id, n_node.shape[0])
    n_pair = count * (count - 1) // 2
    total = int(xp.sum(n_pair))
    if total == 0:
        return xp.zeros((2, 0), dtype=xp.int64, device=device)
    pair_end = xp.cumulative_sum(n_pair)
    rank = xp.arange(total, dtype=xp.int64, device=device)
    pair_frame = xp.astype(xp.searchsorted(pair_end, rank, side="right"), xp.int64)
    local = rank - xp.take(pair_end - n_pair, pair_frame, axis=0)
    n = xp.take(count, pair_frame, axis=0)

    def row_start(row: Array) -> Array:
        return row * (2 * n - 1 - row) // 2

    b = xp.astype(2 * n - 1, xp.float64)
    discriminant = b * b - 8.0 * xp.astype(local, xp.float64)
    discriminant = xp.where(
        discriminant > 0.0, discriminant, xp.zeros_like(discriminant)
    )
    row = xp.astype(xp.floor((b - xp.sqrt(discriminant)) / 2.0), xp.int64)
    row = xp.where(local < row_start(row), row - 1, row)
    row = xp.where(local >= row_start(row + 1), row + 1, row)
    col = row + 1 + (local - row_start(row))
    first = xp.take(xp.cumulative_sum(count) - count, pair_frame, axis=0)
    src = first + row
    dst = first + col
    if node_mask is not None:
        # The ranks above count valid nodes only; map them to node positions.
        position = xp.reshape(xp.nonzero(node_mask)[0], (-1,))
        position = xp.astype(position, xp.int64)
        src = xp.take(position, src, axis=0)
        dst = xp.take(position, dst, axis=0)
    return xp.stack([src, dst])


def build_pair_graph(
    coord: Array,
    n_node: Array,
    atype: Array,
    node_mask: Array | None = None,
    pair_excl: PairExcludeMask | None = None,
) -> PairGraph:
    """Build the :data:`ALL_PAIRS` topology of flat node coordinates.

    Parameters
    ----------
    coord
        Node coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
        A differentiable input keeps :attr:`PairGraph.vec` differentiable.
    n_node
        Per-frame node counts with shape ``(nf,)``.
    atype
        Node types with shape ``(N,)``; invalid nodes carry any type.
    node_mask
        Validity of the nodes with shape ``(N,)``, or ``None`` when every
        node is valid.
    pair_excl
        The model-level pair exclusion, or ``None``; pairs of excluded types
        are left out.

    Returns
    -------
    PairGraph
        The unordered pairs of valid nodes and their displacements, followed
        by two masked guard pairs.
    """
    xp = array_api_compat.array_namespace(coord)
    index = unordered_pair_index(n_node, node_mask)
    if pair_excl is not None:
        keep = pair_excl.build_edge_exclude_mask(index, atype)
        index = xp.take(index, xp.reshape(xp.nonzero(keep)[0], (-1,)), axis=1)
    vec = xp.take(coord, index[0], axis=0) - xp.take(coord, index[1], axis=0)
    index, vec, mask = pad_and_guard_edges(index, vec, None)
    return PairGraph(index=index, vec=vec, mask=mask)


def build_topologies(
    names: Iterable[str],
    coord: Array,
    n_node: Array,
    box: Array | None = None,
    node_mask: Array | None = None,
    *,
    atype: Array,
    pair_excl: PairExcludeMask | None = None,
) -> dict[str, PairGraph]:
    """Build the topologies an atomic model declares, beside its neighbor graph.

    Parameters
    ----------
    names
        Topology names from ``required_topologies()`` of the atomic model.
    coord
        Node coordinates with shape ``(N, 3)``, frame-major over ``n_node``.
    n_node
        Per-frame node counts with shape ``(nf,)``.
    box
        Simulation cells, or ``None`` for non-periodic systems.
    node_mask
        Validity of the nodes with shape ``(N,)``, or ``None`` when every
        node is valid.
    atype
        Node types with shape ``(N,)``; invalid nodes carry any type.
    pair_excl
        The model-level pair exclusion, or ``None``.

    Returns
    -------
    dict[str, PairGraph]
        Each requested topology under its name; empty when none is required.

    Raises
    ------
    NotImplementedError
        If a cell is given: the pair topology describes complete non-periodic
        systems only.
    ValueError
        If a name is unknown.
    """
    topologies = {}
    for name in names:
        if name != ALL_PAIRS:
            raise ValueError(f"unknown topology {name!r}; expected {ALL_PAIRS!r}")
        if box is not None:
            raise NotImplementedError(
                "the all-pairs topology of the long-range modules describes "
                "complete non-periodic systems only; periodic cells are not "
                "supported (pass box=None for a non-periodic system)"
            )
        topologies[name] = build_pair_graph(coord, n_node, atype, node_mask, pair_excl)
    return topologies
