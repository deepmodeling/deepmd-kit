# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all NeighborGraph builder backed by the native CPU cell list.

The graph builders that a Python inference call can reach all run the pair
search on one thread, which makes the search rather than the model the cost of
an evaluation: on an 8000-atom cell the search takes about 90 ms against 4 to
18 ms for a released DPA4C grade. ``deepmd::neighbor_search`` is the same
algorithm threaded over destination atoms, and it emits its pairs
destination-grouped, which is the order the compressed-sparse-row views want.

The builder is CPU-only by construction. CUDA hosts keep the ``nv`` builder,
whose search already runs on the device.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    apply_pair_exclusion,
    attach_edge_csr,
    neighbor_graph_from_ijs,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )


def is_cell_search_available() -> bool:
    """Return whether the native CPU cell-list search is registered.

    Returns
    -------
    bool
        Whether ``deepmd::neighbor_search`` can be called.
    """
    try:
        import deepmd.pt.cxx_op  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops.deepmd, "neighbor_search")


def build_neighbor_graph_fused(
    coord: torch.Tensor,
    atype: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    edge_dtype: torch.dtype = torch.float64,
) -> NeighborGraph:
    """Build a single-frame destination-major graph entirely in the operator.

    The search already computes every displacement it tests, so handing them
    back removes the gathers that would recompute them, the sort that would
    group the payload, and the reordering of every edge field. What remains is
    the search itself and one pass for the source permutation.

    The displacements are therefore *not* a differentiable function of
    ``coord``. This builder serves a frozen artifact, whose forces come from
    the model's own analytical backward with the displacements as inputs; a
    caller that differentiates through the graph must use
    :func:`build_neighbor_graph_cell`.

    Parameters
    ----------
    coord : torch.Tensor
        Coordinates with shape ``(nloc, 3)``.
    atype : torch.Tensor
        Atom types with shape ``(nloc,)``. Virtual atoms are rejected because
        the fused path has no filtering stage.
    box : torch.Tensor or None
        Lattice matrix with shape ``(3, 3)``, or ``None`` when the system is
        not periodic.
    rcut : float
        Cutoff radius.
    edge_dtype : torch.dtype
        Scalar type of the returned displacements.

    Returns
    -------
    NeighborGraph
        A destination-major graph with both CSR views attached.

    Raises
    ------
    ValueError
        If any atom is virtual.
    """
    if bool((atype < 0).any()):
        raise ValueError(
            "the fused graph builder has no virtual-atom filter; use "
            "build_neighbor_graph_cell for a system carrying atype < 0"
        )
    empty_cell = torch.zeros((3, 3), dtype=coord.dtype, device=coord.device)
    (
        edge_index,
        edge_vec,
        edge_mask,
        destination_row_ptr,
        source_order,
        source_row_ptr,
    ) = torch.ops.deepmd.neighbor_graph(
        coord.detach(),
        box.detach() if box is not None else empty_cell,
        box is not None,
        float(rcut),
        edge_dtype,
    )
    return NeighborGraph(
        n_node=torch.full((1,), coord.shape[0], dtype=torch.int64, device=coord.device),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
        destination_order=torch.arange(
            edge_index.shape[1], dtype=torch.int64, device=coord.device
        ),
        destination_row_ptr=destination_row_ptr,
        source_order=source_order,
        source_row_ptr=source_row_ptr,
        destination_sorted=True,
    )


def cell_search_ijs(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    periodic: bool,
    rcut: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the native cell-list search for one frame.

    Parameters
    ----------
    positions : torch.Tensor
        Detached coordinates with shape ``(nloc, 3)``.
    cell : torch.Tensor or None
        Lattice matrix with shape ``(3, 3)``, rows being the lattice vectors.
        Ignored when the system is not periodic.
    periodic : bool
        Whether the lattice wraps.
    rcut : float
        Cutoff radius.

    Returns
    -------
    ii : torch.Tensor
        Center index of each pair with shape ``(E,)``, ascending.
    jj : torch.Tensor
        Neighbor index of each pair with shape ``(E,)``.
    ss : torch.Tensor
        Integer lattice image of each pair with shape ``(E, 3)``.
    """
    empty_cell = torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)
    return torch.ops.deepmd.neighbor_search(
        positions,
        cell if periodic else empty_cell,
        periodic,
        float(rcut),
    )


def build_neighbor_graph_cell(
    coord: Any,
    atype: Any,
    box: Any | None,
    rcut: float,
    layout: GraphLayout | None = None,
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
    pair_excl: PairExcludeMask | None = None,
    compact: bool = False,
) -> NeighborGraph:
    """Build a carry-all NeighborGraph with the native CPU cell list.

    Emits the same neighbor set as every other builder; the choice is
    performance-only. Frames are searched one at a time, which is what the
    single-frame inference call this builder serves needs.

    Parameters
    ----------
    coord : Any
        Coordinates with shape ``(nf, nloc, 3)``.
    atype : Any
        Atom types with shape ``(nf, nloc)``.
    box : Any or None
        Simulation cells with shape ``(nf, 3, 3)``, or ``None`` when the system
        is not periodic.
    rcut : float
        Cutoff radius.
    layout : GraphLayout or None
        Edge-axis length policy.
    with_csr : bool
        Whether to construct destination and source CSR views.
    canonicalize : bool
        Whether to reorder every edge field into destination-major form.
        Implies ``with_csr``.
    pair_excl : PairExcludeMask or None
        Model-level ``pair_exclude_types`` mask, applied after the geometric
        search.
    compact : bool
        Passed to :func:`apply_pair_exclusion`.

    Returns
    -------
    NeighborGraph
        The carry-all graph over the local atoms.

    Raises
    ------
    ImportError
        If the native search is not registered for this build.
    """
    if not is_cell_search_available():
        raise ImportError(
            "build_neighbor_graph_cell requires the DeePMD-kit PyTorch "
            "operator library; use neighbor_graph_method='dense'."
        )

    xp = array_api_compat.array_namespace(coord)
    dev = array_api_compat.device(coord)
    nf = coord.shape[0] if coord.ndim == 3 else 1
    coord = xp.reshape(coord, (nf, -1, 3))
    nloc = coord.shape[1]
    periodic = box is not None
    if periodic:
        box = xp.reshape(box, (nf, 3, 3))

    centers, neighbors, images, frames = [], [], [], []
    for frame in range(nf):
        ii, jj, ss = cell_search_ijs(
            coord[frame].detach(),
            box[frame].detach() if periodic else None,
            periodic,
            rcut,
        )
        centers.append(ii)
        neighbors.append(jj)
        images.append(ss)
        frames.append(torch.full((ii.shape[0],), frame, dtype=torch.int64, device=dev))

    def _concat(parts: list[torch.Tensor], width: int = 0) -> torch.Tensor:
        if parts:
            return torch.cat(parts)
        shape = (0, width) if width else (0,)
        return torch.zeros(shape, dtype=torch.int64, device=dev)

    center_all = _concat(centers)
    neighbor_all = _concat(neighbors)
    image_all = _concat(images, width=3)
    frame_all = _concat(frames)

    # Virtual atoms (atype < 0) are excluded as centers and as neighbours --
    # the builder contract shared with the dense reference builder, which the
    # geometric search cannot know about.
    types = torch.as_tensor(atype, device=dev).reshape(nf, nloc)
    keep = (types[frame_all, center_all] >= 0) & (types[frame_all, neighbor_all] >= 0)
    center_all = center_all[keep]
    neighbor_all = neighbor_all[keep]
    image_all = image_all[keep]
    frame_all = frame_all[keep]

    # The original, gradient-carrying coordinates go through: the search is
    # non-differentiable and the displacements are recomputed from them.
    #
    # The search walks its centers in order and frames are concatenated in
    # order, so the destination grouping holds without a sort -- unless a type
    # exclusion clears mask bits in the middle of the payload, which breaks the
    # invariant that masked entries occupy the suffix.
    graph = neighbor_graph_from_ijs(
        center_all,
        neighbor_all,
        image_all,
        coord,
        box,
        frame_all,
        torch.full((nf,), nloc, dtype=torch.int64, device=dev),
        layout=layout,
    )
    destination_sorted = pair_excl is None
    if pair_excl is not None:
        graph = apply_pair_exclusion(
            graph,
            torch.as_tensor(atype, device=dev).reshape(-1),
            pair_excl,
            compact=compact,
        )
    if with_csr or canonicalize:
        graph = attach_edge_csr(
            graph,
            nf * nloc,
            canonicalize=canonicalize,
            destination_sorted=destination_sorted,
        )
    return graph
