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

import math
import warnings
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)

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
        Sequence,
    )

    from deepmd.dpmodel.array_api import (
        Array,
    )
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

ALL_PAIRS = "all_pairs"
"""Name of the topology holding every unordered pair of valid nodes of a frame."""

K_GRID = "k_grid"
"""Name of the topology holding the reciprocal lattice points of periodic frames.

A long-range module declares it beside :data:`ALL_PAIRS` when it serves
periodic systems: the reciprocal-space evaluation of the pair kernel needs the
wave numbers, which the pair topology cannot carry.
"""

PME_MESH = "pme_mesh"
"""Name of the topology carrying the cells of periodic frames for a mesh evaluation.

A long-range module declares it beside :data:`ALL_PAIRS` when it evaluates the
reciprocal sum of a smooth kernel on a real-space mesh (SPME): charges spread
to a static mesh by a B-spline, transformed by an FFT, and convolved with the
kernel transform. The mesh axes are static and live on the module, so the
topology holds no ragged axis of its own — only the per-frame cells and the
geometry leaves.
"""


@dataclass
class PairGraph:
    """Unordered pairs of nodes on the flat node axis of a NeighborGraph."""

    index: Array
    """(2, P) int64  [src, dst] with src < dst; both endpoints in one frame."""
    vec: Array
    """(P, 3) float  r_src - r_dst; the geometry leaf of the pair interaction."""
    mask: Array
    """(P,) bool  real (1) vs guard (0) pairs; guards join node 0 to itself."""


@dataclass
class KGrid:
    r"""Reciprocal lattice points of the periodic frames of a batch.

    The wave numbers enumerate :math:`\\mathbf{k} = 2\\pi B^{-1}\\mathbf{n}` for
    integer ``n`` with :math:`0 < |\\mathbf{k}| \\le k_{max}`; they are built
    eagerly outside any compiled region, concatenated ragged over the frames,
    and the k axis ends in masked guard points that keep its length a dynamic
    dimension of its own however few points a batch holds.
    """

    frame: Array
    """(nk,) int64  the frame of every wave number."""
    vec: Array
    """(nk, 3) float  the wave numbers, in Å⁻¹."""
    mask: Array
    """(nk,) bool  real (1) vs guard (0) points."""
    volume: Array
    """(nf,) float  the cell volumes, in Å³."""
    coord: Array | None = None
    """(N, 3) float  node positions; the geometry leaf of the k-space energy.

    ``None`` when the grid is reconstructed from flattened inputs (export); a
    graph lower re-leafs it before the evaluation.
    """
    strain: Array | None = None
    """(nf, 3, 3) float  zero strain leaf for the k-space virial.

    The module evaluates the energy on coordinates strained by ``I + strain``,
    wave numbers strained by ``(I + strain)^{-T}`` and volumes scaled by
    ``det(I + strain)``; the gradient of the energy with respect to this leaf
    is the negative k-space virial. The leaf is built eagerly beside the grid
    (all zeros) and re-leafed by the graph lower, so the backward roots at an
    input of the traced graph — a leaf created INSIDE a ``make_fx`` trace
    produces a NaN gradient under inductor.
    """


@dataclass
class MeshGrid:
    r"""Cells and geometry of the periodic frames for a mesh (FFT) evaluation.

    The reciprocal sum is evaluated on a static real-space mesh shared by the
    frames, so the topology carries no ragged axis: the cells (for the
    reciprocal vectors), the volumes, the geometry leaves, and a per-frame
    flag telling periodic frames from non-periodic ones (a batch without a
    cell comes back with an all-false mask). The volume is computed eagerly
    (as in :class:`KGrid`): the graph lower only rescales it by the strain
    determinant, which keeps ``linalg.det`` out of the traced graph.
    """

    box: Array
    """(nf, 3, 3) float  simulation cells; rows are the lattice vectors."""
    volume: Array
    """(nf,) float  the cell volumes, in Å³."""
    coord: Array | None
    """(N, 3) float  node positions; the geometry leaf of the mesh energy.

    ``None`` when the grid is reconstructed from flattened inputs (export); a
    graph lower re-leafs it before the evaluation.
    """
    strain: Array | None
    """(nf, 3, 3) float  zero strain leaf for the mesh virial.

    Same contract as :attr:`KGrid.strain`: the energy is evaluated with the
    coordinates strained by ``I + strain`` and the cells by
    ``B' = B (I + strain)^T`` with the lattice vectors as rows, so the
    fractional coordinates are strain-invariant and the gradient with respect
    to the leaf is the negative virial.
    """
    mask: Array
    """(nf,) bool  periodic (1) vs non-periodic (0) frames."""


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


def build_k_grid(
    box: Array,
    k_max: float,
    n_node: Array,
    coord: Array | None = None,
) -> KGrid:
    r"""Enumerate the reciprocal lattice points of the periodic frames.

    For every frame the wave numbers are
    :math:`\\mathbf{k} = 2\\pi \\, B^{-1} \\mathbf{n}` over integer
    :math:`\\mathbf{n}` with :math:`0 < |\\mathbf{k}| \\le k_{max}` — note the
    cell enters through ``B^{-1}``, NOT ``B^{-T}``: with the lattice vectors as
    the ROWS of ``B``, orthogonality reads ``B @ (B^{-1} n) = n``. The count is
    data dependent, so the enumeration runs eagerly. Every frame's points come
    in lexicographic integer order, and the batch ends in two masked guard
    points.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)``; rows are the lattice
        vectors.
    k_max
        The wave-number cutoff, in Å⁻¹.
    n_node
        Per-frame node counts with shape ``(nf,)``.
    coord
        Node positions with shape ``(N, 3)``, stored on the grid as the
        geometry context (see :attr:`KGrid.coord`).

    Returns
    -------
    KGrid
        The reciprocal lattice points of every frame, ragged-concatenated.
    """
    xp = array_api_compat.array_namespace(box)
    device = array_api_compat.device(box)
    # A flat (nf, 9) layout arrives from callers that carry the cell flattened.
    if box.ndim == 2:
        box = xp.reshape(box, (box.shape[0], 3, 3))
    nf = box.shape[0]
    k_lists = []
    frame_lists = []
    for f in range(nf):
        cell = np.asarray(to_numpy_array(box[f]), dtype=np.float64)
        rec = 2.0 * math.pi * np.linalg.inv(cell)  # columns are g_d
        a_len = np.linalg.norm(cell, axis=1)
        n_d = np.ceil(k_max * a_len / (2.0 * math.pi)).astype(np.int64)
        grids = np.meshgrid(
            *[np.arange(-n, n + 1, dtype=np.float64) for n in n_d], indexing="ij"
        )
        n_all = np.stack(grids, axis=-1).reshape(-1, 3)
        k = n_all @ rec.T
        k2 = np.sum(k * k, axis=-1)
        keep = (k2 > 0.0) & (k2 <= k_max * k_max)
        k_lists.append(k[keep])
        frame_lists.append(np.full(int(np.sum(keep)), f, dtype=np.int64))
    if k_lists:
        vec_np = np.concatenate(k_lists, axis=0)
        frame_np = np.concatenate(frame_lists, axis=0)
    else:
        vec_np = np.zeros((0, 3), dtype=np.float64)
        frame_np = np.zeros((0,), dtype=np.int64)
    # Guard points: masked out, keep the k axis a dynamic dimension of its own.
    guard = np.tile(np.array([[1e-6, 0.0, 0.0]]), (2, 1))
    vec_np = np.concatenate([vec_np, guard], axis=0)
    frame_np = np.concatenate([frame_np, np.zeros(2, dtype=np.int64)], axis=0)
    mask_np = np.zeros(vec_np.shape[0], dtype=bool)
    mask_np[: vec_np.shape[0] - 2] = True
    volume = xp.linalg.det(box)
    volume = xp.abs(volume)
    return KGrid(
        frame=xp.asarray(frame_np, device=device),
        vec=xp.astype(xp.asarray(vec_np, device=device), box.dtype),
        mask=xp.asarray(mask_np, device=device),
        volume=xp.astype(volume, box.dtype),
        coord=coord,
        strain=xp.zeros((nf, 3, 3), dtype=box.dtype, device=device),
    )


def k_grid_bounds_of_cells(box: Array, k_max: float) -> np.ndarray:
    r"""Per-axis integer wave-number bounds the cells need at ``k_max``.

    A wave number :math:`\mathbf{k} = 2\pi B^{-1}\mathbf{n}` with
    :math:`|\mathbf{k}| \le k_{max}` needs :math:`|n_d| \le k_{max} |a_d| / 2\pi`
    per axis (Cauchy-Schwarz), so the bound of every frame is the ceiling of
    the right-hand side.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)`` or ``(nf, 9)``; rows are
        the lattice vectors.
    k_max
        The wave-number cutoff, in Å⁻¹.

    Returns
    -------
    np.ndarray
        The per-frame, per-axis integer bounds with shape ``(nf, 3)``.
    """
    cell = np.asarray(to_numpy_array(box), dtype=np.float64).reshape(-1, 3, 3)
    a_len = np.linalg.norm(cell, axis=-1)
    return np.ceil(k_max * a_len / (2.0 * math.pi)).astype(np.int64)


def enumerate_k_grid_bounds(bounds: Array | list[int]) -> np.ndarray:
    r"""Enumerate the integer wave-number indices within per-axis bounds.

    The grid is the full meshgrid of :math:`\mathbf{n}` with
    :math:`|n_d| \le n_d^{max}`, in lexicographic order (the first axis
    outermost). The zero vector is left out: its kernel transform is
    singular and the tin-foil convention drops the :math:`\mathbf{k} = 0`
    mode anyway.

    Parameters
    ----------
    bounds
        The per-axis bounds :math:`n_d^{max}` with shape ``(3,)``.

    Returns
    -------
    np.ndarray
        The integer indices with shape ``(nk, 3)``, int64.
    """
    bounds = np.asarray(to_numpy_array(bounds), dtype=np.int64).reshape(-1)
    if bounds.shape != (3,) or np.any(bounds < 0):
        raise ValueError(
            f"k-grid bounds must be three non-negative integers, got {bounds}"
        )
    grids = np.meshgrid(
        *[np.arange(-n, n + 1, dtype=np.int64) for n in bounds], indexing="ij"
    )
    n_grid = np.stack(grids, axis=-1).reshape(-1, 3)
    return n_grid[np.any(n_grid != 0, axis=-1)]


def build_fixed_k_grid(
    box: Array,
    n_grid: Array,
    k_max: float,
    n_node: Array,
    coord: Array | None = None,
) -> KGrid:
    r"""Evaluate a fixed integer k grid on the cells of the periodic frames.

    Every frame shares the integer indices ``n_grid`` (see
    :func:`enumerate_k_grid_bounds`), so the k axis has the static length
    ``nf * nk``; the wave numbers of a frame are
    :math:`\mathbf{k} = 2\pi B^{-1}\mathbf{n}` and the points the frame does
    not reach — outside :math:`0 < |\mathbf{k}| \le k_{max}` — are masked out.
    The mask is evaluated eagerly with the wave-number cutoff current at
    build time, exactly as in :func:`build_k_grid`. The enumeration runs
    eagerly beside the ragged one, but no count is data dependent: the grid
    comes back without guard points.

    The grid is HARD-FIXED: a frame whose cell would need integer indices
    beyond the grid (a larger cell than the grid was sized for, or a kernel
    that moved the cutoff out since) is served by the grid alone — its
    reciprocal lattice is denser than the grid, so its energy is the
    truncated k-sum of the grid, a well-defined approximation whose forces
    and virial stay the exact derivatives of that energy. A warning is
    issued when this happens.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)``; rows are the lattice
        vectors.
    n_grid
        The shared integer wave-number indices with shape ``(nk, 3)``.
    k_max
        The wave-number cutoff, in Å⁻¹.
    n_node
        Per-frame node counts with shape ``(nf,)`` (interface symmetry with
        :func:`build_k_grid`; the grid does not read it).
    coord
        Node positions with shape ``(N, 3)``, stored on the grid as the
        geometry context (see :attr:`KGrid.coord`).

    Returns
    -------
    KGrid
        The wave numbers of every frame, frame-major over the shared grid.

    Raises
    ------
    ValueError
        Never for an oversized cell (it is truncated, see above); kept for
        interface symmetry.
    """
    xp = array_api_compat.array_namespace(box)
    device = array_api_compat.device(box)
    if box.ndim == 2:
        box = xp.reshape(box, (box.shape[0], 3, 3))
    nf = box.shape[0]
    n_grid = np.asarray(to_numpy_array(n_grid), dtype=np.int64).reshape(-1, 3)
    nk = n_grid.shape[0]
    bounds = np.max(np.abs(n_grid), axis=0) if nk else np.zeros(3, dtype=np.int64)
    required = k_grid_bounds_of_cells(box, k_max)
    over = np.any(required > bounds[None, :], axis=1)
    if np.any(over):
        f = int(np.argmax(over))
        warnings.warn(
            f"frame {f} needs integer k-grid bounds {required[f].tolist()}, "
            f"beyond the fixed grid {bounds.tolist()}: its k-sum is truncated "
            f"to the grid (an approximation for this frame); enlarge "
            f"`k_grid_bounds` for full accuracy",
            stacklevel=2,
        )
    cells = np.asarray(to_numpy_array(box), dtype=np.float64)
    rec = 2.0 * math.pi * np.linalg.inv(cells)  # (nf, 3, 3), columns are g_d
    k = np.einsum("jm,fdm->fjd", n_grid.astype(np.float64), rec)
    k2 = np.sum(k * k, axis=-1)
    mask_np = (k2 > 0.0) & (k2 <= k_max * k_max)  # (nf, nk)
    vec_np = k.reshape(nf * nk, 3)
    frame_np = np.repeat(np.arange(nf, dtype=np.int64), nk)
    volume = xp.linalg.det(box)
    volume = xp.abs(volume)
    return KGrid(
        frame=xp.asarray(frame_np, device=device),
        vec=xp.astype(xp.asarray(vec_np, device=device), box.dtype),
        mask=xp.asarray(mask_np.reshape(-1), device=device),
        volume=xp.astype(volume, box.dtype),
        coord=coord,
        strain=xp.zeros((nf, 3, 3), dtype=box.dtype, device=device),
    )


def _empty_k_grid(nf: int, like: Array, coord: Array | None) -> KGrid:
    """A guard-only k grid for a batch without periodic frames."""
    xp = array_api_compat.array_namespace(like)
    device = array_api_compat.device(like)
    return KGrid(
        frame=xp.zeros((2,), dtype=xp.int64, device=device),
        vec=xp.tile(
            xp.asarray([[1e-6, 0.0, 0.0]], dtype=like.dtype, device=device), (2, 1)
        ),
        mask=xp.zeros((2,), dtype=xp.bool, device=device),
        volume=xp.ones((nf,), dtype=like.dtype, device=device),
        coord=coord,
        strain=xp.zeros((nf, 3, 3), dtype=like.dtype, device=device),
    )


def _empty_mesh_grid(nf: int, like: Array, coord: Array | None) -> MeshGrid:
    """A non-periodic mesh grid: every frame masked out, identity cells."""
    xp = array_api_compat.array_namespace(like)
    device = array_api_compat.device(like)
    return MeshGrid(
        box=xp.tile(xp.eye(3, dtype=like.dtype, device=device), (nf, 1, 1)),
        volume=xp.ones((nf,), dtype=like.dtype, device=device),
        coord=coord,
        strain=xp.zeros((nf, 3, 3), dtype=like.dtype, device=device),
        mask=xp.zeros((nf,), dtype=xp.bool, device=device),
    )


def build_mesh_grid(
    box: Array,
    n_mesh: Array,
    coord: Array | None = None,
) -> MeshGrid:
    r"""Pack the cells of the periodic frames for the mesh (FFT) evaluation.

    The mesh is shared by every frame; a frame whose cell is larger than the
    mesh was sized for is evaluated with a coarser effective spacing
    :math:`|a_d|/n_d` than the target — an undersampled kernel spectrum, a
    well-defined approximation whose forces and virial stay the exact
    derivatives of that energy. Sizing the mesh (data statistics or
    ``pme_mesh``) is the caller's responsibility; no warning is issued here.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)``; rows are the lattice
        vectors.
    n_mesh
        The per-axis mesh dimensions with shape ``(3,)``.
    coord
        Node positions with shape ``(N, 3)``, stored on the grid as the
        geometry context (see :attr:`MeshGrid.coord`).

    Returns
    -------
    MeshGrid
        The cells and geometry of the frames, every frame flagged periodic.
    """
    xp = array_api_compat.array_namespace(box)
    if box.ndim == 2:
        box = xp.reshape(box, (box.shape[0], 3, 3))
    volume = xp.linalg.det(box)
    volume = xp.abs(volume)
    return MeshGrid(
        box=box,
        volume=xp.astype(volume, box.dtype),
        coord=coord,
        strain=xp.zeros(
            (box.shape[0], 3, 3), dtype=box.dtype, device=array_api_compat.device(box)
        ),
        mask=xp.ones(
            (box.shape[0],), dtype=xp.bool, device=array_api_compat.device(box)
        ),
    )


def _empty_pair_graph(like: Array, atype: Array) -> PairGraph:
    """A guard-only pair topology for a batch without non-periodic frames."""
    xp = array_api_compat.array_namespace(like)
    device = array_api_compat.device(like)
    index = xp.zeros((2, 0), dtype=xp.int64, device=device)
    vec = xp.zeros((0, 3), dtype=like.dtype, device=device)
    index, vec, mask = pad_and_guard_edges(index, vec, None)
    return PairGraph(index=index, vec=vec, mask=mask)


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
    topology_params: dict[str, dict] | None = None,
) -> dict[str, PairGraph | KGrid]:
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
        Simulation cells, or ``None`` for non-periodic systems. With a cell,
        the pair topology comes back guard-only and the k grid holds the
        reciprocal lattice points of the frames; without one, the reverse.
    node_mask
        Validity of the nodes with shape ``(N,)``, or ``None`` when every
        node is valid.
    atype
        Node types with shape ``(N,)``; invalid nodes carry any type.
    pair_excl
        The model-level pair exclusion, or ``None``.
    topology_params
        Per-name build parameters; the k grid takes ``{"k_max": ...}`` and,
        for a fixed integer grid, additionally ``{"n_grid": ...}`` (see
        :func:`build_fixed_k_grid`).

    Returns
    -------
    dict[str, PairGraph | KGrid]
        Each requested topology under its name; empty when none is required.

    Raises
    ------
    NotImplementedError
        If a cell is given for a pair-only declaration, or a pair exclusion is
        set for a periodic batch (the k-space sum is global).
    ValueError
        If a name is unknown, or the k grid parameters are missing.
    """
    topologies: dict[str, PairGraph | KGrid | MeshGrid] = {}
    for name in names:
        if name == ALL_PAIRS:
            if box is None:
                topologies[name] = build_pair_graph(
                    coord, n_node, atype, node_mask, pair_excl
                )
            elif K_GRID in names or PME_MESH in names:
                # A periodic-capable module reads the k grid or the mesh for
                # these frames.
                topologies[name] = _empty_pair_graph(coord, atype)
            else:
                check_complete_frames(box)
        elif name == K_GRID:
            if box is None:
                topologies[name] = _empty_k_grid(n_node.shape[0], coord, coord)
            else:
                if pair_excl is not None:
                    raise NotImplementedError(
                        "the k-space sum of a periodic system is global; "
                        "`pair_exclude_types` cannot exclude pairs from it"
                    )
                params = (topology_params or {}).get(K_GRID, {})
                if "k_max" not in params:
                    raise ValueError(
                        f"the {K_GRID!r} topology requires its `k_max` build parameter"
                    )
                if "n_grid" in params:
                    topologies[name] = build_fixed_k_grid(
                        box, params["n_grid"], params["k_max"], n_node, coord=coord
                    )
                else:
                    topologies[name] = build_k_grid(
                        box, params["k_max"], n_node, coord=coord
                    )
        elif name == PME_MESH:
            if box is None:
                topologies[name] = _empty_mesh_grid(n_node.shape[0], coord, coord)
            else:
                if pair_excl is not None:
                    raise NotImplementedError(
                        "the k-space sum of a periodic system is global; "
                        "`pair_exclude_types` cannot exclude pairs from it"
                    )
                params = (topology_params or {}).get(PME_MESH, {})
                if "n_mesh" not in params:
                    raise ValueError(
                        f"the {PME_MESH!r} topology requires its `n_mesh` "
                        f"build parameter"
                    )
                topologies[name] = build_mesh_grid(box, params["n_mesh"], coord=coord)
        else:
            raise ValueError(
                f"unknown topology {name!r}; expected one of "
                f"({ALL_PAIRS!r}, {K_GRID!r}, {PME_MESH!r})"
            )
    return topologies


def check_complete_frames(box: Array | None) -> None:
    """Reject a cell, since the all-pairs topology covers non-periodic frames.

    Parameters
    ----------
    box
        Simulation cells, or ``None`` for non-periodic systems.

    Raises
    ------
    NotImplementedError
        If a cell is given.
    """
    if box is not None:
        raise NotImplementedError(
            "the all-pairs topology of the long-range modules describes "
            "complete non-periodic systems only; periodic cells are not "
            "supported (pass box=None for a non-periodic system)"
        )


def flatten_topologies(topologies: dict[str, PairGraph | KGrid]) -> tuple[Array, ...]:
    """Lay topologies out as positional inputs of a graph lower.

    A compiled or exported graph lower receives every topology positionally,
    in the order of the dict, after its other inputs: a pair topology as its
    ``(index, vec, mask)``, a k grid as its ``(frame, vec, mask, volume,
    coord)``;
    :func:`unflatten_topologies` restores them.

    Parameters
    ----------
    topologies
        The topologies by name.

    Returns
    -------
    tuple
        The flattened tensors of every topology, concatenated.
    """
    return tuple(
        value
        for topology in topologies.values()
        for value in topology_flat_fields(topology)
    )


def topology_flat_fields(topology: PairGraph | KGrid | MeshGrid) -> tuple[Array, ...]:
    """The positional tensors of one topology."""
    if isinstance(topology, KGrid):
        if topology.coord is None or topology.strain is None:
            raise ValueError(
                "a k grid carries no coordinates or strain; it cannot be laid out "
                "positionally"
            )
        return (
            topology.frame,
            topology.vec,
            topology.mask,
            topology.volume,
            topology.coord,
            topology.strain,
        )
    if isinstance(topology, MeshGrid):
        if topology.coord is None or topology.strain is None:
            raise ValueError(
                "a mesh grid carries no coordinates or strain; it cannot be laid "
                "out positionally"
            )
        return (
            topology.box,
            topology.volume,
            topology.coord,
            topology.strain,
            topology.mask,
        )
    return (topology.index, topology.vec, topology.mask)


def unflatten_topologies(
    names: Sequence[str], values: Sequence[Array]
) -> dict[str, PairGraph | KGrid]:
    """Restore topologies laid out by :func:`flatten_topologies`.

    Parameters
    ----------
    names
        The topology names, in the order they were laid out.
    values
        The flattened tensors of every topology, concatenated.

    Returns
    -------
    dict[str, PairGraph | KGrid]
        The topologies by name.

    Raises
    ------
    ValueError
        If the number of values does not match the layout of the names.
    """
    arity = {ALL_PAIRS: 3, K_GRID: 6, PME_MESH: 5}
    topologies: dict[str, PairGraph | KGrid | MeshGrid] = {}
    offset = 0
    for name in names:
        if name not in arity:
            raise ValueError(f"unknown topology {name!r}")
        fields = values[offset : offset + arity[name]]
        if len(fields) != arity[name]:
            raise ValueError(
                f"expected {arity[name]} inputs for topology {name!r}, "
                f"got {len(values) - offset} remaining"
            )
        if name == ALL_PAIRS:
            topologies[name] = PairGraph(*fields)
        elif name == K_GRID:
            topologies[name] = KGrid(*fields)
        else:
            topologies[name] = MeshGrid(*fields)
        offset += arity[name]
    if offset != len(values):
        raise ValueError(
            f"{len(values)} flattened values given, {offset} consumed by {tuple(names)}"
        )
    return topologies
