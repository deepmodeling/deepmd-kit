# SPDX-License-Identifier: LGPL-3.0-or-later
"""Reciprocal-space topologies of the periodic frames of a batch.

A long-range module that serves periodic systems evaluates the lattice sum of
its pair kernel in reciprocal space, which needs the cell of every frame and
the absolute node positions rather than pair displacements. It declares one of
two topologies beside
:data:`~deepmd.dpmodel.utils.neighbor_graph.pair_graph.ALL_PAIRS`:

- :data:`K_GRID` lists the wave numbers of every frame explicitly, ragged over
  the frames, for a direct sum over them;
- :data:`PME_MESH` carries the cells and the mode indices of a real-space mesh
  shared by the frames, for a smooth particle-mesh evaluation.

Both carry the node positions and a zero strain as geometry: an autograd
backend makes them leaves, whose gradients are the force and the negative
virial of the reciprocal-space energy. A batch without cells comes back with
masked topologies, so a module serving periodic systems also serves
non-periodic batches through its pair topology.
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
    ClassVar,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )

K_GRID = "k_grid"
"""Name of the topology holding the reciprocal lattice points of periodic frames."""

PME_MESH = "pme_mesh"
"""Name of the topology carrying the cells and the mesh of periodic frames."""


@dataclass
class KGrid:
    r"""Reciprocal lattice points of the periodic frames of a batch, frame by frame.

    Row ``f`` of :attr:`vec` holds the wave numbers :math:`\mathbf{k} = 2\pi
    B_f^{-1}\mathbf{n}` of frame ``f``, for integer :math:`\mathbf{n}`, built
    eagerly outside any compiled region, in blocks of :data:`K_BLOCK` points
    padded with masked guard points to the blocks of the longest row; row
    ``f`` of :attr:`slot` holds the nodes of frame ``f``, padded with ``-1``.
    A frame thus pairs its own wave numbers with its own nodes alone, at a
    cost of the frame count times the padded wave-number count and the
    padded node count, and every sum over the wave numbers of a frame runs
    over the points of a block first. The block, point and slot axes are dynamic dimensions of
    their own, of at least two entries.

    The energy is evaluated on the strained geometry of :attr:`strain`:
    positions by ``S = I + strain``, wave numbers by :math:`S^{-T}` and volumes
    by :math:`\det S`, so that the gradient with respect to the strain is the
    negative virial; the phases :math:`\mathbf{k} \cdot \mathbf{r}` do not
    change under the strain. The strain is zero, and the evaluation takes it
    to first order, which fixes the energy and that gradient. The graph lower
    makes :attr:`coord` and :attr:`strain` autograd leaves at its inputs.
    """

    vec: Array
    """(nf, nb, kb, 3) float  the wave numbers of every frame in blocks, in Å⁻¹."""
    mask: Array
    """(nf, nb, kb) bool  real (1) vs guard (0) points."""
    slot: Array
    """(nf, ns) int64  the nodes of every frame, ``-1`` past its last node."""
    volume: Array
    """(nf,) float  the cell volumes, in Å³."""
    coord: Array
    """(N, 3) float  node positions, in Å."""
    strain: Array
    """(nf, 3, 3) float  zero strain of every frame."""

    GEOMETRY: ClassVar[tuple[str, ...]] = ("vec", "volume", "coord", "strain")
    """Fields in the precision of the geometry."""
    LEAVES: ClassVar[tuple[str, ...]] = ("coord", "strain")
    """Fields an autograd backend makes leaves of the energy."""
    AXES: ClassVar[dict[str, dict[int, str]]] = {
        "vec": {0: "frame", 1: "entry", 2: "k_point"},
        "mask": {0: "frame", 1: "entry", 2: "k_point"},
        "slot": {0: "frame", 1: "slot"},
        "volume": {0: "frame"},
        "coord": {0: "node"},
        "strain": {0: "frame"},
    }
    """Dynamic axes of every field: the frame, block, point, slot and node axes."""


@dataclass
class MeshGrid:
    r"""Cells and mesh of the periodic frames of a batch.

    The frames share a real-space mesh of :math:`n_1 \times n_2 \times n_3`
    points spanning each cell, whose integer mode indices along every axis
    are the fields :attr:`mode_1`, :attr:`mode_2` and :attr:`mode_3`: their
    lengths are the mesh dimensions, so a compiled or exported graph takes the
    mesh as dynamic dimensions. Every dimension is even. A batch without a
    cell comes back with identity cells, a minimal mesh and an all-false
    :attr:`mask`.

    The geometry contract is that of :class:`KGrid`: the energy is evaluated
    with the positions strained by ``S = I + strain`` and the cells by
    :math:`B S^{\mathsf{T}}`, the lattice vectors being the rows of
    :math:`B`, so that the fractional coordinates are invariant and the
    gradient with respect to the strain is the negative virial. The volumes
    of the unstrained cells are computed eagerly, and the evaluation scales
    them by :math:`\det S`, to first order in the strain.
    """

    box: Array
    """(nf, 3, 3) float  simulation cells; rows are the lattice vectors, in Å."""
    volume: Array
    """(nf,) float  the cell volumes, in Å³."""
    coord: Array
    """(N, 3) float  node positions, in Å."""
    strain: Array
    """(nf, 3, 3) float  zero strain of every frame."""
    mask: Array
    """(nf,) bool  periodic (1) vs non-periodic (0) frames."""
    mode_1: Array
    """(n1,) float  mode indices of the first mesh axis, in FFT order."""
    mode_2: Array
    """(n2,) float  mode indices of the second mesh axis, in FFT order."""
    mode_3: Array
    """(n3,) float  mode indices of the third mesh axis, in FFT order."""

    GEOMETRY: ClassVar[tuple[str, ...]] = (
        "box",
        "volume",
        "coord",
        "strain",
        "mode_1",
        "mode_2",
        "mode_3",
    )
    """Fields in the precision of the geometry."""
    LEAVES: ClassVar[tuple[str, ...]] = ("coord", "strain")
    """Fields an autograd backend makes leaves of the energy."""
    AXES: ClassVar[dict[str, dict[int, str]]] = {
        "box": {0: "frame"},
        "volume": {0: "frame"},
        "coord": {0: "node"},
        "strain": {0: "frame"},
        "mask": {0: "frame"},
        "mode_1": {0: "mesh_1"},
        "mode_2": {0: "mesh_2"},
        "mode_3": {0: "mesh_3"},
    }
    """Dynamic axes of every field: the frame axis, the node axis, the mesh axes."""


def check_periodic_cells(box: Array) -> None:
    """Reject a cell that spans no lattice.

    A cell whose volume is a vanishing fraction of the product of the lengths
    of its lattice vectors, or that is not finite, spans no lattice; an
    all-zero cell is no marker of a non-periodic frame, which takes no cell.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 9)`` or ``(nf, 3, 3)``; rows are
        the lattice vectors.

    Raises
    ------
    ValueError
        If a cell spans no lattice.
    """
    cells = np.asarray(to_numpy_array(box), dtype=np.float64).reshape(-1, 3, 3)
    volume = np.abs(np.linalg.det(cells))
    spanned = volume > 1e-6 * np.prod(np.linalg.norm(cells, axis=-1), axis=-1)
    if not np.all(spanned):
        f = int(np.argmin(spanned))
        raise ValueError(
            f"the cell of periodic frame {f} has no volume; a periodic frame "
            "needs three independent lattice vectors, and a non-periodic "
            "system takes no cell"
        )


def _cells(box: Array) -> Array:
    """The cells with shape ``(nf, 3, 3)``, from a ``(nf, 9)`` or ``(nf, 3, 3)`` array."""
    xp = array_api_compat.array_namespace(box)
    return xp.reshape(box, (box.shape[0], 3, 3))


def _volume(box: Array) -> Array:
    """The cell volumes |a_1 . (a_2 x a_3)| with shape ``(nf,)``."""
    xp = array_api_compat.array_namespace(box)
    triple = box[:, 0, :] * xp.linalg.cross(box[:, 1, :], box[:, 2, :])
    return xp.abs(xp.sum(triple, axis=-1))


def _zero_strain(box: Array) -> Array:
    xp = array_api_compat.array_namespace(box)
    return xp.zeros(
        (box.shape[0], 3, 3), dtype=box.dtype, device=array_api_compat.device(box)
    )


def k_grid_bounds_of_cells(box: Array, k_max: float) -> np.ndarray:
    r"""Per-axis integer wave-number bounds the cells need at ``k_max``.

    A wave number :math:`\mathbf{k} = 2\pi B^{-1}\mathbf{n}` with
    :math:`|\mathbf{k}| \le k_{max}` has :math:`|n_d| \le k_{max} |a_d| / 2\pi`
    along every lattice vector :math:`\mathbf{a}_d` (Cauchy-Schwarz), so the
    bound of every frame is the ceiling of the right-hand side.

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
    length = np.linalg.norm(cell, axis=-1)
    return np.ceil(k_max * length / (2.0 * math.pi)).astype(np.int64)


def enumerate_k_grid_bounds(bounds: Array | list[int]) -> np.ndarray:
    r"""Enumerate the integer wave-number indices within per-axis bounds.

    The grid holds every :math:`\mathbf{n} \neq \mathbf{0}` with
    :math:`|n_d| \le n_d^{max}`, in lexicographic order with the first axis
    outermost. The zero vector is left out: the transform of a Coulomb-like
    kernel is singular there, and the tin-foil convention drops that mode.

    Parameters
    ----------
    bounds
        The per-axis bounds :math:`n_d^{max}` with shape ``(3,)``.

    Returns
    -------
    np.ndarray
        The integer indices with shape ``(nk, 3)``, int64.

    Raises
    ------
    ValueError
        If the bounds are not three non-negative integers.
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


K_BLOCK = 128
"""Wave numbers per block of a row of a :class:`KGrid`."""
# A guard point carries a nonzero wave number, whose kernel transform stays
# finite, and no weight.
_GUARD_K = (1e-6, 0.0, 0.0)
# The fewest entries of the block and slot axes, which keeps each a dynamic
# dimension of its own however few points or nodes a batch holds.
_MIN_ENTRIES = 2


def _k_blocks(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Lay the wave numbers of every frame out in blocks, padded with guard points.

    Returns the wave numbers with shape ``(nf, nb, K_BLOCK, 3)`` and their
    mask with shape ``(nf, nb, K_BLOCK)``.
    """
    longest = max([row.shape[0] for row in rows], default=0)
    blocks = max(-(-longest // K_BLOCK), _MIN_ENTRIES)
    vec = np.tile(np.asarray(_GUARD_K), (len(rows), blocks * K_BLOCK, 1))
    mask = np.zeros((len(rows), blocks * K_BLOCK), dtype=bool)
    for f, row in enumerate(rows):
        vec[f, : row.shape[0]] = row
        mask[f, : row.shape[0]] = True
    return (
        vec.reshape(len(rows), blocks, K_BLOCK, 3),
        mask.reshape(len(rows), blocks, K_BLOCK),
    )


def _frame_slots(n_node: Array) -> np.ndarray:
    """The nodes of every frame, one row each, padded with ``-1`` to the longest frame."""
    counts = np.asarray(to_numpy_array(n_node), dtype=np.int64).reshape(-1)
    width = max(int(counts.max(initial=0)), _MIN_ENTRIES)
    first = np.cumsum(counts) - counts
    column = np.arange(width, dtype=np.int64)[None, :]
    return np.where(column < counts[:, None], first[:, None] + column, -1)


def _k_grid(
    box: Array, vec: np.ndarray, mask: np.ndarray, slot: np.ndarray, coord: Array
) -> KGrid:
    xp = array_api_compat.array_namespace(box)
    device = array_api_compat.device(box)
    return KGrid(
        vec=xp.astype(xp.asarray(vec, device=device), box.dtype),
        mask=xp.asarray(mask, device=device),
        slot=xp.asarray(slot, device=device),
        volume=xp.astype(_volume(box), box.dtype),
        coord=coord,
        strain=_zero_strain(box),
    )


def build_k_grid(box: Array, k_max: float, coord: Array, n_node: Array) -> KGrid:
    r"""Enumerate the reciprocal lattice points of the periodic frames.

    With the lattice vectors as the rows of the cell :math:`B`, the wave
    numbers of a frame are :math:`\mathbf{k} = 2\pi B^{-1}\mathbf{n}` over
    integer :math:`\mathbf{n}` with :math:`0 < |\mathbf{k}| \le k_{max}`, so
    that :math:`\mathbf{a}_d \cdot \mathbf{k} = 2\pi n_d`. Their count is data
    dependent, so the enumeration runs eagerly. The points of every frame
    follow the lexicographic order of their integer indices, and the rows of
    the frames end in masked guard points up to the blocks of the longest
    row.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)`` or ``(nf, 9)``; rows are
        the lattice vectors.
    k_max
        The wave-number cutoff, in Å⁻¹.
    coord
        Node positions with shape ``(N, 3)``.
    n_node
        Per-frame node counts with shape ``(nf,)``; the frames occupy
        contiguous spans of the node axis.

    Returns
    -------
    KGrid
        The reciprocal lattice points of every frame, one row each.
    """
    box = _cells(box)
    cells = np.asarray(to_numpy_array(box), dtype=np.float64)
    bounds = k_grid_bounds_of_cells(cells, k_max)
    rows = []
    for f, cell in enumerate(cells):
        n_grid = enumerate_k_grid_bounds(bounds[f]).astype(np.float64)
        k = n_grid @ (2.0 * math.pi * np.linalg.inv(cell)).T
        rows.append(k[np.sum(k * k, axis=-1) <= k_max * k_max])
    vec, mask = _k_blocks(rows)
    return _k_grid(box, vec, mask, _frame_slots(n_node), coord)


def build_fixed_k_grid(
    box: Array, bounds: Array, k_max: float, coord: Array, n_node: Array
) -> KGrid:
    r"""Evaluate a fixed integer k grid on the cells of the periodic frames.

    Every frame shares the integer indices within ``bounds`` (see
    :func:`enumerate_k_grid_bounds`), so the blocks of a row have the static
    count of the grid. The wave numbers of a frame are :math:`\mathbf{k} =
    2\pi B^{-1}\mathbf{n}`, and the points beyond the cutoff
    :math:`|\mathbf{k}| \le k_{max}` and the guard points that complete the
    last block are masked out.

    A frame whose cell needs indices beyond the grid, a larger cell than the
    grid was sized for or a kernel whose cutoff has grown since, is evaluated
    on the grid alone: its sum is truncated to the points of the grid, which
    is issued as a warning. Its forces and virial remain the derivatives of
    that energy.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)`` or ``(nf, 9)``; rows are
        the lattice vectors.
    bounds
        The per-axis bounds :math:`n_d^{max}` of the shared integer grid with
        shape ``(3,)``.
    k_max
        The wave-number cutoff, in Å⁻¹.
    coord
        Node positions with shape ``(N, 3)``.
    n_node
        Per-frame node counts with shape ``(nf,)``; the frames occupy
        contiguous spans of the node axis.

    Returns
    -------
    KGrid
        The wave numbers of every frame over the shared grid, one row each.
    """
    box = _cells(box)
    cells = np.asarray(to_numpy_array(box), dtype=np.float64)
    n_grid = enumerate_k_grid_bounds(bounds)
    bounds = np.asarray(to_numpy_array(bounds), dtype=np.int64).reshape(3)
    required = k_grid_bounds_of_cells(cells, k_max)
    beyond = np.any(required > bounds[None, :], axis=1)
    if np.any(beyond):
        f = int(np.argmax(beyond))
        warnings.warn(
            f"frame {f} needs integer k-grid bounds {required[f].tolist()}, "
            f"beyond the fixed grid {bounds.tolist()}: its k-space sum is "
            "truncated to the grid; enlarge `k_grid_bounds` for the full sum",
            stacklevel=2,
        )
    rec = 2.0 * math.pi * np.linalg.inv(cells)
    k = np.einsum("jm,fdm->fjd", n_grid.astype(np.float64), rec)
    vec, real = _k_blocks(list(k))
    mask = real & (np.sum(vec * vec, axis=-1) <= k_max * k_max)
    return _k_grid(box, vec, mask, _frame_slots(n_node), coord)


def empty_k_grid(nf: int, coord: Array) -> KGrid:
    """The k grid of a batch without periodic frames: guard points and no nodes."""
    xp = array_api_compat.array_namespace(coord)
    device = array_api_compat.device(coord)
    eye = xp.tile(xp.eye(3, dtype=coord.dtype, device=device)[None, ...], (nf, 1, 1))
    vec, mask = _k_blocks([np.zeros((0, 3), dtype=np.float64)] * nf)
    return _k_grid(
        eye, vec, mask, np.full((nf, _MIN_ENTRIES), -1, dtype=np.int64), coord
    )


def _even_fft_size(n: int) -> int:
    """The smallest even integer of at least ``n`` with no prime factor above 7."""
    m = max(2, n + n % 2)
    while True:
        rest = m
        for p in (2, 3, 5, 7):
            while rest % p == 0:
                rest //= p
        if rest == 1:
            return m
        m += 2


def mesh_dims_of_cells(box: Array, spacing: float) -> tuple[int, int, int]:
    r"""The mesh that resolves every cell at ``spacing``.

    Along each lattice vector :math:`\mathbf{a}_d` the mesh holds the
    smallest even FFT-friendly number of points (no prime factor above 7) of
    at least :math:`\lceil \max_f |\mathbf{a}_d| / h \rceil`, the maximum
    running over the frames, so that no frame is evaluated with a spacing
    coarser than :math:`h`. The C++ interface sizes the mesh of a frozen
    model by the same rule.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)`` or ``(nf, 9)``; rows are
        the lattice vectors.
    spacing
        The largest mesh spacing :math:`h`, in Å.

    Returns
    -------
    tuple[int, int, int]
        The mesh dimensions.
    """
    cell = np.asarray(to_numpy_array(box), dtype=np.float64).reshape(-1, 3, 3)
    # The length is formed term by term, as the C++ interface forms it, so
    # that both round the ceiling identically.
    length = np.sqrt(
        cell[..., 0] * cell[..., 0]
        + cell[..., 1] * cell[..., 1]
        + cell[..., 2] * cell[..., 2]
    )
    longest = np.max(length, axis=0)
    n1, n2, n3 = (_even_fft_size(math.ceil(float(a) / spacing)) for a in longest)
    return n1, n2, n3


def _mesh_modes(xp: object, n: int, dtype: object, device: object) -> Array:
    """The integer mode indices of an axis of ``n`` points, in FFT order."""
    m = xp.arange(n, dtype=dtype, device=device)
    return xp.where(m > n // 2, m - n, m)


def mesh_grid_of_dims(
    box: Array, coord: Array, dims: tuple[int, int, int], mask: Array
) -> MeshGrid:
    """Pack cells, node positions and a mesh of the given dimensions.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)``; rows are the lattice
        vectors.
    coord
        Node positions with shape ``(N, 3)``.
    dims
        The even mesh dimensions.
    mask
        Periodic (1) vs non-periodic (0) frames with shape ``(nf,)``.

    Returns
    -------
    MeshGrid
        The cells, the mesh and the geometry of the frames.
    """
    xp = array_api_compat.array_namespace(box)
    device = array_api_compat.device(box)
    mode_1, mode_2, mode_3 = (_mesh_modes(xp, n, box.dtype, device) for n in dims)
    return MeshGrid(
        box=box,
        volume=xp.astype(_volume(box), box.dtype),
        coord=coord,
        strain=_zero_strain(box),
        mask=mask,
        mode_1=mode_1,
        mode_2=mode_2,
        mode_3=mode_3,
    )


def build_mesh_grid(box: Array, spacing: float, coord: Array) -> MeshGrid:
    """Pack the cells of the periodic frames and their shared mesh.

    Parameters
    ----------
    box
        Simulation cells with shape ``(nf, 3, 3)`` or ``(nf, 9)``; rows are
        the lattice vectors.
    spacing
        The largest mesh spacing, in Å (see :func:`mesh_dims_of_cells`).
    coord
        Node positions with shape ``(N, 3)``.

    Returns
    -------
    MeshGrid
        The cells, the mesh and the geometry of the frames, every frame
        periodic.
    """
    xp = array_api_compat.array_namespace(box)
    box = _cells(box)
    mask = xp.ones((box.shape[0],), dtype=xp.bool, device=array_api_compat.device(box))
    return mesh_grid_of_dims(box, coord, mesh_dims_of_cells(box, spacing), mask)


def empty_mesh_grid(nf: int, coord: Array) -> MeshGrid:
    """The mesh grid of a batch without periodic frames: every frame masked."""
    xp = array_api_compat.array_namespace(coord)
    device = array_api_compat.device(coord)
    box = xp.tile(xp.eye(3, dtype=coord.dtype, device=device)[None, ...], (nf, 1, 1))
    mask = xp.zeros((nf,), dtype=xp.bool, device=device)
    return mesh_grid_of_dims(box, coord, (2, 2, 2), mask)
