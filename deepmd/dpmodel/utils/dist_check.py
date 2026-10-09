# SPDX-License-Identifier: LGPL-3.0-or-later
r"""Pair-clearance check for frame validity filtering.

A frame is admissible when every atom pair keeps the clearance the
training-frame filter prescribes. That clearance is a fraction of each pair's
own length scale, so the check is scale-free: the quantity compared against
one is the margin

.. math::

   m = \min_{i<j} \frac{d_{ij}}{h_{t_i} + h_{t_j}},

with :math:`h_t` the half-threshold of type :math:`t`. A radius given directly
in Å is the same rule measured against a unit length scale, where every type
carries the same half-threshold and the margin reduces to the minimum pair
distance divided by that radius.
"""

from __future__ import (
    annotations,
)

import functools
from typing import (
    Any,
)

import numpy as np

from deepmd.utils.element_radii import (
    UNIT_CONTACT_RADIUS,
    covalent_radii_from_type_map,
)

#: Pairs a row block of the single-frame scan holds at once.
_MIN_PAIR_DIST_BLOCK_PAIRS = 262_144
#: Frames a block of the batched scan needs to be worth its fixed cost. Below
#: it the frames are scanned one by one, which is what the batched scan spends
#: its per-block array operations on.
_MIN_PAIR_DIST_BATCH_FRAMES = 8


def pair_half_thresholds(
    fraction: float,
    scale: str,
    type_map: list[str] | None = None,
    ntypes: int | None = None,
) -> np.ndarray:
    """
    Tabulate the half-threshold each type contributes to a pair.

    A pair's threshold is ``half[a] + half[b]``. Under the ``"covalent"`` scale
    that sum is the given fraction of the pair's covalent bond length, so one
    fraction describes every element combination; under ``"absolute"`` every
    type carries the unit radius, the sum is the fraction itself, and the
    margin reduces to the pair distance divided by a threshold in Å.

    Parameters
    ----------
    fraction : float
        Filter radius as a fraction of the pair's length scale, or directly in
        Å under the ``"absolute"`` scale.
    scale : str
        Either ``"covalent"`` or ``"absolute"``.
    type_map : list[str], optional
        Element symbols, required by the ``"covalent"`` scale.
    ntypes : int, optional
        Number of types; taken from ``type_map`` when omitted.

    Returns
    -------
    np.ndarray
        Half-thresholds in Å with shape ``(ntypes,)``.

    Raises
    ------
    ValueError
        If the fraction is not positive, if the scale is unknown, if the
        covalent scale is requested without a type map, or if the map does not
        cover every type.
    """
    if fraction <= 0.0:
        raise ValueError(f"A pair window needs a positive radius, got {fraction}.")
    if scale not in ("covalent", "absolute"):
        raise ValueError(
            f"Unknown pair length scale {scale!r}; expected 'covalent' or 'absolute'."
        )
    if scale == "absolute":
        if ntypes is None:
            if type_map is None:
                raise ValueError(
                    "An absolute pair length scale needs either `ntypes` or a "
                    "`type_map` to size its table."
                )
            ntypes = len(type_map)
        return np.full(ntypes, fraction * UNIT_CONTACT_RADIUS, dtype=np.float64)
    if type_map is None or (ntypes is not None and len(type_map) != ntypes):
        raise ValueError(
            "A covalent pair length scale needs a `type_map` covering all "
            f"{ntypes if ntypes is not None else 'model'} types."
        )
    return fraction * covalent_radii_from_type_map(type_map)


@functools.lru_cache(maxsize=8)
def requirement_half_thresholds(
    fraction: float, scale: str, type_map: tuple[str, ...] | None, ntypes: int
) -> np.ndarray:
    """
    Resolve and retain the half-thresholds a derived margin measures against.

    A reader scans every frame and every batch against one window and one type
    map, so the table is resolved once per combination instead of once per
    scan. The result is shared and therefore read-only.

    Parameters
    ----------
    fraction : float
        Filter radius, as a fraction of the pair length scale.
    scale : str
        Either ``"covalent"`` or ``"absolute"``.
    type_map : tuple[str, ...], optional
        Type names, required by the ``"covalent"`` scale.
    ntypes : int
        Number of model types.

    Returns
    -------
    np.ndarray
        Half-thresholds in Å with shape ``(ntypes,)``.
    """
    table = pair_half_thresholds(
        fraction,
        scale,
        type_map=None if type_map is None else list(type_map),
        ntypes=ntypes,
    )
    table.flags.writeable = False
    return table


def compute_min_pair_margin_single(
    coord: np.ndarray,
    box: np.ndarray | None,
    atype: np.ndarray,
    half: np.ndarray,
    screened: bool = False,
) -> float:
    """
    Compute the minimum pair margin of a single frame.

    Pairs are visited in bounded row blocks, every one of them unless the scan
    is screened.

    Parameters
    ----------
    coord : np.ndarray
        Atomic coordinates, flattened with shape (natoms * 3,)
        or reshaped as (natoms, 3), in Å.
    box : np.ndarray or None
        Box vectors with shape (9,) for PBC, or None for non-PBC.
    atype : np.ndarray
        Atom types with shape (natoms,). Virtual atoms (type < 0)
        are excluded from the check.
    half : np.ndarray
        Half-threshold of every type with shape (ntypes,) in Å.
    screened : bool
        Whether the scan may stop at a margin of one. A caller that only asks
        whether the frame clears its window learns that from the first pair
        inside it, and the remaining pairs of a failing frame cost the scan its
        whole cost on the largest frames.

    Returns
    -------
    float
        Minimum pair margin, or inf if fewer than 2 real atoms exist. A
        screened scan returns the exact margin for a frame that clears its
        window and some margin below one for a frame that does not.
    """
    coord = coord.reshape(-1, 3)

    # === Step 1. Filter out virtual atoms ===
    real_mask = atype.ravel() >= 0
    real_coord = coord[real_mask]
    real_half = np.asarray(half, dtype=np.float64)[atype.ravel()[real_mask]]
    n_real = real_coord.shape[0]
    if n_real < 2:
        return float("inf")

    # === Step 2. Prepare minimum image convention for PBC ===
    if box is not None:
        cell = box.reshape(3, 3)
        inv_cell = np.linalg.inv(cell)
    else:
        cell = None
        inv_cell = None

    # === Step 3. Compute margins in bounded row blocks ===
    block_size = max(1, min(n_real, _MIN_PAIR_DIST_BLOCK_PAIRS // n_real))
    min_margin_sq = float("inf")
    stop_below_sq = 1.0 if screened else 0.0
    for start in range(0, n_real, block_size):
        stop = min(start + block_size, n_real)
        diff = real_coord[np.newaxis, :, :] - real_coord[start:stop, np.newaxis, :]

        if cell is not None and inv_cell is not None:
            frac_diff = diff @ inv_cell
            frac_diff -= np.round(frac_diff)
            diff = frac_diff @ cell

        threshold = real_half[np.newaxis, :] + real_half[start:stop, np.newaxis]
        margin_sq = np.sum(diff * diff, axis=-1) / (threshold * threshold)
        rows = np.arange(stop - start, dtype=np.int64)
        margin_sq[rows, start + rows] = np.inf
        min_margin_sq = min(min_margin_sq, float(margin_sq.min()))
        # A screened scan is settled by the first pair strictly inside its
        # window; an exhaustive one can stop only at zero, the margin's floor.
        if min_margin_sq < stop_below_sq or min_margin_sq == 0.0:
            break

    return float(np.sqrt(min_margin_sq))


def compute_min_pair_margin_batch(
    coord: np.ndarray,
    box: np.ndarray | None,
    atype: np.ndarray,
    half: np.ndarray,
    n_node: np.ndarray | None = None,
    screened: bool = False,
) -> np.ndarray:
    """
    Compute the minimum pair margin of every frame of a batch.

    A batch is scanned in one pass over the frames that share an atom count
    rather than one pass per frame, and the scan of a frame visits the atom
    pairs that lie close along one lattice direction instead of every pair
    (see :func:`_min_pair_margin_block`).

    Parameters
    ----------
    coord : np.ndarray
        Atomic coordinates, either `(nframes, natoms, 3)` (equivalently
        `(nframes, natoms * 3)`) for a rectangular batch, or `(nnodes, 3)`
        (equivalently `(nnodes * 3,)`) on the flat node axis of a ragged one,
        in Å.
    box : np.ndarray or None
        Box vectors with shape `(nframes, 9)` for periodic frames, or None.
    atype : np.ndarray
        Atom types, `(nframes, natoms)` or `(nnodes,)`. Virtual atoms
        (type < 0) are excluded from the check.
    half : np.ndarray
        Half-threshold of every type with shape `(ntypes,)` in Å.
    n_node : np.ndarray or None
        Per-frame atom counts of a ragged batch, with shape `(nframes,)`.
        None marks a rectangular batch.
    screened : bool
        Bound the scan of a frame at a margin of one, which is all the frame
        filter reads. A frame holding a pair inside its window then keeps its
        exact margin, and a frame clearing it carries a margin above one that
        it reaches.

    Returns
    -------
    np.ndarray
        Margin of every frame, with shape `(nframes,)`. A frame without a pair
        of real atoms carries inf.
    """
    half = np.asarray(half, dtype=np.float64)
    boxes = None if box is None else np.asarray(box).reshape(-1, 9)
    if n_node is None:
        atype = np.asarray(atype).reshape(len(atype), -1)
        nframes, natoms = atype.shape
        coord = np.asarray(coord).reshape(nframes, natoms, 3)
        real = atype >= 0
        if natoms > 0 and real.all():
            return _scan_uniform(coord, half[atype], boxes, screened)
        counts = real.sum(axis=1)
        positions = coord[real]
        halves = half[atype[real]]
    else:
        n_node = np.asarray(n_node).reshape(-1)
        coord = np.asarray(coord).reshape(-1, 3)
        atype = np.asarray(atype).reshape(-1)
        real = atype >= 0
        counts = np.bincount(
            np.repeat(np.arange(n_node.shape[0], dtype=np.int64), n_node)[real],
            minlength=n_node.shape[0],
        )
        positions = coord[real]
        halves = half[atype[real]]
    return _scan_by_atom_count(positions, halves, boxes, counts, screened)


def _scan_by_atom_count(
    positions: np.ndarray,
    halves: np.ndarray,
    boxes: np.ndarray | None,
    counts: np.ndarray,
    screened: bool,
) -> np.ndarray:
    """Scan frames of real atoms laid out on one flat node axis.

    Frames of equal atom count form the rectangular blocks the scan works on.
    A batch of the LMDB reader normally holds one count, and a padded
    rectangular batch reduces to one as soon as its filler atoms are left out,
    so this usually yields a single block; a block too small to pay for the
    batched scan is served frame by frame instead.

    Parameters
    ----------
    positions : np.ndarray
        Coordinates of the real atoms of every frame in order, `(nnodes, 3)`.
    halves : np.ndarray
        Half-threshold of each of those atoms, `(nnodes,)` in Å.
    boxes : np.ndarray or None
        Box vectors with shape `(nframes, 9)`, or None for open boundaries.
    counts : np.ndarray
        Real atom count of every frame, with shape `(nframes,)`.
    screened : bool
        Whether the scan is bounded at a margin of one.

    Returns
    -------
    np.ndarray
        Margin of every frame, with shape `(nframes,)`.
    """
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    margins = np.full(counts.shape[0], np.inf, dtype=np.float64)
    for size in np.unique(counts):
        if size < 2:
            continue
        rows = np.flatnonzero(counts == size)
        block_boxes = None if boxes is None else boxes[rows]
        if rows.shape[0] < _MIN_PAIR_DIST_BATCH_FRAMES:
            # The half-thresholds of these atoms are already resolved, so each
            # atom stands for its own type and the lookup becomes the identity.
            all_real = np.arange(size, dtype=np.int64)
            for index, row in enumerate(rows):
                start = offsets[row]
                cell = None if block_boxes is None else block_boxes[index]
                margins[row] = compute_min_pair_margin_single(
                    positions[start : start + size],
                    None if cell is None or not cell.any() else cell,
                    all_real,
                    halves[start : start + size],
                    screened=screened,
                )
            continue
        nodes = offsets[rows][:, np.newaxis] + np.arange(size, dtype=np.int64)
        margins[rows] = _scan_uniform(
            positions[nodes], halves[nodes], block_boxes, screened
        )
    return margins


def _scan_uniform(
    coord: np.ndarray,
    halves: np.ndarray,
    boxes: np.ndarray | None,
    screened: bool,
) -> np.ndarray:
    """Scan frames of one atom count, all of whose atoms are real.

    A cell of all zeros marks a frame without periodicity, which the readers
    may mix with periodic ones; the two kinds are scanned apart because the
    minimum image of a frame only exists with a cell.

    Parameters
    ----------
    coord : np.ndarray
        Coordinates with shape `(nframes, natoms, 3)`.
    halves : np.ndarray
        Half-threshold of every atom, `(nframes, natoms)` in Å.
    boxes : np.ndarray or None
        Box vectors with shape `(nframes, 9)`, or None for open boundaries.
    screened : bool
        Whether the scan is bounded at a margin of one.

    Returns
    -------
    np.ndarray
        Margin of every frame, with shape `(nframes,)`.
    """
    nframes, natoms = coord.shape[:2]
    if natoms < 2:
        return np.full(nframes, np.inf, dtype=np.float64)
    if boxes is None:
        return _min_pair_margin_block(coord, halves, None, screened)
    periodic = ~np.all(boxes == 0.0, axis=1)
    if periodic.all():
        return _min_pair_margin_block(coord, halves, boxes, screened)
    margins = np.empty(nframes, dtype=np.float64)
    if periodic.any():
        margins[periodic] = _min_pair_margin_block(
            coord[periodic], halves[periodic], boxes[periodic], screened
        )
    margins[~periodic] = _min_pair_margin_block(
        coord[~periodic], halves[~periodic], None, screened
    )
    return margins


def _min_pair_margin_block(
    coord: np.ndarray,
    halves: np.ndarray,
    box: np.ndarray | None,
    screened: bool = False,
) -> np.ndarray:
    """Minimum pair margin of frames of equal atom count.

    The atoms of every frame are ordered along one direction -- the lattice
    planes that lie furthest apart, or the widest Cartesian spread without a
    cell -- and each atom is paired with its successors in that order. A pair
    can only improve a frame's margin `m` when its distance falls below
    `m` times its own threshold, hence below `m` times twice the largest
    half-threshold the frame carries; the separation along the ordering
    direction only grows with the distance in the order, so a frame is
    finished as soon as every atom's window reaches that far. The scan
    therefore visits the successors that lie within one slab instead of every
    pair, and its bound is a bound on the margin rather than on the distance:
    the pair that minimizes the margin need not be the pair that minimizes the
    distance, and the scan never assumes it is.

    Under periodicity the order is cyclic and every unordered pair is met from
    both of its atoms, at the windows `shift` and `natoms - shift`. That is
    what lets the advance be measured in the direction of the order: the pair
    whose periodic image is close the other way around is met from its other
    atom, where the advance is short.

    Parameters
    ----------
    coord : np.ndarray
        Coordinates with shape `(nframes, natoms, 3)`. Every atom counts.
    halves : np.ndarray
        Half-threshold of every atom, `(nframes, natoms)` in Å.
    box : np.ndarray or None
        Box vectors with shape `(nframes, 9)`, or None for open boundaries.
    screened : bool
        Whether the scan is bounded at a margin of one.

    Returns
    -------
    np.ndarray
        Margin of every frame, with shape `(nframes,)`.
    """
    nframes, natoms = coord.shape[:2]
    if natoms < 2:
        return np.full(nframes, np.inf, dtype=np.float64)

    coord = np.ascontiguousarray(coord, dtype=np.float64)
    if box is None:
        # Open boundaries: the order follows a Cartesian axis and a separation
        # along it is already a distance.
        position = coord
        metric = None
        spacing = np.ones((nframes, 3), dtype=np.float64)
        extent = np.ptp(coord, axis=1)
    else:
        cell = np.asarray(box, dtype=np.float64).reshape(nframes, 3, 3)
        inverse = np.linalg.inv(cell)
        position = coord @ inverse
        position -= np.floor(position)
        metric = cell @ np.swapaxes(cell, -1, -2)
        # Perpendicular width of each family of lattice planes: a fractional
        # separation of one along an axis is this distance in space.
        spacing = 1.0 / np.linalg.norm(np.swapaxes(inverse, -1, -2), axis=-1)
        extent = spacing

    # One direction serves the whole block, so it is the one that separates
    # the atoms of its worst frame the most.
    axis = int(np.argmax(extent.min(axis=0)))
    order = np.argsort(position[:, :, axis], axis=1)
    rows = np.arange(nframes, dtype=np.int64)[:, np.newaxis]
    position = position[rows, order]
    # Half-thresholds follow the atoms into the order, and the largest of a
    # frame turns its margin bound into a distance the scan can measure.
    halves = np.asarray(halves, dtype=np.float64)[rows, order]
    widest = (2.0 * halves.max(axis=1)) ** 2

    minima = np.full(nframes, np.inf, dtype=np.float64)
    active = np.arange(nframes, dtype=np.int64)
    shift = 1
    cyclic = box is not None
    while active.size and shift < natoms:
        window = position[active]
        threshold = halves[active]
        partner = np.roll(window, -shift, axis=1) if cyclic else window[:, shift:]
        head = window if cyclic else window[:, : natoms - shift]
        partner_half = (
            np.roll(threshold, -shift, axis=1) if cyclic else threshold[:, shift:]
        )
        head_half = threshold if cyclic else threshold[:, : natoms - shift]
        separation = partner - head
        if cyclic:
            separation -= np.round(separation)
            # The quadratic form is taken in two steps so that the metric
            # enters as a matrix product, which reaches BLAS; the three-operand
            # contraction falls back to an unblocked loop and costs twice as
            # much for the same result.
            squared = np.einsum("fnx,fnx->fn", separation @ metric[active], separation)
        else:
            squared = np.sum(separation * separation, axis=-1)
        pair_threshold = head_half + partner_half
        minima[active] = np.minimum(
            minima[active], (squared / (pair_threshold * pair_threshold)).min(axis=1)
        )

        pending = minima[active]
        if screened:
            # No frame needs its window to travel further than one margin's
            # worth: beyond that only a pair already outside its own window
            # could still be found, which cannot change the frame's side. A
            # frame that has reached one therefore keeps the exact margin,
            # because from then on this bound is its running minimum.
            pending = np.minimum(pending, 1.0)
        advance = partner[:, :, axis] - head[:, :, axis]
        if cyclic:
            advance -= np.floor(advance)
        reach = advance * spacing[active, axis, np.newaxis]
        settled = (reach**2 >= (pending * widest[active])[:, np.newaxis]).all(axis=1)
        active = active[~settled]
        shift += 1
    return np.sqrt(minima)


def pair_margin_frame_mask(
    batch: dict[str, Any],
    enabled: bool,
) -> np.ndarray | None:
    """Mark the frames of a batch whose atom pairs all clear their window.

    The filter radius the requirement was registered with is already spent:
    it sized the per-type half-thresholds the margin was measured against.
    What is left here is the same comparison against one for every length
    scale, so the only decision this function still takes from the
    configuration is whether the filter runs at all.

    Parameters
    ----------
    batch : dict[str, Any]
        Normalized NumPy batch carrying the derived `pair_margin` field,
        which holds each frame's margin against its own pair thresholds.
    enabled : bool
        Whether the pair-clearance filter is configured for this data.

    Returns
    -------
    np.ndarray or None
        Boolean mask with shape (nframes,), or None when the filter is
        disabled.

    Raises
    ------
    RuntimeError
        If the batch carries a placeholder instead of a derived margin,
        which would otherwise accept every frame silently.
    """
    if not enabled:
        return None
    if not batch.get("find_pair_margin", False):
        raise RuntimeError(
            "The pair-clearance filter is enabled, but the data source did not "
            "derive the pair margin of the batch."
        )
    return batch["pair_margin"].reshape(-1) >= 1.0


def select_frames(batch: dict[str, Any], frame_mask: np.ndarray) -> dict[str, Any]:
    """Keep the masked frames of a normalized NumPy batch.

    A rectangular batch leads every per-frame and per-atom array with the
    frame axis. A ragged batch, which carries `n_node`, leads its per-frame
    arrays with the frame axis as well but concatenates its per-atom arrays
    frame by frame on one flat node axis, with a whole number of rows per atom;
    those are sliced by the node mask that the frame mask expands to.

    Parameters
    ----------
    batch : dict[str, Any]
        Normalized NumPy batch, rectangular or ragged.
    frame_mask : np.ndarray
        Boolean mask with shape (nframes,).

    Returns
    -------
    dict[str, Any]
        New batch in which every array on the frame axis or on the flat node
        axis is sliced; all other entries pass through.
    """
    nframes = frame_mask.shape[0]
    n_node = batch.get("n_node")
    nnodes = 0 if n_node is None else int(n_node.sum())
    node_mask = None if n_node is None else np.repeat(frame_mask, n_node)

    def select(value: Any) -> Any:
        if not isinstance(value, np.ndarray) or value.ndim == 0:
            return value
        # Frames of one atom make the two axes coincide, and the two masks too.
        if value.shape[0] == nframes:
            return value[frame_mask]
        if nnodes > 0 and value.shape[0] % nnodes == 0:
            return value[np.repeat(node_mask, value.shape[0] // nnodes)]
        return value

    return {key: select(value) for key, value in batch.items()}
