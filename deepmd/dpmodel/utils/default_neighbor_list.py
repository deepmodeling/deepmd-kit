# SPDX-License-Identifier: LGPL-3.0-or-later
"""Default neighbor-list builder with dense and cell-list search paths."""

from typing import (
    TYPE_CHECKING,
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
    xp_hint_dynamic_size,
    xp_scatter_sum,
    xp_take_along_axis,
)
from deepmd.dpmodel.utils.neighbor_list import (
    EdgeNeighborList,
)

from .neighbor_list import (
    NeighborList,
)
from .nlist import (
    apply_pair_exclusion_nlist,
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from .region import (
    normalize_coord,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )


# Dense broadcasting has a low fixed cost, but its backend scaling differs enough
# that one shared threshold causes regressions.  These conservative thresholds
# are the later crossover measured at rcut=3 and rcut=6 on an AMD EPYC 7K62 CPU
# and RTX 5090 GPU.  ``nloc`` counts local atoms before periodic ghost extension.
_NUMPY_CPU_PERIODIC_CELL_LIST_THRESHOLD = 32
_NUMPY_CPU_NONPERIODIC_CELL_LIST_THRESHOLD = 256
_TORCH_CPU_PERIODIC_CELL_LIST_THRESHOLD = 32
_TORCH_CPU_NONPERIODIC_CELL_LIST_THRESHOLD = 2048
_JAX_CPU_PERIODIC_CELL_LIST_THRESHOLD = 32
_JAX_CPU_NONPERIODIC_CELL_LIST_THRESHOLD = 256
_TF_CPU_PERIODIC_CELL_LIST_THRESHOLD = 512
_TF_CPU_NONPERIODIC_CELL_LIST_THRESHOLD = 4096
_TORCH_CUDA_PERIODIC_CELL_LIST_THRESHOLD = 1024
_TORCH_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD = 8192
_JAX_CUDA_PERIODIC_CELL_LIST_THRESHOLD = 2048
_JAX_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD = 12288
_TF_CUDA_PERIODIC_CELL_LIST_THRESHOLD = 1024
_TF_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD = 8192

# A Cartesian cell width of ``rcut`` guarantees that two atoms within the cutoff
# can differ by at most one cell in each direction.  Keeping the offsets as Python
# data avoids recreating the Cartesian product with backend-specific meshgrid APIs.
_NEIGHBOR_CELL_OFFSETS = tuple(
    (ii, jj, kk) for ii in (-1, 0, 1) for jj in (-1, 0, 1) for kk in (-1, 0, 1)
)

# Bound row padding to four times the compact edge count; more skewed candidate
# distributions keep the compact representation to avoid excessive memory use.
_PADDED_CANDIDATE_OVERHEAD_LIMIT = 4


def _supports_cell_list(coord: Array, nloc: Any, *, periodic: bool) -> bool:
    """Whether automatic dispatch should use a backend-safe cell-list path.

    All supported namespaces use the compact dynamic-candidate implementation.
    JAX neighbor-list construction runs eagerly outside the compiled model step,
    so its data-dependent ``repeat`` and ``nonzero`` result lengths are valid.  A
    symbolic ``nloc`` keeps the dense path because the threshold decision itself
    must remain a Python-level, shape-only choice while tracing.
    """
    if not isinstance(nloc, int):
        return False
    xp = array_api_compat.array_namespace(coord)
    is_numpy = array_api_compat.is_numpy_array(coord)
    is_jax = array_api_compat.is_jax_array(coord)
    is_torch = array_api_compat.is_torch_array(coord)
    if is_jax:
        import jax

        # The compact candidate axis is deliberately eager.  Preserve the dense
        # fallback for callers that do trace the entire builder, where JAX/XLA
        # requires ``repeat`` and ``nonzero`` result lengths to be static.
        if isinstance(coord, jax.core.Tracer):
            return False
    is_ndtensorflow = getattr(xp, "__name__", "") == "deepmd._vendors.ndtensorflow"
    if not (is_numpy or is_jax or is_torch or is_ndtensorflow):
        return False
    device = array_api_compat.device(coord)
    if is_torch and getattr(device, "type", None) not in ("cpu", "cuda"):
        # Cell-list primitives are only validated on PyTorch CPU/CUDA.  Treat
        # MPS, XPU, and future device types conservatively instead of assuming
        # that searchsorted/repeat/nonzero have complete backend coverage.
        return False
    if is_jax and getattr(device, "platform", None) not in ("cpu", "gpu"):
        # TPU and future JAX platforms need their own measurements and operation
        # coverage before inheriting either the CPU or CUDA crossover.
        return False
    device_name = str(device).upper()
    tf_cpu = is_ndtensorflow and (
        device_name.startswith("CPU") or "/DEVICE:CPU:" in device_name
    )
    tf_gpu = is_ndtensorflow and (
        device_name.startswith("GPU") or "/DEVICE:GPU:" in device_name
    )
    tf_unplaced = is_ndtensorflow and not device_name
    if is_ndtensorflow and not (tf_cpu or tf_gpu or tf_unplaced):
        # TensorFlow eager arrays identify CPU/GPU placement directly, while
        # symbolic tensors may remain unplaced until graph execution.  Reject
        # named accelerators that have not been validated by this path.
        return False
    is_cuda = (
        getattr(device, "type", None) == "cuda"
        or getattr(device, "platform", None) == "gpu"
        or tf_gpu
    )
    if is_cuda:
        if is_jax:
            threshold = (
                _JAX_CUDA_PERIODIC_CELL_LIST_THRESHOLD
                if periodic
                else _JAX_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD
            )
        elif is_ndtensorflow:
            threshold = (
                _TF_CUDA_PERIODIC_CELL_LIST_THRESHOLD
                if periodic
                else _TF_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD
            )
        else:
            threshold = (
                _TORCH_CUDA_PERIODIC_CELL_LIST_THRESHOLD
                if periodic
                else _TORCH_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD
            )
    elif is_jax:
        threshold = (
            _JAX_CPU_PERIODIC_CELL_LIST_THRESHOLD
            if periodic
            else _JAX_CPU_NONPERIODIC_CELL_LIST_THRESHOLD
        )
    elif tf_unplaced:
        # An unplaced TensorFlow graph may execute on either CPU or GPU.  The
        # later measured crossover avoids selecting the cell list too early on
        # whichever validated device the runtime eventually chooses.
        threshold = max(
            _TF_CPU_PERIODIC_CELL_LIST_THRESHOLD
            if periodic
            else _TF_CPU_NONPERIODIC_CELL_LIST_THRESHOLD,
            _TF_CUDA_PERIODIC_CELL_LIST_THRESHOLD
            if periodic
            else _TF_CUDA_NONPERIODIC_CELL_LIST_THRESHOLD,
        )
    elif is_ndtensorflow:
        threshold = (
            _TF_CPU_PERIODIC_CELL_LIST_THRESHOLD
            if periodic
            else _TF_CPU_NONPERIODIC_CELL_LIST_THRESHOLD
        )
    elif is_torch:
        threshold = (
            _TORCH_CPU_PERIODIC_CELL_LIST_THRESHOLD
            if periodic
            else _TORCH_CPU_NONPERIODIC_CELL_LIST_THRESHOLD
        )
    else:
        threshold = (
            _NUMPY_CPU_PERIODIC_CELL_LIST_THRESHOLD
            if periodic
            else _NUMPY_CPU_NONPERIODIC_CELL_LIST_THRESHOLD
        )
    return nloc >= threshold


def _select_nearest_padded(
    center: Array,
    neighbor_ext: Array,
    distance: Array,
    ncenters: int,
    nsel: int,
) -> Array | None:
    """Select neighbors by sorting independent, padded center rows.

    The compact candidate stream is already grouped by center.  Padding those
    groups into rows makes the center key implicit, so two row-wise stable sorts
    implement ``(distance, ext_index)`` ordering instead of three global sorts
    plus a final scatter.  Highly imbalanced rows can waste substantial memory;
    return ``None`` in that case so the compact global-sort path remains usable.

    This helper requires eager, concrete candidate counts because the maximum
    row width controls an allocation.  Callers must retain the compact path for
    traced namespaces with symbolic data-dependent dimensions.
    """
    xp = array_api_compat.array_namespace(center, neighbor_ext, distance)
    device = array_api_compat.device(center)
    ones = xp.ones_like(center)
    count_per_center = xp_scatter_sum(
        xp.zeros((ncenters,), dtype=xp.int64, device=device), 0, center, ones
    )
    max_candidates = int(xp.max(count_per_center))
    if max_candidates == 0:
        return xp.full((ncenters, nsel), -1, dtype=xp.int64, device=device)

    edge_count = center.shape[0]
    padded_size = ncenters * max_candidates
    if padded_size > _PADDED_CANDIDATE_OVERHEAD_LIMIT * max(edge_count, 1):
        return None

    center_start = xp.cumulative_sum(count_per_center) - count_per_center
    edge_iota = xp.cumulative_sum(ones) - 1
    rank = edge_iota - xp.take(center_start, center, axis=0)
    slot = center * max_candidates + rank

    neighbor_rows = xp_scatter_sum(
        xp.zeros((padded_size,), dtype=xp.int64, device=device),
        0,
        slot,
        neighbor_ext + 1,
    )
    neighbor_rows = xp.reshape(neighbor_rows - 1, (ncenters, max_candidates))
    distance_rows = xp_scatter_sum(
        xp.zeros((padded_size,), dtype=distance.dtype, device=device),
        0,
        slot,
        distance,
    )
    distance_rows = xp.reshape(distance_rows, (ncenters, max_candidates))
    distance_rows = xp.where(
        neighbor_rows >= 0,
        distance_rows,
        xp.full_like(distance_rows, float("inf")),
    )

    # Stable sorting the secondary key first preserves the dense builder's
    # extended-index tie break for equal distances.
    order = xp.argsort(neighbor_rows, axis=1, stable=True)
    neighbor_rows = xp_take_along_axis(neighbor_rows, order, axis=1)
    distance_rows = xp_take_along_axis(distance_rows, order, axis=1)
    order = xp.argsort(distance_rows, axis=1, stable=True)
    neighbor_rows = xp_take_along_axis(neighbor_rows, order, axis=1)

    selected_width = min(nsel, max_candidates)
    nlist = neighbor_rows[:, :selected_width]
    if selected_width < nsel:
        nlist = xp.concat(
            (
                nlist,
                xp.full(
                    (ncenters, nsel - selected_width),
                    -1,
                    dtype=xp.int64,
                    device=device,
                ),
            ),
            axis=1,
        )
    return nlist


def _supports_padded_selection(coord: Array) -> bool:
    """Whether row padding can use eager, data-dependent Python dimensions."""
    if array_api_compat.is_numpy_array(coord):
        return True
    if array_api_compat.is_torch_array(coord):
        import torch

        # torch.export/compile must keep the compact path: converting the maximum
        # candidate count to ``int`` would specialize an unbacked symbolic value.
        # Keep unmeasured accelerator implementations on the compact path too.
        device = array_api_compat.device(coord)
        return device.type in ("cpu", "cuda") and not torch.compiler.is_compiling()
    if array_api_compat.is_jax_array(coord):
        import jax

        # Neighbor-list construction normally runs eagerly before the compiled
        # model step; a tracer still needs the static compact fallback.  On JAX
        # accelerators, materializing the row width on the host costs more than
        # the saved sort work, so retain compact device sorting.
        device = array_api_compat.device(coord)
        return not isinstance(coord, jax.core.Tracer) and device.platform == "cpu"
    return False


def _build_neighbor_list_cell(
    coord: Array,
    atype: Array,
    nloc: int,
    rcut: float,
    nsel: int,
    pair_excl: "PairExcludeMask | None" = None,
) -> Array:
    """Build a fixed-width neighbor list from Cartesian spatial cells.

    Extended coordinates already contain all required periodic images, so the
    search itself is non-periodic: atoms are assigned to axis-aligned cells of
    width ``rcut`` and each local center examines only its 27 adjacent cells.
    Cell members are represented by sorted integer keys; ``searchsorted`` finds
    the member ranges and an array-valued ``repeat`` expands only real candidate
    pairs.  At constant density this uses O(N) candidate memory and O(N log N)
    work, instead of the dense O(N**2) distance matrix.  Eager CPU namespaces
    select neighbors with two bounded row-wise sorts, while traced namespaces
    and JAX GPU retain compact global sorting to avoid a host-dependent shape.

    The final stable lexicographic ordering is ``(center, distance, ext_index)``.
    It matches the dense builder's nearest-neighbor contract, including selecting
    before applying pair exclusions (excluded entries leave holes, without
    backfilling farther neighbors).
    """
    xp = array_api_compat.array_namespace(coord, atype)
    device = array_api_compat.device(coord)
    nframes, nall = atype.shape
    coord = xp.reshape(coord, (nframes, nall, 3))

    if nloc == 0:
        return xp.full((nframes, 0, nsel), -1, dtype=xp.int64, device=device)

    # Bin in Cartesian space.  Subtracting a per-frame origin keeps every cell
    # coordinate non-negative, which lets a single collision-free row-major key
    # encode the 3-D cell tuple.
    origin = xp.min(coord, axis=1, keepdims=True)
    cell = xp.astype(xp.floor((coord - origin) / rcut), xp.int64)
    dims = xp.max(xp.reshape(cell, (-1, 3)), axis=0) + 1
    cells_per_frame = dims[0] * dims[1] * dims[2]
    frame_base = xp.arange(nframes, dtype=xp.int64, device=device) * cells_per_frame
    cell_key = cell[..., 0] + dims[0] * (cell[..., 1] + dims[1] * cell[..., 2])
    cell_key = cell_key + frame_base[:, None]

    # Virtual atoms share a sentinel key outside the searchable cell range.  They
    # remain in the sorted array (preserving a static nall axis) but no valid query
    # ever searches the sentinel cell.
    sentinel = cells_per_frame * nframes
    cell_key = xp.where(atype >= 0, cell_key, sentinel)
    flat_key = xp.reshape(cell_key, (-1,))
    local_ext_index = xp.broadcast_to(
        xp.arange(nall, dtype=xp.int64, device=device)[None, :],
        (nframes, nall),
    )
    order = xp.argsort(flat_key, stable=True)
    sorted_key = xp.take(flat_key, order, axis=0)
    sorted_ext_index = xp.take(xp.reshape(local_ext_index, (-1,)), order, axis=0)

    # Query the 27 cells surrounding every local center.  Out-of-grid queries and
    # virtual centers get key -1; searchsorted then returns an empty interval.
    offsets = xp.asarray(_NEIGHBOR_CELL_OFFSETS, dtype=xp.int64, device=device)
    query_cell = cell[:, :nloc, None, :] + offsets[None, None, :, :]
    in_bounds = xp.all(
        (query_cell >= 0) & (query_cell < dims[None, None, None, :]), axis=-1
    )
    in_bounds = in_bounds & (atype[:, :nloc, None] >= 0)
    query_key = query_cell[..., 0] + dims[0] * (
        query_cell[..., 1] + dims[1] * query_cell[..., 2]
    )
    query_key = query_key + frame_base[:, None, None]
    query_key = xp.where(in_bounds, query_key, xp.full_like(query_key, -1))
    query_key = xp.reshape(query_key, (-1,))

    starts = xp.astype(xp.searchsorted(sorted_key, query_key, side="left"), xp.int64)
    ends = xp.astype(xp.searchsorted(sorted_key, query_key, side="right"), xp.int64)
    counts = ends - starts

    # Expand each cell query into its actual members.  The candidate length is
    # data-dependent; cumulative sums provide iotas without calling arange on an
    # unbacked symbolic size (important for torch.export-compatible namespaces).
    query_ids = xp.repeat(
        xp.arange(query_key.shape[0], dtype=xp.int64, device=device), counts
    )
    if not isinstance(query_ids.shape[0], int):
        xp_hint_dynamic_size(query_ids)
    query_output_start = xp.cumulative_sum(counts) - counts
    candidate_iota = xp.cumulative_sum(xp.ones_like(query_ids)) - 1
    member_offset = candidate_iota - xp.take(query_output_start, query_ids, axis=0)
    sorted_position = xp.take(starts, query_ids, axis=0) + member_offset
    neighbor_ext = xp.take(sorted_ext_index, sorted_position, axis=0)

    center = query_ids // len(_NEIGHBOR_CELL_OFFSETS)
    frame = center // nloc
    center_local = center - frame * nloc
    coord_flat = xp.reshape(coord, (nframes * nall, 3))
    center_coord = xp.take(coord_flat, frame * nall + center_local, axis=0)
    neighbor_coord = xp.take(coord_flat, frame * nall + neighbor_ext, axis=0)
    diff = neighbor_coord - center_coord
    # Use the same norm as the historical dense implementation for both cutoff
    # and ordering.  Sorting squared distances can distinguish values that round
    # to the same norm, changing the stable order of symmetry-equivalent images.
    distance = xp.linalg.vector_norm(diff, axis=-1)
    keep_mask = (neighbor_ext != center_local) & (distance <= rcut)
    (keep,) = xp.nonzero(keep_mask)
    keep = xp.reshape(keep, (-1,))
    if not isinstance(keep.shape[0], int):
        xp_hint_dynamic_size(keep)
    center = xp.take(center, keep, axis=0)
    neighbor_ext = xp.take(neighbor_ext, keep, axis=0)
    distance = xp.take(distance, keep, axis=0)

    ncenters = nframes * nloc
    if _supports_padded_selection(coord):
        nlist = _select_nearest_padded(center, neighbor_ext, distance, ncenters, nsel)
        if nlist is not None:
            nlist = xp.reshape(nlist, (nframes, nloc, nsel))
            return apply_pair_exclusion_nlist(nlist, atype, pair_excl)

    # Stable sorts from the least- to most-significant key produce the desired
    # lexicographic order while using only standard Array API operations.
    order = xp.argsort(neighbor_ext, stable=True)
    order = xp.take(
        order,
        xp.argsort(xp.take(distance, order, axis=0), stable=True),
        axis=0,
    )
    order = xp.take(
        order,
        xp.argsort(xp.take(center, order, axis=0), stable=True),
        axis=0,
    )
    center = xp.take(center, order, axis=0)
    neighbor_ext = xp.take(neighbor_ext, order, axis=0)

    ones = xp.ones_like(center)
    count_per_center = xp_scatter_sum(
        xp.zeros((ncenters,), dtype=xp.int64, device=device), 0, center, ones
    )
    center_start = xp.cumulative_sum(count_per_center) - count_per_center
    edge_iota = xp.cumulative_sum(ones) - 1
    rank = edge_iota - xp.take(center_start, center, axis=0)
    (selected,) = xp.nonzero(rank < nsel)
    selected = xp.reshape(selected, (-1,))
    if not isinstance(selected.shape[0], int):
        xp_hint_dynamic_size(selected)
    selected_center = xp.take(center, selected, axis=0)
    selected_rank = xp.take(rank, selected, axis=0)
    selected_neighbor = xp.take(neighbor_ext, selected, axis=0)

    # Each (center, rank) slot is unique, so scatter-add into a zero array and
    # store ``neighbor + 1``; subtracting one afterwards creates -1 padding.
    slot = selected_center * nsel + selected_rank
    nlist = xp_scatter_sum(
        xp.zeros((ncenters * nsel,), dtype=xp.int64, device=device),
        0,
        slot,
        selected_neighbor + 1,
    )
    nlist = xp.reshape(nlist - 1, (nframes, nloc, nsel))
    return apply_pair_exclusion_nlist(nlist, atype, pair_excl)


class DefaultNeighborList(NeighborList):
    """Adaptive builder using dense search for small systems and spatial cells
    for larger systems.

    Both paths first replicate the cell into periodic images with
    :func:`~deepmd.dpmodel.utils.nlist.extend_coord_with_ghosts`.  Small systems
    retain the historical all-pairs distance matrix for its lower constant
    overhead.  Larger supported arrays use a Cartesian cell list whose candidate
    count is linear at constant density.  JAX builds this data-dependent list
    eagerly before the fixed-shape model computation enters ``jit``.
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
        # Types are distinguished in the lower interface, so keep them merged
        # here.  The dense path remains faster for small systems; the spatial
        # path avoids the nloc*nall distance matrix for larger supported arrays.
        if _supports_cell_list(coord, nloc, periodic=box is not None):
            nlist = _build_neighbor_list_cell(
                extended_coord,
                extended_atype,
                nloc,
                rcut,
                sum(sel),
                pair_excl=pair_excl,
            )
        else:
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
