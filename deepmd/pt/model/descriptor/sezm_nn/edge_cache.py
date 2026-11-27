# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Edge cache construction utilities for SeZM.

This module defines the shared procedures that assemble per-edge geometry,
radial features, rotation blocks, and normalization terms used by the SeZM
descriptor.
"""

from __future__ import (
    annotations,
)

import functools
import logging
import math
from collections.abc import (
    Callable,
)
from typing import (
    NamedTuple,
)

import torch
import torch.nn.functional as F
from einops import (
    rearrange,
)

from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
    clamped_edge_geometry,
)

from .utils import (
    get_promoted_dtype,
    nvtx_range,
    safe_norm,
)
from .wignerd import (
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)

WignerCalculatorFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
EdgeTypeKeepMaskFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
# Distance and keep weight to the keep-weighted envelope and radial basis, the
# fused replacement of applying the two modules separately.
FusedRadialFn = Callable[
    [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


class EdgeFeatureCache(NamedTuple):
    """
    Global edge feature cache created once per forward().

    All tensors are aligned on the same edge axis (E = number of valid edges).

    Parameters
    ----------
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    edge_type_feat
        Per-edge type embeddings with shape (E, C), computed as src+dst.
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_len
        Regularized read distances with shape (E, 1) in Å.
    edge_rbf
        Radial basis with shape (E, n_radial).
        The C^3 cutoff envelope is already baked in.
    edge_env
        C^3 cutoff envelope weights with shape (E, 1).
    deg
        Envelope-squared smooth degree with shape (N,), computed as
        ``sum(edge_env**2)`` over incoming edges.
        Used for smooth normalization in EnvironmentInitialEmbedding.
    inv_sqrt_deg
        Inverse square root smooth degree normalization with shape (N, 1, 1).
    D_full
        Block-diagonal Wigner-D matrix with shape (E, D, D) where D=(lmax+1)^2.
        Used for efficient batched rotation. None if not available.
    Dt_full
        Transpose of D_full with shape (E, D, D). None if not available.
    edge_quat
        Per-edge global-to-local quaternion actually used to build ``D_full`` and
        ``Dt_full`` with shape (E, 4). Includes the optional random local-Z roll.
    D_packed
        Structural Wigner band with shape ``(E, 3*D-2)`` (the ``m = 0/-1/+1``
        rows per degree block, flash packed convention). Present only when the
        banded inference storage is enabled and supported; then ``D_full`` and
        ``Dt_full`` are ``None`` and the fused reduced-layout consumers address
        this band instead. None otherwise.
    D_to_m_cache
        Lazy cache for projected D matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    csr_cache
        Lazy cache for endpoint CSR views used by segmented accelerated
        operators, keyed by endpoint role (``"dst"`` or ``"src"``). Built once
        per step and shared by every consumer.
    """

    src: torch.Tensor
    dst: torch.Tensor
    edge_type_feat: torch.Tensor
    edge_vec: torch.Tensor
    edge_len: torch.Tensor
    edge_rbf: torch.Tensor
    edge_env: torch.Tensor
    deg: torch.Tensor
    inv_sqrt_deg: torch.Tensor
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    D_to_m_cache: dict[str, torch.Tensor] | None = None
    Dt_from_m_cache: dict[str, torch.Tensor] | None = None
    csr_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    edge_quat: torch.Tensor | None = None
    D_packed: torch.Tensor | None = None


log = logging.getLogger(__name__)


@functools.cache
def _warn_packed_inert() -> None:
    """Say once that the band was admitted but the calculator cannot build it."""
    log.warning(
        "The blocks admit the banded Wigner layout, but the resolved "
        "Wigner calculator cannot produce it (an unsupported degree); "
        "building dense Wigner surfaces."
    )


def cached_edge_csr(
    edge_cache: EdgeFeatureCache, endpoint: str, n_node: int | torch.SymInt
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the CSR view of one edge endpoint, built once per step.

    Several accelerated operators walk the edges of one endpoint in segment
    order: the fused convolution and the initial embedding on the CUDA path,
    the flash aggregation and the rotate-mix backward on the Triton path. They
    all share one edge set, so the sorted view is built once and kept on the
    edge cache; whichever consumer runs first pays for it.

    Parameters
    ----------
    edge_cache : EdgeFeatureCache
        The step's edge feature cache.
    endpoint : str
        ``"dst"`` or ``"src"``.
    n_node : int or torch.SymInt
        Number of nodes the endpoint indexes into.

    Returns
    -------
    tuple of torch.Tensor
        The stable sorting permutation with shape (E,) and the row pointer with
        shape (n_node + 1,), both int64. Stability fixes the within-segment
        edge order, which is what makes the segment reductions bitwise
        reproducible.
    """
    store = edge_cache.csr_cache
    cached = None if store is None else store.get(endpoint)
    if cached is not None:
        return cached
    key = getattr(edge_cache, endpoint)
    order = torch.argsort(key, dim=0, stable=True)
    counts = key.new_zeros(n_node).scatter_add(0, key, torch.ones_like(key))
    row_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])
    if store is not None:
        store[endpoint] = (order, row_ptr)
    return order, row_ptr


@torch.amp.autocast("cuda", enabled=False)
def build_edge_cache(
    *,
    type_ebed: torch.Tensor,
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    pair_keep_mask: torch.Tensor,
    eps: float,
    deg_norm_floor: float,
    edge_envelope: Callable[[torch.Tensor], torch.Tensor],
    radial_basis: Callable[[torch.Tensor], torch.Tensor],
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    packed_wigner: bool = False,
) -> EdgeFeatureCache:
    """
    Build the global edge cache from DeePMD padded neighbor list.

    This converts DeePMD's per-frame padded neighbor list into a flat list of
    valid edges used by message passing, and computes all per-edge tensors that
    are reused across blocks.

    The resulting cache contains:

    - per-edge endpoints: ``src``, ``dst`` and per-edge type features: ``edge_type_feat`` (src+dst)
    - per-edge geometry: ``edge_vec``
    - per-edge smooth weights: C^3 cutoff envelope ``edge_env``
    - per-edge radial basis: ``edge_rbf`` (envelope already baked in)
    - per-edge rotation blocks: block-diagonal Wigner-D matrices ``D_full`` and ``Dt_full``
    - destination-node smooth normalization: ``inv_sqrt_deg`` from
      envelope-squared degree ``sum(edge_env**2)``

    Notes
    -----
    Input formats follow DeePMD conventions:

    - ``extended_coord`` has shape ``(nf, nall, 3)``.
    - ``nlist`` has shape ``(nf, nloc, nnei)`` and stores indices into the extended axis
      (``0..nall-1``), with ``-1`` indicating padding.
    - ``mapping`` (when provided) maps extended indices to local indices ``0..nloc-1``.
      When ``mapping`` is ``None``, the function assumes the neighbor indices are already local.

    This function builds the edge cache directly on the valid edge set, so
    padded or excluded neighbor slots never enter the geometry, radial basis,
    or Wigner-D evaluation.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C), where N=nf*nloc.
    extended_coord
        Extended coordinates with shape (nf, nall, 3).
    nlist
        Neighbor list with shape (nf, nloc, nnei).
    mapping
        Mapping from extended indices to local indices with shape (nf, nall), or None.
    pair_keep_mask
        Pair keep mask from `PairExcludeMask` with shape (nf, nloc, nnei). True means keep.
    eps
        Small positive epsilon for safe norm.
    deg_norm_floor
        Floor added to the envelope-squared degree before inverse-sqrt
        normalization (see :func:`_finalize_edge_cache`).
    edge_envelope
        C^3 edge envelope module.
    radial_basis
        Radial basis module.
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.

    Returns
    -------
    EdgeFeatureCache
        Per-edge cache.
    """
    nf, nloc, nnei = nlist.shape
    n_nodes = nf * nloc

    # === Step 1. Force fp32+ for geometry ===
    geom_dtype = get_promoted_dtype(extended_coord.dtype)
    coord = extended_coord.to(dtype=geom_dtype)  # (nf, nall, 3)
    nall = coord.shape[1]

    # === Step 2. Build valid edge indices once ===
    with nvtx_range("index"):
        src, dst, center_coord_index, neighbor_coord_index = _build_standard_edge_index(
            nlist=nlist,
            mapping=mapping,
            pair_keep_mask=pair_keep_mask,
            nall=nall,
        )

    # === Step 3-5. Edge geometry/RBF chain ===
    #   gather -> edge_vec -> edge_len -> edge_env -> edge_rbf
    coord_flat = coord.reshape(nf * nall, 3)
    # === Step 3. Gather per-edge geometry ===
    # edge_vec points from center -> neighbor: r_ij = r_j - r_i (in Å).
    # edge_len is the scalar distance.
    with nvtx_range("edge_geom"):
        center_pos = coord_flat.index_select(0, center_coord_index)
        neighbor_pos = coord_flat.index_select(0, neighbor_coord_index)
        edge_vec = neighbor_pos - center_pos  # (E, 3)
        edge_len = safe_norm(edge_vec, eps)  # (E, 1)

    # === Step 4. C^3 envelope weight ===
    # Edges with r >= rcut are not removed from the cache. Their envelope is
    # exactly zero, so messages vanish naturally while degree normalization
    # remains smooth at the cutoff boundary.
    with nvtx_range("envelope"):
        edge_env = edge_envelope(edge_len)  # (E, 1)

    # === Step 5. Radial basis (envelope already baked in) ===
    with nvtx_range("radial_basis"):
        edge_rbf = radial_basis(edge_len)  # (E, n_radial)

    # === Step 6. Edge quaternion -> Wigner-D blocks ===
    with nvtx_range("wigner_d"):
        D_full, Dt_full, edge_quat, D_packed = _build_edge_wigner(
            edge_vec=edge_vec,
            edge_len=edge_len,
            eps=eps,
            random_gamma=random_gamma,
            wigner_calc=wigner_calc,
            build_full=build_wigner,
            packed_wigner=packed_wigner,
        )  # (E, D, D), (E, D, D), (E, 4)

    edge_type_feat = build_edge_type_feat(type_ebed, src, dst)  # (E, C)

    return _finalize_edge_cache(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_len=edge_len,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_quat=edge_quat,
        D_packed=D_packed,
        deg_norm_floor=deg_norm_floor,
    )


@torch.amp.autocast("cuda", enabled=False)
def build_edge_cache_from_edges(
    *,
    type_ebed: torch.Tensor,
    atype_flat: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_mask: torch.Tensor,
    compute_dtype: torch.dtype,
    eps: float,
    deg_norm_floor: float,
    read_distance: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
    edge_envelope: Callable[[torch.Tensor], torch.Tensor],
    radial_basis: Callable[[torch.Tensor], torch.Tensor],
    has_exclude_types: bool,
    edge_type_keep_mask: EdgeTypeKeepMaskFn,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    packed_wigner: bool = False,
    fused_radial: FusedRadialFn | None = None,
    fused_wigner: WignerCalculatorFn | None = None,
) -> EdgeFeatureCache:
    """
    Build the global edge cache from a sparse edge list.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C), where N=nf*nloc.
    atype_flat
        Flattened local atom types with shape (N,).
    edge_index
        Edge indices with shape (2, E).
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_mask
        Edge mask with shape (E,). True means keep.
    compute_dtype
        Promoted compute dtype used for geometry and radial features.
    eps
        Small positive epsilon for safe norm.
    deg_norm_floor
        Floor added to the envelope-squared degree before inverse-sqrt
        normalization (see :func:`_finalize_edge_cache`).
    read_distance
        Optional contact-clamp map returning the read lengths and a mask of
        edges at or above their data edge, both with shape (E, 1).
    edge_envelope
        C^3 edge envelope module.
    radial_basis
        Radial basis module.
    fused_radial
        Optional fused replacement of ``edge_envelope`` and ``radial_basis``,
        returning both keep-weighted results from one pass over the distance.
    has_exclude_types
        Whether excluded type pairs should be filtered in this path.
    edge_type_keep_mask
        Callable that builds the keep mask for edge type exclusions.
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.
    fused_wigner
        Optional fused replacement of ``wigner_calc`` that builds the packed
        pair in one kernel pass.

    Returns
    -------
    EdgeFeatureCache
        Per-edge cache.
    """
    n_nodes = type_ebed.shape[0]
    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)

    # === Step 1. Normalize mask and apply type exclusions ===
    edge_keep = edge_mask.to(dtype=torch.bool)
    if has_exclude_types:
        edge_keep = edge_keep & edge_type_keep_mask(atype_flat, src, dst)

    # === Step 2. Promote geometry dtype ===
    edge_vec = edge_vec.to(dtype=compute_dtype)
    edge_keep_f = edge_keep.to(dtype=compute_dtype).unsqueeze(-1)
    edge_vec = edge_vec * edge_keep_f
    # Masked-out edges (zeroed above) are assigned the canonical +z direction so the
    # length normalization and quaternion construction remain finite. Padding the
    # keep-complement into the z channel constructs this term entirely on device.
    edge_vec = edge_vec + F.pad(1.0 - edge_keep_f, (2, 0))

    # === Step 3. Edge length, envelope, and radial basis ===
    with nvtx_range("envelope"):
        if read_distance is None:
            edge_len = safe_norm(edge_vec, eps)
        else:
            edge_vec, edge_len = clamped_edge_geometry(edge_vec, read_distance, eps)
        if fused_radial is not None:
            edge_env, edge_rbf = fused_radial(edge_len, edge_keep_f)
        else:
            edge_env = edge_envelope(edge_len) * edge_keep_f  # (E, 1)
            edge_rbf = radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

    # === Step 4. Edge quaternion -> Wigner-D blocks ===
    with nvtx_range("wigner_d"):
        D_full, Dt_full, edge_quat, D_packed = _build_edge_wigner(
            edge_vec=edge_vec,
            edge_len=edge_len,
            eps=eps,
            random_gamma=random_gamma,
            wigner_calc=fused_wigner if fused_wigner is not None else wigner_calc,
            build_full=build_wigner,
            packed_wigner=packed_wigner,
        )  # (E, D, D), (E, D, D), (E, 4)

    # === Step 5. Edge type features ===
    edge_type_feat = build_edge_type_feat(type_ebed, src, dst)
    edge_type_feat = edge_type_feat * edge_keep_f.to(dtype=edge_type_feat.dtype)

    return _finalize_edge_cache(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_len=edge_len,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_quat=edge_quat,
        D_packed=D_packed,
        deg_norm_floor=deg_norm_floor,
    )


def _build_edge_wigner(
    *,
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor,
    eps: float,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_full: bool = True,
    packed_wigner: bool = False,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Build packed Wigner-D blocks from edge vectors.

    Parameters
    ----------
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_len
        Edge lengths with shape (E, 1).
    eps
        Small positive epsilon used in quaternion construction.
    random_gamma
        Whether to apply a random roll around the local +Z axis.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.
    build_full
        Whether to materialize the full ``(E, D, D)`` Wigner-D blocks. When
        False (all message-passing blocks take the Cartesian path), only the
        quaternion is returned and the blocks are ``None``; the geometric
        initial embedding reconstructs the zonal coupling from the quaternion.
    packed_wigner
        Whether every selected rotation consumer addresses the banded
        layout. The banded storage is only chosen when this holds; any
        dense-layout consumer keeps the dense surfaces.  See
        :func:`deepmd.pt.model.descriptor.sezm.block_supports_packed_wigner`
        for the full exclusion set (the cuTile value path, the fused CUDA
        convolution, the CUDA training value path, the plain SO2 reference
        path, and the aggregation routes that rotate messages back through
        ``Dt_full``; training is dense by construction).

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]
        Packed Wigner-D matrices ``(D_full, Dt_full)`` with shape ``(E, D, D)``
        (or ``None`` when ``build_full`` is False), the quaternion used to
        build them with shape ``(E, 4)``, and the banded storage ``D_packed``
        with shape ``(E, 3 * D - 2)`` when the banded layout was selected
        (``None`` otherwise, including every training build).
    """
    # === Step 1. Build edge-aligned quaternions ===
    edge_quat = build_edge_quaternion(
        edge_vec,
        edge_len=edge_len,
        eps=eps,
    )

    # === Step 2. Apply optional random local-Z roll ===
    if random_gamma:
        gamma = torch.rand(
            edge_quat.shape[0],
            dtype=edge_quat.dtype,
            device=edge_quat.device,
        ) * (2.0 * math.pi)
        edge_quat = quaternion_multiply(quaternion_z_rotation(gamma), edge_quat)

    # === Step 3. Convert quaternions to packed Wigner-D blocks ===
    if not build_full:
        return None, None, edge_quat, None
    # ``wigner_calc`` may arrive as the calculator instance (callable through
    # ``nn.Module.__call__``) or as a bound forward; accept both.
    calc = getattr(wigner_calc, "__self__", wigner_calc)
    if (
        packed_wigner
        and callable(getattr(calc, "forward_band", None))
        and calc.band_supported()
    ):
        # Banded inference storage: the fused reduced-layout consumers (the
        # rotate-mix operator and the flash aggregation) read only the three
        # structural rows per degree block, so the dense surfaces are never
        # materialized. The admission rides the inference level's structural
        # gate (every consumer band-capable, eval, uniform degree), not a
        # per-optimization opt-in.
        return None, None, edge_quat, calc.forward_band_fused(edge_quat)
    if packed_wigner:
        # The admission gate said yes but the resolved calculator cannot
        # produce the band -- typically the ladder degree exceeds the
        # band support.  Dense surfaces are built instead; say so once
        # instead of leaving the band silently unbuilt.
        _warn_packed_inert()
    D_full, Dt_full = wigner_calc(edge_quat)
    return D_full, Dt_full, edge_quat, None


def _finalize_edge_cache(
    *,
    n_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_type_feat: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor,
    edge_rbf: torch.Tensor,
    edge_env: torch.Tensor,
    D_full: torch.Tensor | None,
    Dt_full: torch.Tensor | None,
    edge_quat: torch.Tensor,
    deg_norm_floor: float,
    D_packed: torch.Tensor | None = None,
) -> EdgeFeatureCache:
    """
    Assemble the shared `EdgeFeatureCache` layout.

    Parameters
    ----------
    n_nodes
        Number of local nodes in the flattened frame-major layout.
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).
    edge_type_feat
        Per-edge type features with shape (E, C).
    edge_vec
        Edge vectors with shape (E, 3).
    edge_len
        Regularized read distances with shape (E, 1) in Å.
    edge_rbf
        Radial basis features with shape (E, n_radial).
    edge_env
        Smooth edge envelope weights with shape (E, 1).
    D_full
        Packed Wigner-D matrices with shape (E, D, D), or None when the
        full Wigner-D construction is skipped (all-Cartesian model).
    Dt_full
        Transposed packed Wigner-D matrices with shape (E, D, D), or None
        when the full Wigner-D construction is skipped.
    edge_quat
        Global-to-local quaternions used to build the Wigner-D matrices with
        shape (E, 4).
    deg_norm_floor
        Floor added to the envelope-squared degree before the inverse-sqrt
        normalization. A tiny ``eps`` reproduces the legacy behavior; an
        ``O(1)`` value makes sparse-neighborhood features vanish smoothly at
        ``rcut`` instead of saturating and kinking.

    Returns
    -------
    EdgeFeatureCache
        Finalized per-edge cache shared by eager and compile paths.
    """
    # === Step 1. Build smooth destination degrees ===
    with nvtx_range("degree"):
        deg = torch.zeros(n_nodes, dtype=edge_vec.dtype, device=edge_vec.device)  # (N,)
        deg.index_add_(0, dst, edge_env.squeeze(-1).to(dtype=edge_vec.dtype).square())
        inv_sqrt_deg = rearrange(
            torch.rsqrt(deg + deg_norm_floor), "N -> N 1 1"
        )  # (N, 1, 1)

    return EdgeFeatureCache(
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_len=edge_len,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache={},
        Dt_from_m_cache={},
        csr_cache={},
        edge_quat=edge_quat,
        D_packed=D_packed,
    )


def _build_standard_edge_index(
    *,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    pair_keep_mask: torch.Tensor,
    nall: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flatten DeePMD valid neighbor slots into per-edge indices.

    This helper keeps the original edge semantics used by the eager standard path:

    - padding slots (``nlist == -1``) are removed
    - excluded type pairs are removed
    - no distance-based filtering is applied here; edges beyond ``rcut`` remain
      in the cache and are later zeroed naturally by the smooth envelope

    Parameters
    ----------
    nlist
        DeePMD neighbor list with shape ``(nf, nloc, nnei)``.
    mapping
        Optional extended-to-local mapping with shape ``(nf, nall)``.
    pair_keep_mask
        Pair exclusion keep mask with shape ``(nf, nloc, nnei)``.
    nall
        Number of atoms on the extended axis per frame.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(src, dst, center_coord_index, neighbor_coord_index)`` for the valid
        standard-path edges. All tensors have shape ``(E,)``.
    """
    nf, nloc, nnei = nlist.shape
    nlist_flat = nlist.reshape(-1)

    # === Step 1. Identify valid edge slots ===
    # An edge is valid if:
    #   - it is not padding (nlist >= 0)
    #   - the type pair is allowed (pair_keep_mask)
    # Note: We do NOT filter by distance here. Edges beyond rcut stay in the
    # cache and will later get edge_env=0 from the cutoff envelope.
    valid_nlist = nlist >= 0
    edge_keep = (valid_nlist & pair_keep_mask).reshape(-1)
    edge_slot = torch.nonzero(edge_keep).squeeze(-1).to(dtype=torch.long)

    if edge_slot.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=nlist.device)
        return empty, empty, empty, empty

    # === Step 2. Decode flat edge slots ===
    # edge_slot indexes the flattened (nf, nloc, nnei) axis in row-major order.
    # Convert it back to:
    #   frame_idx   in [0, nf)
    #   center_local in [0, nloc)
    #   neighbor_ext from the extended axis in [0, nall)
    frame_idx = edge_slot // (nloc * nnei)
    rem = edge_slot % (nloc * nnei)
    center_local = rem // nnei
    neighbor_ext = nlist_flat.index_select(0, edge_slot)

    if mapping is None:
        # Neighbor indices are already local indices in [0, nloc).
        src_local = neighbor_ext
    else:
        # Map extended index -> local index for each frame.
        # mapping_flat packs (nf, nall), so frame k uses offset k * nall.
        mapping_flat = mapping.reshape(-1)
        src_local = mapping_flat.index_select(0, frame_idx * nall + neighbor_ext)

    src_ok = (src_local >= 0) & (src_local < nloc)
    if not bool(src_ok.all()):
        # Drop edges that map outside the local range, e.g. broken mapping
        # or ghost-only neighbors.
        frame_idx = frame_idx[src_ok]
        center_local = center_local[src_ok]
        neighbor_ext = neighbor_ext[src_ok]
        src_local = src_local[src_ok]

    if src_local.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=nlist.device)
        return empty, empty, empty, empty

    # === Step 3. Build node and coordinate indices ===
    # dst is the center atom: per-frame local index -> global node index.
    # src is the neighbor atom: per-frame local index -> global node index.
    # The coordinate indices still point to the extended coordinate tensor.
    src = frame_idx * nloc + src_local
    dst = frame_idx * nloc + center_local
    center_coord_index = frame_idx * nall + center_local
    neighbor_coord_index = frame_idx * nall + neighbor_ext
    return src, dst, center_coord_index, neighbor_coord_index


def build_edge_type_feat(
    type_ebed: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    """
    Build per-edge type features by summing src/dst embeddings.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C).
    src
        Source node indices with shape (E,).
    dst
        Destination node indices with shape (E,).

    Returns
    -------
    torch.Tensor
        Per-edge type features with shape (E, C).
    """
    # === Step 1. Normalize index dtypes ===
    if src.dtype != torch.long:
        src = src.to(dtype=torch.long)
    if dst.dtype != torch.long:
        dst = dst.to(dtype=torch.long)

    # === Step 2. Sum source and destination embeddings ===
    return type_ebed.index_select(0, src) + type_ebed.index_select(0, dst)


def edge_cache_to_dtype(
    cache: EdgeFeatureCache, dtype: torch.dtype
) -> EdgeFeatureCache:
    """
    Convert all floating-point tensors in EdgeFeatureCache to the specified dtype.

    Integer tensors (src, dst) are unchanged. This is a standalone function
    (not a method) to keep it side-effect free.

    Parameters
    ----------
    cache
        The edge feature cache to convert.
    dtype
        Target dtype for floating-point tensors.

    Returns
    -------
    EdgeFeatureCache
        New cache with converted tensors.
    """
    # Handle Optional tensors explicitly.
    # Use local variables with explicit None check and assignment.
    _D_full = cache.D_full
    _Dt_full = cache.Dt_full
    _edge_quat = cache.edge_quat
    _D_packed = cache.D_packed
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    edge_quat: torch.Tensor | None = None
    D_packed: torch.Tensor | None = None
    if _D_full is not None:
        D_full = _D_full.to(dtype=dtype)
    if _Dt_full is not None:
        Dt_full = _Dt_full.to(dtype=dtype)
    if _edge_quat is not None:
        edge_quat = _edge_quat.to(dtype=dtype)
    if _D_packed is not None:
        D_packed = _D_packed.to(dtype=dtype)

    # CSR views contain only integer topology. Preserve them across the dtype
    # conversion so every accelerated consumer shares the per-step sort.
    return EdgeFeatureCache(
        src=cache.src,
        dst=cache.dst,
        edge_type_feat=cache.edge_type_feat.to(dtype=dtype),
        edge_vec=cache.edge_vec.to(dtype=dtype),
        edge_len=cache.edge_len.to(dtype=dtype),
        edge_rbf=cache.edge_rbf.to(dtype=dtype),
        edge_env=cache.edge_env.to(dtype=dtype),
        deg=cache.deg.to(dtype=dtype),
        inv_sqrt_deg=cache.inv_sqrt_deg.to(dtype=dtype),
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache=None if cache.D_to_m_cache is None else {},
        Dt_from_m_cache=None if cache.Dt_from_m_cache is None else {},
        csr_cache=None if cache.csr_cache is None else dict(cache.csr_cache),
        edge_quat=edge_quat,
        D_packed=D_packed,
    )
