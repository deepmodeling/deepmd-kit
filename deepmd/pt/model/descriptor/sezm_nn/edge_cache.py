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

import math
from collections.abc import (
    Callable,
)
from typing import (
    NamedTuple,
)

import torch
from einops import (
    rearrange,
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
    D_to_m_cache
        Lazy cache for projected D matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by a normalized
        ``"lmax:mmax"`` identifier.
    edge_src_gate
        Optional per-edge Source Freeze Propagation Gate (SFPG) weight with
        shape (E, 1). Equals ``eta[src]`` where
        ``eta[j] = prod_{k in N(j)} w(r_{jk})`` and ``w`` is the
        :class:`BridgingSwitch` C3 switching amplitude. Present only when
        the model runs in bridging mode; ``None`` otherwise. Aggregation
        sites (``GeometricInitialEmbedding``, ``EnvironmentInitialEmbedding``,
        ``SO2Convolution``) multiply their per-edge message contribution
        by this gate to forbid any node whose local neighborhood enters
        the frozen zone from propagating information along its outgoing
        edges.
    """

    src: torch.Tensor
    dst: torch.Tensor
    edge_type_feat: torch.Tensor
    edge_vec: torch.Tensor
    edge_rbf: torch.Tensor
    edge_env: torch.Tensor
    deg: torch.Tensor
    inv_sqrt_deg: torch.Tensor
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    D_to_m_cache: dict[str, torch.Tensor] | None = None
    Dt_from_m_cache: dict[str, torch.Tensor] | None = None
    edge_src_gate: torch.Tensor | None = None
    edge_quat: torch.Tensor | None = None


def compute_edge_src_gate(
    *,
    edge_len: torch.Tensor,
    src: torch.Tensor,
    n_nodes: int,
    bridging_switch: Callable[[torch.Tensor], torch.Tensor],
    edge_keep_f: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the per-edge source gate for SFPG from edge lengths.

    The gate implements a per-node "non-frozen confidence" and broadcasts
    it back to edges along the source axis::

        w_e      = bridging_switch(edge_len_e)               in [0, 1]
        eta_j    = prod_{e: src_e = j} w_e                   in [0, 1]
        gate_e   = eta_{src_e}                               in [0, 1]

    ``w_e = 0`` at ``r_{jk} <= r_inner`` ensures ``eta_j = 0`` for any
    node with at least one neighbor in the frozen zone. Masked edges
    (padding, excluded type pairs) must contribute the multiplicative
    identity ``1`` so they never spuriously mute a valid source node;
    callers supply ``edge_keep_f`` for this.

    The product is **not** realised by ``scatter_reduce(reduce="prod")``:
    its registered backward handles exact zeros with a data-dependent
    "count leave-one-out" branch that creates unbacked symints under
    ``make_fx(tracing_mode="symbolic")`` and breaks the SeZM compile
    path's double-backward tracing. Instead, the product is decomposed
    into a log-sum on non-zero contributions combined with an explicit
    "any zero per group" indicator that routes the frozen case through
    ``torch.where``. Both branches use only shape-preserving standard
    ops (``scatter_add``, ``where``, ``exp``, ``log``) with backed
    symints, so the graph survives symbolic tracing cleanly.

    The gradient consequence at the plateau is exact: ``BridgingSwitch``
    places ``w'(r) = 0`` for every ``r <= r_inner``, so the chain rule
    ``d eta / d r = (leave-one-out factor) * w'(r) = anything * 0 = 0``
    holds regardless of how the muted ``torch.where`` branch treats the
    upstream gradient. In the transition zone every edge has strictly
    positive ``w`` and the log-sum branch gives the standard product
    gradient.

    Parameters
    ----------
    edge_len
        Per-edge distances with shape (E, 1).
    src
        Source node indices with shape (E,).
    n_nodes
        Total number of nodes N.
    bridging_switch
        Callable ``r -> w(r)`` with ``w: [0, ∞) -> [0, 1]``, typically a
        :class:`BridgingSwitch` instance.
    edge_keep_f
        Optional per-edge keep weights with shape (E, 1), with ``0`` on
        masked edges and ``1`` on kept edges. If provided, masked edges
        are rewritten to ``w = 1`` before the product reduction.

    Returns
    -------
    torch.Tensor
        Per-edge source gate with shape (E, 1), aligned on the same edge
        axis as the rest of the cache.
    """
    # === Step 1. Per-edge switching amplitude w(r) in [0, 1] ===
    edge_w = bridging_switch(edge_len)  # (E, 1)
    if edge_keep_f is not None:
        # Force w = 1 on masked edges so they are neutral for the product.
        edge_w = edge_w * edge_keep_f + (1.0 - edge_keep_f)

    edge_w_flat = edge_w.squeeze(-1)  # (E,)
    is_zero = edge_w_flat <= 0.0  # (E,) bool

    # === Step 2. Log-sum reduction on non-zero contributions ===
    # Replace exact zeros with the multiplicative identity 1 so their
    # ``log`` contribution is 0 and the group-wise sum equals the log of
    # the product of non-zero ``w`` values.
    safe_w = torch.where(is_zero, torch.ones_like(edge_w_flat), edge_w_flat)
    log_safe = torch.log(safe_w)
    log_eta = torch.zeros(
        n_nodes, dtype=edge_w.dtype, device=edge_w.device
    ).scatter_add(0, src, log_safe)
    eta_nonzero_path = torch.exp(log_eta)

    # === Step 3. Exact-zero indicator per source node ===
    # ``scatter_add`` over an ``int64`` cast of the zero mask counts how
    # many frozen edges each source node owns. A strictly positive count
    # means the product is 0 by the hard-freeze rule.
    zero_count = torch.zeros(
        n_nodes, dtype=torch.int64, device=edge_w.device
    ).scatter_add(0, src, is_zero.to(torch.int64))
    any_zero = zero_count > 0

    # === Step 4. Combine and broadcast back to edges via source ===
    eta = torch.where(any_zero, torch.zeros_like(eta_nonzero_path), eta_nonzero_path)
    return eta.index_select(0, src).unsqueeze(-1)


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
    n_radial: int,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
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
    n_radial
        Number of radial basis channels used for empty-cache allocation.
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

    if src.numel() == 0:
        return _get_empty_edge_cache(
            n_nodes=n_nodes,
            n_radial=n_radial,
            n_channel=type_ebed.shape[1],
            device=extended_coord.device,
            dtype=extended_coord.dtype,
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
        D_full, Dt_full, edge_quat = _build_edge_wigner(
            edge_vec=edge_vec,
            edge_len=edge_len,
            eps=eps,
            random_gamma=random_gamma,
            wigner_calc=wigner_calc,
        )  # (E, D, D), (E, D, D), (E, 4)

    edge_type_feat = build_edge_type_feat(type_ebed, src, dst)  # (E, C)

    return _finalize_edge_cache(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_quat=edge_quat,
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
    inner_clamp: Callable[[torch.Tensor], torch.Tensor] | None,
    bridging_switch: Callable[[torch.Tensor], torch.Tensor] | None,
    edge_envelope: Callable[[torch.Tensor], torch.Tensor],
    radial_basis: Callable[[torch.Tensor], torch.Tensor],
    has_exclude_types: bool,
    edge_type_keep_mask: EdgeTypeKeepMaskFn,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
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
    inner_clamp
        Optional inner clamp used to freeze short-range geometry below `r_inner`.
    bridging_switch
        Optional C3 switching amplitude ``w(r) -> [0, 1]`` that drives
        the Source Freeze Propagation Gate. When provided, a per-edge
        ``edge_src_gate`` is computed from the node-wise product of
        ``w(r_{jk})`` along each source node's outgoing edges. Masked
        edges (``edge_keep=False``) are forced to ``w=1`` so they never
        leak into the product.
    edge_envelope
        C^3 edge envelope module.
    radial_basis
        Radial basis module.
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
    edge_vec = edge_vec + (1.0 - edge_keep_f) * edge_vec.new_tensor([0.0, 0.0, 1.0])

    # === Step 3. Edge length, envelope, and radial basis ===
    with nvtx_range("envelope"):
        edge_len = safe_norm(edge_vec, eps)
        if inner_clamp is not None:
            clamped = inner_clamp(edge_len)
            scale = clamped / edge_len
            edge_vec = edge_vec * scale
            edge_len = clamped
        edge_env = edge_envelope(edge_len) * edge_keep_f  # (E, 1)
        edge_rbf = radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

    # === Step 4. Edge quaternion -> Wigner-D blocks ===
    with nvtx_range("wigner_d"):
        D_full, Dt_full, edge_quat = _build_edge_wigner(
            edge_vec=edge_vec,
            edge_len=edge_len,
            eps=eps,
            random_gamma=random_gamma,
            wigner_calc=wigner_calc,
        )  # (E, D, D), (E, D, D), (E, 4)

    # === Step 5. Edge type features ===
    edge_type_feat = build_edge_type_feat(type_ebed, src, dst)
    edge_type_feat = edge_type_feat * edge_keep_f.to(dtype=edge_type_feat.dtype)

    # === Step 6. Source Freeze Propagation Gate (optional) ===
    # The sparse-edge path packs masked dummy edges so the compiled graph sees
    # a statically non-empty, non-singular edge tensor. ``edge_keep_f`` rewrites
    # any such slot to ``w=1`` inside ``compute_edge_src_gate``, keeping the
    # product reduction unaffected by padding.
    edge_src_gate: torch.Tensor | None = None
    if bridging_switch is not None:
        with nvtx_range("src_gate"):
            edge_src_gate = compute_edge_src_gate(
                edge_len=edge_len,
                src=src,
                n_nodes=n_nodes,
                bridging_switch=bridging_switch,
                edge_keep_f=edge_keep_f,
            )

    return _finalize_edge_cache(
        n_nodes=n_nodes,
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_quat=edge_quat,
        deg_norm_floor=deg_norm_floor,
        edge_src_gate=edge_src_gate,
    )


def _build_edge_wigner(
    *,
    edge_vec: torch.Tensor,
    edge_len: torch.Tensor,
    eps: float,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Packed Wigner-D matrices ``(D_full, Dt_full)`` with shape ``(E, D, D)``
        and the quaternion used to build them with shape ``(E, 4)``.
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
    D_full, Dt_full = wigner_calc(edge_quat)
    return D_full, Dt_full, edge_quat


def _finalize_edge_cache(
    *,
    n_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_type_feat: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_rbf: torch.Tensor,
    edge_env: torch.Tensor,
    D_full: torch.Tensor,
    Dt_full: torch.Tensor,
    edge_quat: torch.Tensor,
    deg_norm_floor: float,
    edge_src_gate: torch.Tensor | None = None,
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
    edge_rbf
        Radial basis features with shape (E, n_radial).
    edge_env
        Smooth edge envelope weights with shape (E, 1).
    D_full
        Packed Wigner-D matrices with shape (E, D, D).
    Dt_full
        Transposed packed Wigner-D matrices with shape (E, D, D).
    edge_quat
        Global-to-local quaternions used to build the Wigner-D matrices with
        shape (E, 4).
    deg_norm_floor
        Floor added to the envelope-squared degree before the inverse-sqrt
        normalization. A tiny ``eps`` reproduces the legacy behavior; an
        ``O(1)`` value makes sparse-neighborhood features vanish smoothly at
        ``rcut`` instead of saturating and kinking.
    edge_src_gate
        Optional per-edge SFPG weight with shape (E, 1). ``None`` in
        non-bridging mode.

    Returns
    -------
    EdgeFeatureCache
        Finalized per-edge cache shared by eager and compile paths.
    """
    # === Step 1. Build smooth destination degrees ===
    with nvtx_range("degree"):
        deg = torch.zeros(n_nodes, dtype=edge_vec.dtype, device=edge_vec.device)  # (N,)
        deg.index_add_(0, dst, edge_env.squeeze(-1).to(dtype=edge_vec.dtype).square())
        floor_tensor = deg.new_tensor(deg_norm_floor)
        inv_sqrt_deg = rearrange(
            torch.rsqrt(deg + floor_tensor), "N -> N 1 1"
        )  # (N, 1, 1)

    return EdgeFeatureCache(
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache={},
        Dt_from_m_cache={},
        edge_src_gate=edge_src_gate,
        edge_quat=edge_quat,
    )


def _get_empty_edge_cache(
    *,
    n_nodes: int,
    n_radial: int,
    n_channel: int,
    device: torch.device,
    dtype: torch.dtype,
) -> EdgeFeatureCache:
    """
    Allocate an empty edge cache for one SeZM forward pass.

    Parameters
    ----------
    n_nodes
        Number of local nodes in the flattened frame-major layout.
    n_radial
        Number of radial basis channels.
    n_channel
        Edge type feature width.
    device
        Target device for the cache tensors.
    dtype
        Target floating-point dtype for the cache tensors.

    Returns
    -------
    EdgeFeatureCache
        Empty cache with valid tensor shapes and neutral degree normalization.
    """
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    empty_vec = torch.empty(0, 3, dtype=dtype, device=device)
    empty_quat = torch.empty(0, 4, dtype=dtype, device=device)
    empty_rbf = torch.empty(0, n_radial, dtype=dtype, device=device)
    empty_type_feat = torch.empty(0, n_channel, dtype=dtype, device=device)
    deg = torch.zeros(n_nodes, dtype=dtype, device=device)
    inv_sqrt_deg = torch.ones(n_nodes, 1, 1, dtype=dtype, device=device)
    return EdgeFeatureCache(
        src=empty_long,
        dst=empty_long,
        edge_type_feat=empty_type_feat,
        edge_vec=empty_vec,
        edge_rbf=empty_rbf,
        edge_env=torch.empty(0, 1, dtype=dtype, device=device),
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=None,
        Dt_full=None,
        D_to_m_cache={},
        Dt_from_m_cache={},
        edge_src_gate=None,
        edge_quat=empty_quat,
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
    _edge_src_gate = cache.edge_src_gate
    _edge_quat = cache.edge_quat
    D_full: torch.Tensor | None = None
    Dt_full: torch.Tensor | None = None
    edge_src_gate: torch.Tensor | None = None
    edge_quat: torch.Tensor | None = None
    if _D_full is not None:
        D_full = _D_full.to(dtype=dtype)
    if _Dt_full is not None:
        Dt_full = _Dt_full.to(dtype=dtype)
    if _edge_src_gate is not None:
        edge_src_gate = _edge_src_gate.to(dtype=dtype)
    if _edge_quat is not None:
        edge_quat = _edge_quat.to(dtype=dtype)

    return EdgeFeatureCache(
        src=cache.src,
        dst=cache.dst,
        edge_type_feat=cache.edge_type_feat.to(dtype=dtype),
        edge_vec=cache.edge_vec.to(dtype=dtype),
        edge_rbf=cache.edge_rbf.to(dtype=dtype),
        edge_env=cache.edge_env.to(dtype=dtype),
        deg=cache.deg.to(dtype=dtype),
        inv_sqrt_deg=cache.inv_sqrt_deg.to(dtype=dtype),
        D_full=D_full,
        Dt_full=Dt_full,
        D_to_m_cache=None if cache.D_to_m_cache is None else {},
        Dt_from_m_cache=None if cache.Dt_from_m_cache is None else {},
        edge_src_gate=edge_src_gate,
        edge_quat=edge_quat,
    )
