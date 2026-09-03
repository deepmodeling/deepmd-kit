# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Edge cache construction utilities for DPA4/SeZM.

This module defines the shared procedures that assemble per-edge geometry,
radial features, rotation blocks, and normalization terms used by the SeZM
descriptor.

This module is the dpmodel (array-API) port of
``deepmd.pt.model.descriptor.sezm_nn.edge_cache``.
"""

from __future__ import (
    annotations,
)

import math
from collections.abc import (
    Callable,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    xp_add_at,
    xp_asarray_nodetach,
    xp_uniform,
)

from .utils import (
    safe_norm,
)
from .wignerd import (
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)

WignerCalculatorFn = Callable[[Any], "tuple[Any, Any]"]
PackedWignerFn = Callable[[Any, WignerCalculatorFn], Any | None]
# Distance and keep weight to the keep-weighted envelope and radial basis, the
# fused replacement of applying the two modules separately.
FusedRadialFn = Callable[[Any, Any], "tuple[Any, Any]"]


@dataclass
class EdgeCache:
    """
    Global edge feature cache created once per forward().

    All per-edge arrays are aligned on the same edge axis (E).

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
    csr_cache
        Lazy cache for endpoint CSR views used by segmented accelerated
        operators, keyed by endpoint role (``"dst"`` or ``"src"``). Built once
        per step and shared by every consumer. Permutations and row pointers
        use int64 regardless of the input graph index dtype.
    cute_infer_so2_metadata
        Optional ``(destination_row_ptr, source_order, source_row_ptr)`` tuple
        prepared once for a backend CuTe SO2 implementation.
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
    edge_mask
        Validity mask for the padded standard-path layout with shape (E,) or
        (E, 1); 1 marks a real edge, 0 a padded/invalid slot. ``None`` means
        all slots are valid (e.g. the sparse
        :func:`_edge_cache_from_arrays` path, where masking is folded into
        the per-edge weights). This field has no pt counterpart.
    destinations_sorted
        Whether the edge payload is destination-major with masked entries
        confined to the suffix.
    D_packed
        Optional backend-specific packed Wigner panel. Dense consumers observe
        ``D_full`` and ``Dt_full`` as ``None`` when this storage is active.
    """

    src: Any
    dst: Any
    edge_type_feat: Any
    edge_vec: Any
    edge_rbf: Any
    edge_env: Any
    deg: Any
    inv_sqrt_deg: Any
    D_full: Any = None
    Dt_full: Any = None
    D_to_m_cache: dict[str, Any] = field(default_factory=dict)
    Dt_from_m_cache: dict[str, Any] = field(default_factory=dict)
    csr_cache: dict[str, tuple[Any, Any]] | None = field(default_factory=dict)
    edge_src_gate: Any = None
    edge_quat: Any = None
    edge_mask: Any = None
    cute_infer_so2_metadata: tuple[Any, Any, Any] | None = None
    destinations_sorted: bool = False
    D_packed: Any = None


def compute_edge_src_gate(
    *,
    edge_len: Any,
    src: Any,
    n_nodes: int,
    bridging_switch: Callable[[Any], Any],
    edge_keep_f: Any = None,
    node_partial_exchange: Callable[[Any], Any] | None = None,
) -> Any:
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
    ``where``. Both branches use only shape-preserving standard
    ops (``scatter_add``, ``where``, ``exp``, ``log``) with backed
    symints, so the graph survives symbolic tracing cleanly.

    The gradient consequence at the plateau is exact: ``BridgingSwitch``
    places ``w'(r) = 0`` for every ``r <= r_inner``, so the chain rule
    ``d eta / d r = (leave-one-out factor) * w'(r) = anything * 0 = 0``
    holds regardless of how the muted ``where`` branch treats the
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
    node_partial_exchange
        Optional cross-rank completion hook for the per-node partials
        (issue #5906). Receives the ``(n_nodes, 2)`` float tensor
        ``[log_eta, zero_count]`` and returns the globally completed one
        (reverse-accumulate ghost rows into owners, then broadcast the
        completed owner values back onto ghosts). ``None`` (the default)
        is the single-process path where the local partials are already
        complete.

    Returns
    -------
    Array
        Per-edge source gate with shape (E, 1), aligned on the same edge
        axis as the rest of the cache.
    """
    xp = array_api_compat.array_namespace(edge_len, src)
    device = array_api_compat.device(edge_len)
    # === Step 1. Per-edge switching amplitude w(r) in [0, 1] ===
    edge_w = bridging_switch(edge_len)  # (E, 1)
    if edge_keep_f is not None:
        # Force w = 1 on masked edges so they are neutral for the product.
        edge_w = edge_w * edge_keep_f + (1.0 - edge_keep_f)

    edge_w_flat = edge_w[..., 0]  # (E,)
    is_zero = edge_w_flat <= 0.0  # (E,) bool

    # === Step 2. Log-sum reduction on non-zero contributions ===
    # Replace exact zeros with the multiplicative identity 1 so their
    # ``log`` contribution is 0 and the group-wise sum equals the log of
    # the product of non-zero ``w`` values.
    safe_w = xp.where(is_zero, xp.ones_like(edge_w_flat), edge_w_flat)
    log_safe = xp.log(safe_w)
    log_eta = xp_add_at(
        xp.zeros((n_nodes,), dtype=edge_w.dtype, device=device), src, log_safe
    )

    # === Step 3. Exact-zero indicator per source node ===
    # ``scatter_add`` over the zero mask counts how many frozen edges each
    # source node owns. A strictly positive count means the product is 0 by
    # the hard-freeze rule. Float count (values are small integers, exact
    # in fp) so both partials ride ONE border-exchange tensor when
    # completing across ranks.
    zero_count = xp_add_at(
        xp.zeros((n_nodes,), dtype=edge_w.dtype, device=device),
        src,
        xp.astype(is_zero, edge_w.dtype),
    )

    # === Step 3b. Cross-rank completion of the per-node partials ===
    # A rank only holds edges whose dst is owned, so the src-keyed sums
    # above are PARTIAL for every node under domain decomposition. The hook
    # (reverse-accumulate ghost->owner, then forward-broadcast owner->ghost)
    # completes them; log-products are additive and each edge lives on
    # exactly one rank, so nothing double-counts (issue #5906).
    if node_partial_exchange is not None:
        packed = xp.stack([log_eta, zero_count], axis=-1)  # (n_nodes, 2)
        packed = node_partial_exchange(packed)
        log_eta = packed[..., 0]
        zero_count = packed[..., 1]

    eta_nonzero_path = xp.exp(log_eta)
    any_zero = zero_count > 0.5

    # === Step 4. Combine and broadcast back to edges via source ===
    eta = xp.where(any_zero, xp.zeros_like(eta_nonzero_path), eta_nonzero_path)
    return xp.take(eta, src, axis=0)[:, None]


def _edge_cache_from_arrays(
    *,
    type_ebed: Any,
    edge_index: Any,
    edge_vec: Any,
    edge_mask: Any,
    compute_dtype: Any,
    eps: float,
    deg_norm_floor: float,
    inner_clamp: Callable[[Any], Any] | None,
    bridging_switch: Callable[[Any], Any] | None,
    edge_envelope: Callable[[Any], Any],
    radial_basis: Callable[[Any], Any],
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    gamma: Any = None,
    node_partial_exchange: Callable[[Any], Any] | None = None,
    fused_radial: FusedRadialFn | None = None,
    fused_wigner: WignerCalculatorFn | None = None,
    packed_wigner: bool = False,
    destinations_sorted: bool = False,
    packed_wigner_fn: PackedWignerFn | None = None,
    csr_cache: dict[str, tuple[Any, Any]] | None = None,
) -> EdgeCache:
    """
    Build the global edge cache from a sparse edge list.

    Private core, invoked only from ``DescrptDPA4._run_graph``. The descriptor's
    own ``exclude_types`` masking is not applied here: ``_run_graph`` applies it
    exactly once, upstream, on the ``NeighborGraph``'s ``edge_mask`` via
    ``apply_pair_exclusion``. (Model-level ``pair_exclude_types`` is a separate,
    graph-BUILD transform, already folded into the incoming graph.)

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C), where N=nf*nloc.
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
    fused_radial
        Optional fused replacement of ``edge_envelope`` and ``radial_basis``,
        returning both keep-weighted results from one pass over the distance.
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.
    fused_wigner
        Optional fused replacement of ``wigner_calc`` that builds the packed
        pair in one kernel pass.
    packed_wigner
        Whether the backend-specific packed Wigner contract is satisfied.
    destinations_sorted
        Whether the edge payload is destination-major with masked entries in
        a suffix.
    packed_wigner_fn
        Optional backend implementation that builds a packed Wigner panel.
    csr_cache
        Optional endpoint CSR views inherited from the input graph. The cache
        owns the same edge axis as ``edge_index``. Permutations and row pointers
        are normalized to int64 at this graph-to-cache boundary.
    gamma
        Optional per-edge roll angles with shape (E,), used only when
        ``random_gamma`` is True. When None, drawn with the backend's RNG
        (:func:`~deepmd.dpmodel.array_api.xp_uniform`) uniformly in
        ``[0, 2*pi)``; callers may inject angles to pin a draw.
    node_partial_exchange
        Optional cross-rank completion hook forwarded to
        :func:`compute_edge_src_gate` (issue #5906); only meaningful when
        ``bridging_switch`` is provided.

    Returns
    -------
    EdgeCache
        Per-edge cache.
    """
    xp = array_api_compat.array_namespace(type_ebed, edge_index, edge_vec)
    device = array_api_compat.device(edge_vec)
    n_nodes = type_ebed.shape[0]
    src = xp.astype(edge_index[0, ...], xp.int64)
    dst = xp.astype(edge_index[1, ...], xp.int64)
    if csr_cache is not None:
        csr_cache = {
            endpoint: (
                order if order.dtype == xp.int64 else xp.astype(order, xp.int64),
                (
                    row_ptr
                    if row_ptr.dtype == xp.int64
                    else xp.astype(row_ptr, xp.int64)
                ),
            )
            for endpoint, (order, row_ptr) in csr_cache.items()
        }

    # === Step 1. Normalize mask ===
    edge_keep = xp.astype(edge_mask, xp.bool)

    # === Step 2. Promote geometry dtype ===
    edge_vec = xp.astype(edge_vec, compute_dtype)
    edge_keep_f = xp.astype(edge_keep, compute_dtype)[:, None]
    edge_vec = edge_vec * edge_keep_f
    # Masked-out edges (zeroed above) are assigned the canonical +z direction so the
    # length normalization and quaternion construction remain finite. Padding the
    # keep-complement into the z channel constructs this term entirely on device.
    zeros2 = xp.zeros((edge_keep_f.shape[0], 2), dtype=edge_vec.dtype, device=device)
    edge_vec = edge_vec + xp.concat([zeros2, 1.0 - edge_keep_f], axis=-1)

    # === Step 3. Edge length, envelope, and radial basis ===
    edge_len = safe_norm(edge_vec, eps)
    if inner_clamp is not None:
        clamped = inner_clamp(edge_len)
        scale = clamped / edge_len
        edge_vec = edge_vec * scale
        edge_len = clamped
    if fused_radial is not None:
        edge_env, edge_rbf = fused_radial(edge_len, edge_keep_f)
    else:
        edge_env = edge_envelope(edge_len) * edge_keep_f  # (E, 1)
        edge_rbf = radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

    # === Step 4. Edge quaternion -> Wigner-D blocks ===
    D_full, Dt_full, D_packed, edge_quat = _build_edge_wigner(
        edge_vec=edge_vec,
        edge_len=edge_len,
        eps=eps,
        random_gamma=random_gamma,
        wigner_calc=fused_wigner if fused_wigner is not None else wigner_calc,
        gamma=gamma,
        build_full=build_wigner,
        packed_wigner=packed_wigner,
        packed_wigner_fn=packed_wigner_fn,
    )

    # === Step 5. Edge type features ===
    edge_type_feat = build_edge_type_feat(type_ebed, src, dst)
    edge_type_feat = edge_type_feat * xp.astype(edge_keep_f, edge_type_feat.dtype)

    # === Step 6. Source Freeze Propagation Gate (optional) ===
    # The sparse-edge path packs masked dummy edges so the compiled graph sees
    # a statically non-empty, non-singular edge tensor. ``edge_keep_f`` rewrites
    # any such slot to ``w=1`` inside ``compute_edge_src_gate``, keeping the
    # product reduction unaffected by padding.
    edge_src_gate: Any = None
    if bridging_switch is not None:
        edge_src_gate = compute_edge_src_gate(
            edge_len=edge_len,
            src=src,
            n_nodes=n_nodes,
            bridging_switch=bridging_switch,
            edge_keep_f=edge_keep_f,
            node_partial_exchange=node_partial_exchange,
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
        D_packed=D_packed,
        edge_quat=edge_quat,
        deg_norm_floor=deg_norm_floor,
        edge_src_gate=edge_src_gate,
        destinations_sorted=destinations_sorted,
        csr_cache=csr_cache,
    )


def _build_edge_wigner(
    *,
    edge_vec: Any,
    edge_len: Any,
    eps: float,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    gamma: Any = None,
    build_full: bool = True,
    packed_wigner: bool = False,
    packed_wigner_fn: PackedWignerFn | None = None,
) -> tuple[Any, Any, Any, Any]:
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
    gamma
        Optional per-edge roll angles with shape (E,), used only when
        ``random_gamma`` is True. When None, drawn with the backend's RNG
        (:func:`~deepmd.dpmodel.array_api.xp_uniform`) uniformly in
        ``[0, 2*pi)``.
    build_full
        Whether to materialize the full ``(E, D, D)`` Wigner-D blocks. When
        False (all message-passing blocks take the Cartesian path), only the
        quaternion is returned and the blocks are ``None``; the geometric
        initial embedding reconstructs the zonal coupling from the quaternion.
    packed_wigner
        Whether the backend-specific packed Wigner contract is satisfied.
    packed_wigner_fn
        Optional backend implementation that builds a packed Wigner panel.

    Returns
    -------
    tuple[Array, Array, Array, Array]
        Dense Wigner-D matrices ``(D_full, Dt_full)``, an optional packed panel,
        and the quaternion used to build them.
    """
    xp = array_api_compat.array_namespace(edge_vec)
    device = array_api_compat.device(edge_vec)
    # === Step 1. Build edge-aligned quaternions ===
    edge_quat = build_edge_quaternion(
        edge_vec,
        edge_len=edge_len,
        eps=eps,
    )

    # === Step 2. Apply optional random local-Z roll ===
    # Training-only augmentation: drawn with the backend's own RNG so torch
    # replays it under setup_seed and keeps the draw on-device.
    if random_gamma:
        if gamma is None:
            gamma = xp_uniform(edge_quat, edge_quat.shape[0], 0.0, 2.0 * math.pi)
        gamma = xp.astype(
            xp_asarray_nodetach(xp, gamma, device=device), edge_quat.dtype
        )
        edge_quat = quaternion_multiply(quaternion_z_rotation(gamma), edge_quat)

    # === Step 3. Convert quaternions to packed Wigner-D blocks ===
    if not build_full:
        return None, None, None, edge_quat
    if packed_wigner and packed_wigner_fn is not None:
        D_packed = packed_wigner_fn(edge_quat, wigner_calc)
        if D_packed is not None:
            return None, None, D_packed, edge_quat
    D_full, Dt_full = wigner_calc(edge_quat)
    return D_full, Dt_full, None, edge_quat


def _finalize_edge_cache(
    *,
    n_nodes: int,
    src: Any,
    dst: Any,
    edge_type_feat: Any,
    edge_vec: Any,
    edge_rbf: Any,
    edge_env: Any,
    D_full: Any,
    Dt_full: Any,
    D_packed: Any,
    edge_quat: Any,
    deg_norm_floor: float,
    edge_src_gate: Any = None,
    destinations_sorted: bool = False,
    csr_cache: dict[str, tuple[Any, Any]] | None = None,
) -> EdgeCache:
    """
    Assemble the shared `EdgeCache` layout.

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
        Packed Wigner-D matrices with shape (E, D, D), or None when the
        full Wigner-D construction is skipped (all-Cartesian model).
    Dt_full
        Transposed packed Wigner-D matrices with shape (E, D, D), or None
        when the full Wigner-D construction is skipped.
    D_packed
        Optional backend-specific packed Wigner panel.
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
    destinations_sorted
        Whether ``dst`` is nondecreasing with masked entries in a suffix.
    csr_cache
        Optional endpoint CSR views aligned with the cache edge axis.

    Returns
    -------
    EdgeCache
        Finalized per-edge cache shared by eager and compile paths.
    """
    xp = array_api_compat.array_namespace(edge_vec, dst)
    device = array_api_compat.device(edge_vec)
    # === Step 1. Build smooth destination degrees ===
    deg = xp.zeros((n_nodes,), dtype=edge_vec.dtype, device=device)  # (N,)
    env_flat = xp.astype(edge_env[..., 0], edge_vec.dtype)
    deg = xp_add_at(deg, dst, env_flat * env_flat)
    inv_sqrt_deg = xp.reshape(
        1.0 / xp.sqrt(deg + deg_norm_floor), (n_nodes, 1, 1)
    )  # (N, 1, 1)

    return EdgeCache(
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
        D_packed=D_packed,
        D_to_m_cache={},
        Dt_from_m_cache={},
        csr_cache={} if csr_cache is None else csr_cache,
        edge_src_gate=edge_src_gate,
        edge_quat=edge_quat,
        destinations_sorted=destinations_sorted,
    )


def build_edge_type_feat(
    type_ebed: Any,
    src: Any,
    dst: Any,
) -> Any:
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
    Array
        Per-edge type features with shape (E, C).
    """
    xp = array_api_compat.array_namespace(type_ebed, src, dst)
    # === Step 1. Normalize index dtypes ===
    if src.dtype != xp.int64:
        src = xp.astype(src, xp.int64)
    if dst.dtype != xp.int64:
        dst = xp.astype(dst, xp.int64)

    # === Step 2. Sum source and destination embeddings ===
    return xp.take(type_ebed, src, axis=0) + xp.take(type_ebed, dst, axis=0)


def edge_cache_to_dtype(cache: EdgeCache, dtype: Any) -> EdgeCache:
    """
    Convert all floating-point tensors in EdgeCache to the specified dtype.

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
    EdgeCache
        New cache with converted tensors.
    """
    xp = array_api_compat.array_namespace(cache.edge_vec)
    # Handle Optional tensors explicitly.
    # Use local variables with explicit None check and assignment.
    _D_full = cache.D_full
    _Dt_full = cache.Dt_full
    cached_D_packed = cache.D_packed
    _edge_src_gate = cache.edge_src_gate
    _edge_quat = cache.edge_quat
    D_full: Any = None
    Dt_full: Any = None
    D_packed: Any = None
    edge_src_gate: Any = None
    edge_quat: Any = None
    if _D_full is not None:
        D_full = xp.astype(_D_full, dtype)
    if _Dt_full is not None:
        Dt_full = xp.astype(_Dt_full, dtype)
    if cached_D_packed is not None:
        D_packed = xp.astype(cached_D_packed, dtype)
    if _edge_src_gate is not None:
        edge_src_gate = xp.astype(_edge_src_gate, dtype)
    if _edge_quat is not None:
        edge_quat = xp.astype(_edge_quat, dtype)

    # CSR views contain only integer topology. Preserve them across the dtype
    # conversion so every accelerated consumer shares the per-step sort.
    return EdgeCache(
        src=cache.src,
        dst=cache.dst,
        edge_type_feat=xp.astype(cache.edge_type_feat, dtype),
        edge_vec=xp.astype(cache.edge_vec, dtype),
        edge_rbf=xp.astype(cache.edge_rbf, dtype),
        edge_env=xp.astype(cache.edge_env, dtype),
        deg=xp.astype(cache.deg, dtype),
        inv_sqrt_deg=xp.astype(cache.inv_sqrt_deg, dtype),
        D_full=D_full,
        Dt_full=Dt_full,
        D_packed=D_packed,
        D_to_m_cache=None if cache.D_to_m_cache is None else {},
        Dt_from_m_cache=None if cache.Dt_from_m_cache is None else {},
        csr_cache=None if cache.csr_cache is None else dict(cache.csr_cache),
        edge_src_gate=edge_src_gate,
        edge_quat=edge_quat,
        edge_mask=cache.edge_mask,
        cute_infer_so2_metadata=cache.cute_infer_so2_metadata,
        destinations_sorted=cache.destinations_sorted,
    )
