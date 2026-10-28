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
        C^3 cutoff envelope weights with shape (E, 1). In bridging mode the
        Source Freeze Propagation Gate is folded into it: the envelope of the
        edge ``j -> k`` is multiplied by the product of the switching
        amplitudes ``w(r_{jl})`` of every pair of the source ``j`` other than
        the pair ``(j, k)`` itself, so an edge whose source has a third
        neighbor in the frozen zone vanishes everywhere the envelope enters
        (messages, environment seed, attention masses and degree), exactly as
        an edge beyond the cutoff does, while the edge between the two atoms
        of a frozen pair keeps its clamped geometry.
    node_gate
        Per-node source gate ``eta[j] = prod_{k in N(j)} w(r_{jk})`` with
        shape (N,), the product over every pair of the node; ``None`` without
        bridging. The atomic model fades the learned atomic energy of node
        ``j`` with it.
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
        per step and shared by every consumer.
    edge_mask
        Validity mask for the padded standard-path layout with shape (E,) or
        (E, 1); 1 marks a real edge, 0 a padded/invalid slot. ``None`` means
        all slots are valid (e.g. the sparse
        :func:`_edge_cache_from_arrays` path, where masking is folded into
        the per-edge weights). This field has no pt counterpart.
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
    csr_cache: dict[str, Any] | None = field(default_factory=dict)
    edge_quat: Any = None
    edge_mask: Any = None
    node_gate: Any = None


def compute_source_gates(
    *,
    edge_len: Any,
    edge_contact: Any,
    src: Any,
    n_nodes: int,
    bridging_switch: Callable[[Any, Any], Any],
    edge_keep_f: Any = None,
    node_partial_exchange: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any]:
    """
    Compute the per-node and per-edge source gates of SFPG from edge lengths.

    The per-node gate is the "non-frozen confidence" of a node, the per-edge
    gate the same product with the edge's own pair left out::

        w_e      = bridging_switch(edge_len_e, edge_contact_e)   in [0, 1]
        eta_j    = prod_{e: src_e = j} w_e                       in [0, 1]
        gate_e   = eta_{src_e} / w_e = prod_{e': src_e' = src_e, e' != e} w_e'

    ``w_e = 0`` inside the inner radius of that edge's own pair ensures
    ``eta_j = 0`` for any node with at least one neighbor in the frozen zone,
    and ``gate_e = 0`` for every edge such a node emits except the edge to
    that frozen neighbor. The cache builder multiplies ``gate_e`` into the
    edge envelope, which every consumer weights by, so a muted edge is
    indistinguishable from a deleted one: the frozen atom disappears from the
    environment of every third atom, while the two atoms of a frozen pair
    keep seeing each other at the clamped distance. The fitting scales the
    deviation of its output from the bias of the type by ``eta_j``, so the
    frozen atom contributes that bias alone. Masked edges (padding,
    excluded type pairs) must contribute the multiplicative identity ``1`` so
    they never spuriously mute a valid source node; callers supply
    ``edge_keep_f`` for this.

    The product is decomposed into a log-sum on non-zero contributions
    combined with an explicit "any zero per group" indicator that routes the
    frozen case through ``where``; the leave-one-out product subtracts the
    edge's own log-amplitude and zero flag from the totals of its source.
    Both branches use only shape-preserving standard ops, mirroring the pt
    implementation so the two backends agree bitwise in float64.

    The gradient consequence at the plateau is exact: ``BridgingSwitch``
    places ``w'(r) = 0`` everywhere inside the inner radius, so the chain rule
    ``d eta / d r = (leave-one-out factor) * w'(r) = anything * 0 = 0``
    holds regardless of how the muted ``where`` branch treats the upstream
    gradient. In the transition zone every edge has strictly positive ``w``
    and the log-sum branch gives the standard product gradient.

    Parameters
    ----------
    edge_len
        Per-edge distances with shape (E, 1).
    edge_contact
        Per-edge length scale with shape (E, 1) in Å, the sum of the two
        endpoint radii. The switch measures the window against it, so every
        element combination gets its own pair of radii.
    src
        Source node indices with shape (E,).
    n_nodes
        Total number of nodes N.
    bridging_switch
        Callable ``(r, contact) -> w`` with ``w: [0, ∞) -> [0, 1]``, typically
        a :class:`BridgingSwitch` instance.
    edge_keep_f
        Optional per-edge keep weights with shape (E, 1), with ``0`` on
        masked edges and ``1`` on kept edges. If provided, masked edges
        are rewritten to ``w = 1`` before the product reduction.
    node_partial_exchange
        Optional cross-rank completion hook for the per-node partials
        (issue #5906). Receives the ``(n_nodes, 2)`` float array
        ``[log_eta, zero_count]`` and returns the globally completed one
        (reverse-accumulate ghost rows into owners, then broadcast the
        completed owner values back onto ghosts). ``None`` (the default)
        is the single-process path where the local partials are already
        complete.

    Returns
    -------
    node_gate : Array
        Per-node source gate ``eta`` with shape (N,).
    edge_gate : Array
        Per-edge leave-one-out gate with shape (E, 1), aligned on the same
        edge axis as the rest of the cache.
    """
    xp = array_api_compat.array_namespace(edge_len, src)
    device = array_api_compat.device(edge_len)
    # === Step 1. Per-edge switching amplitude w(r) in [0, 1] ===
    edge_w = bridging_switch(edge_len, edge_contact)  # (E, 1)
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
    is_zero_f = xp.astype(is_zero, edge_w.dtype)
    zero_count = xp_add_at(
        xp.zeros((n_nodes,), dtype=edge_w.dtype, device=device),
        src,
        is_zero_f,
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

    # === Step 4. Per-node gate ===
    node_gate = xp.where(zero_count > 0.5, xp.zeros_like(log_eta), xp.exp(log_eta))

    # === Step 5. Per-edge leave-one-out gate ===
    # The edge's own factor is removed from its source's totals: its
    # log-amplitude (0 when it is a frozen edge, because ``safe_w`` is 1 there)
    # and its zero flag, so the remaining product is over the other pairs.
    loo_log = xp.take(log_eta, src, axis=0) - log_safe  # (E,)
    loo_zero = xp.take(zero_count, src, axis=0) - is_zero_f  # (E,)
    edge_gate = xp.where(loo_zero > 0.5, xp.zeros_like(loo_log), xp.exp(loo_log))
    return node_gate, edge_gate[:, None]


def _edge_cache_from_arrays(
    *,
    type_ebed: Any,
    edge_index: Any,
    edge_vec: Any,
    edge_mask: Any,
    compute_dtype: Any,
    eps: float,
    deg_norm_floor: float,
    bridging_clamp: Callable[[Any, Any], Any] | None,
    bridging_switch: Callable[[Any, Any], Any] | None,
    edge_contact: Any,
    edge_envelope: Callable[[Any], Any],
    radial_basis: Callable[[Any], Any],
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    gamma: Any = None,
    node_partial_exchange: Callable[[Any], Any] | None = None,
    fused_radial: FusedRadialFn | None = None,
    fused_wigner: WignerCalculatorFn | None = None,
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
    bridging_clamp
        Optional distance clamp that freezes the geometry the descriptor sees
        for a pair inside its bridging window (see :class:`InnerClamp`).
    bridging_switch
        Optional C3 switching amplitude ``w(r, contact) -> [0, 1]`` that drives
        the Source Freeze Propagation Gate. When provided, the product of
        ``w(r_{jl})`` over the true lengths of every pair of the source node
        other than the edge's own pair is folded into the envelope of every
        edge that node emits, and the full product is stored on the cache as
        ``node_gate``. Masked
        edges (``edge_keep=False``) are forced to ``w=1`` so they never
        leak into the product.
    edge_contact
        Per-edge length scale with shape (E, 1) in Å that both window modules
        measure their fractions against, or ``None`` when neither is given.
        The descriptor owns the per-type radii and resolves it before the call.
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
    gamma
        Optional per-edge roll angles with shape (E,), used only when
        ``random_gamma`` is True. When None, drawn with the backend's RNG
        (:func:`~deepmd.dpmodel.array_api.xp_uniform`) uniformly in
        ``[0, 2*pi)``; callers may inject angles to pin a draw.
    node_partial_exchange
        Optional cross-rank completion hook forwarded to
        :func:`compute_source_gates` (issue #5906); only meaningful when
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
    edge_len_true = edge_len
    if bridging_clamp is not None:
        clamped = bridging_clamp(edge_len, edge_contact)
        scale = clamped / edge_len
        edge_vec = edge_vec * scale
        edge_len = clamped
    if fused_radial is not None:
        edge_env, edge_rbf = fused_radial(edge_len, edge_keep_f)
    else:
        edge_env = edge_envelope(edge_len) * edge_keep_f  # (E, 1)
        edge_rbf = radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

    # === Step 4. Edge quaternion -> Wigner-D blocks ===
    D_full, Dt_full, edge_quat = _build_edge_wigner(
        edge_vec=edge_vec,
        edge_len=edge_len,
        eps=eps,
        random_gamma=random_gamma,
        wigner_calc=fused_wigner if fused_wigner is not None else wigner_calc,
        gamma=gamma,
        build_full=build_wigner,
    )  # (E, D, D), (E, D, D), (E, 4)

    # === Step 5. Edge type features ===
    edge_type_feat = build_edge_type_feat(type_ebed, src, dst)
    edge_type_feat = edge_type_feat * xp.astype(edge_keep_f, edge_type_feat.dtype)

    # === Step 6. Source Freeze Propagation Gate (optional) ===
    # The leave-one-out gate is folded into the envelope, the one factor every
    # consumer of an edge weights by: the messages, the environment seed, the
    # attention masses and the degree normalization all see a muted edge
    # exactly as they see an edge beyond the cutoff. The gate reads the true
    # pair length, so it closes where the clamp above has frozen the geometry.
    # The radial basis stays ungated because it feeds networks whose output
    # the envelope multiplies. The sparse-edge path packs masked dummy edges
    # so the compiled graph sees a statically non-empty, non-singular edge
    # tensor; ``edge_keep_f`` rewrites any such slot to ``w=1`` inside
    # ``compute_source_gates``, keeping the product reduction unaffected by
    # padding. The per-node product travels on the cache to the atomic model.
    node_gate = None
    if bridging_switch is not None:
        node_gate, edge_gate = compute_source_gates(
            edge_len=edge_len_true,
            edge_contact=edge_contact,
            src=src,
            n_nodes=n_nodes,
            bridging_switch=bridging_switch,
            edge_keep_f=edge_keep_f,
            node_partial_exchange=node_partial_exchange,
        )
        edge_env = edge_env * edge_gate

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
        node_gate=node_gate,
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
) -> tuple[Any, Any, Any]:
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

    Returns
    -------
    tuple[Array, Array, Array]
        Packed Wigner-D matrices ``(D_full, Dt_full)`` with shape ``(E, D, D)``
        (or ``None`` when ``build_full`` is False) and the quaternion used to
        build them with shape ``(E, 4)``.
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
        return None, None, edge_quat
    D_full, Dt_full = wigner_calc(edge_quat)
    return D_full, Dt_full, edge_quat


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
    edge_quat: Any,
    deg_norm_floor: float,
    node_gate: Any = None,
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
    edge_quat
        Global-to-local quaternions used to build the Wigner-D matrices with
        shape (E, 4).
    deg_norm_floor
        Floor added to the envelope-squared degree before the inverse-sqrt
        normalization. A tiny ``eps`` reproduces the legacy behavior; an
        ``O(1)`` value makes sparse-neighborhood features vanish smoothly at
        ``rcut`` instead of saturating and kinking.
    node_gate
        Per-node source gate with shape (N,) of a bridged descriptor, or
        ``None``.

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
        D_to_m_cache={},
        Dt_from_m_cache={},
        csr_cache={},
        edge_quat=edge_quat,
        node_gate=node_gate,
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

    Integer tensors (src, dst) are unchanged, and so is the node gate, which
    no block reads: it is applied to the fitting output in global precision.
    This is a standalone function (not a method) to keep it side-effect free.

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
    _edge_quat = cache.edge_quat
    D_full: Any = None
    Dt_full: Any = None
    edge_quat: Any = None
    if _D_full is not None:
        D_full = xp.astype(_D_full, dtype)
    if _Dt_full is not None:
        Dt_full = xp.astype(_Dt_full, dtype)
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
        D_to_m_cache=None if cache.D_to_m_cache is None else {},
        Dt_from_m_cache=None if cache.Dt_from_m_cache is None else {},
        csr_cache=None if cache.csr_cache is None else dict(cache.csr_cache),
        edge_quat=edge_quat,
        edge_mask=cache.edge_mask,
        node_gate=cache.node_gate,
    )
