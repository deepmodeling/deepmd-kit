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
    edge_len: Any
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


def clamped_edge_geometry(
    edge_vec: Any, read_distance: Callable[[Any], Any], eps: float
) -> tuple[Any, Any]:
    """Map the physical radius while preserving each nonzero bond direction.

    Coincident atoms have a zero direction and the same frozen read radius as
    any other contact on the plateau. Regularize the radius after the clamp so
    that angular and radial consumers see the same effective geometry.
    """
    xp = array_api_compat.array_namespace(edge_vec)
    squared = xp.sum(edge_vec * edge_vec, axis=-1, keepdims=True)
    nonzero = squared > 0
    denominator = xp.linalg.vector_norm(
        xp.where(nonzero, edge_vec, xp.ones_like(edge_vec)), axis=-1, keepdims=True
    )
    length = xp.where(nonzero, denominator, xp.zeros_like(denominator))
    read = read_distance(length)
    direction = xp.where(nonzero, edge_vec / denominator, xp.zeros_like(edge_vec))
    mapped = xp.where(read == length, edge_vec, direction * read)
    return mapped, xp.sqrt(read * read + eps * eps)


def _edge_cache_from_arrays(
    *,
    type_ebed: Any,
    edge_index: Any,
    edge_vec: Any,
    edge_mask: Any,
    compute_dtype: Any,
    eps: float,
    deg_norm_floor: float,
    read_distance: Callable[[Any], Any] | None,
    edge_envelope: Callable[[Any], Any],
    radial_basis: Callable[[Any], Any],
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    gamma: Any = None,
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
    read_distance
        Optional map from the edge lengths, shape (E, 1), to the lengths the
        descriptor reads, the contact clamp of a bridged descriptor (see
        :class:`~deepmd.dpmodel.utils.contact_clamp.ContactClamp`). The edge
        vectors are rescaled to the read lengths, so every geometric read
        sees them while the directions stay exact.
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
        deg_norm_floor=deg_norm_floor,
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
    edge_len: Any,
    edge_rbf: Any,
    edge_env: Any,
    D_full: Any,
    Dt_full: Any,
    edge_quat: Any,
    deg_norm_floor: float,
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
        edge_len=xp.astype(cache.edge_len, dtype),
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
    )
