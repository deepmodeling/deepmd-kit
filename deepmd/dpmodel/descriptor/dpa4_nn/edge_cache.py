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
import numpy as np

from deepmd.dpmodel.array_api import (
    xp_add_at,
    xp_asarray_nodetach,
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
EdgeTypeKeepMaskFn = Callable[[Any, Any, Any], Any]


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
        :func:`build_edge_cache_from_edges` path, where masking is folded into
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
    edge_src_gate: Any = None
    edge_quat: Any = None
    edge_mask: Any = None


def compute_edge_src_gate(
    *,
    edge_len: Any,
    src: Any,
    n_nodes: int,
    bridging_switch: Callable[[Any], Any],
    edge_keep_f: Any = None,
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
    eta_nonzero_path = xp.exp(log_eta)

    # === Step 3. Exact-zero indicator per source node ===
    # ``scatter_add`` over an ``int64`` cast of the zero mask counts how
    # many frozen edges each source node owns. A strictly positive count
    # means the product is 0 by the hard-freeze rule.
    zero_count = xp_add_at(
        xp.zeros((n_nodes,), dtype=xp.int64, device=device),
        src,
        xp.astype(is_zero, xp.int64),
    )
    any_zero = zero_count > 0

    # === Step 4. Combine and broadcast back to edges via source ===
    eta = xp.where(any_zero, xp.zeros_like(eta_nonzero_path), eta_nonzero_path)
    return xp.take(eta, src, axis=0)[:, None]


def build_edge_cache(
    *,
    type_ebed: Any,
    extended_coord: Any,
    nlist: Any,
    mapping: Any,
    pair_keep_mask: Any,
    eps: float,
    deg_norm_floor: float,
    edge_envelope: Callable[[Any], Any],
    radial_basis: Callable[[Any], Any],
    n_radial: int,  # unused in padded layout; kept for pt signature parity
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    gamma: Any = None,
) -> EdgeCache:
    """
    Build the global edge cache from a DeePMD padded neighbor list.

    This converts DeePMD's per-frame padded neighbor list into the per-edge
    tensors reused across blocks. Where pt extracts a sparse list of valid
    edges with ``torch.nonzero`` (data-dependent length), the array-API port
    keeps one edge slot for every neighbor slot, so ``E = nf * nloc * nnei``
    flattened row-major and ``dst == repeat(arange(nf * nloc), nnei)``. Invalid
    slots (``nlist == -1`` padding, excluded type pairs, out-of-range mapped
    sources) stay in the arrays, flagged by ``edge_mask``; their geometry,
    envelope, radial basis, and type features are masked to zero.

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

    Gathered edge vectors on invalid slots are garbage (placeholder index 0)
    and may even be exactly zero (self-difference), which would produce a 0/0
    in the normalization inside the quaternion construction. Although the
    forward contribution of such slots is masked out downstream, a NaN there
    would still poison the backward pass (``where`` propagates NaN gradients
    from the unselected branch). Invalid slots are therefore rewritten to the
    safe dummy unit vector ``+z`` before any norm/quaternion/Wigner evaluation,
    and their envelope, radial basis, and type features are multiplied by the
    mask so they are exactly zero.

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
        normalization.
    edge_envelope
        C^3 edge envelope module.
    radial_basis
        Radial basis module.
    n_radial
        Number of radial basis channels. Unused here; kept for signature
        parity with pt.
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable that converts edge-aligned quaternions into packed Wigner-D
        blocks.
    gamma
        Optional per-edge roll angles with shape (E,), used only when
        ``random_gamma`` is True. pt draws gamma internally with
        ``torch.rand`` and the draw cannot be reproduced here, so callers
        needing determinism (e.g. tests) inject the angles explicitly. When
        None, angles are drawn from ``numpy.random.default_rng()`` uniformly
        in ``[0, 2*pi)``, matching pt's distribution.

    Returns
    -------
    EdgeCache
        Padded per-edge cache with ``edge_mask`` set.
    """
    xp = array_api_compat.array_namespace(type_ebed, extended_coord, nlist)
    device = array_api_compat.device(extended_coord)
    nf, nloc, nnei = nlist.shape
    nall = extended_coord.shape[1]
    n_nodes = nf * nloc

    # === Step 1. Validity mask and safe indices (pt edge_keep semantics) ===
    mask, nlist_safe, src_local_safe = _build_edge_mask_and_src(
        xp, nlist, mapping, pair_keep_mask, nall
    )
    mask_flat = xp.reshape(mask, (-1,))

    # === Step 2. Node indices ===
    # dst is slot-implicit: arange(nf * nloc) repeated nnei times (contract).
    frame_idx = xp.reshape(xp.arange(nf, dtype=xp.int64, device=device), (nf, 1, 1))
    src = xp.reshape(frame_idx * nloc + src_local_safe, (-1,))
    node_idx = xp.arange(n_nodes, dtype=xp.int64, device=device)
    dst = xp.reshape(xp.broadcast_to(node_idx[:, None], (n_nodes, nnei)), (-1,))

    # === Step 3. Gather per-edge geometry from extended coordinates ===
    # edge_vec points from center -> neighbor: r_ij = r_j - r_i (in Å).
    coord_flat = xp.reshape(extended_coord, (nf * nall, 3))
    neighbor_coord_index = xp.reshape(frame_idx * nall + nlist_safe, (-1,))
    loc_idx = xp.reshape(xp.arange(nloc, dtype=xp.int64, device=device), (1, nloc, 1))
    center_ext = xp.broadcast_to(frame_idx * nall + loc_idx, (nf, nloc, nnei))
    center_coord_index = xp.reshape(center_ext, (-1,))
    neighbor_pos = xp.take(coord_flat, neighbor_coord_index, axis=0)
    center_pos = xp.take(coord_flat, center_coord_index, axis=0)
    vec = neighbor_pos - center_pos  # (E, 3)

    # === Step 4. Rewrite invalid slots to the safe +z dummy vector ===
    # Gradient safety: see the function docstring. edge_len is the scalar
    # distance, computed only after the safe rewrite so it stays finite.
    maskf = xp.astype(mask_flat, vec.dtype)[:, None]  # (E, 1)
    z_unit = xp_asarray_nodetach(
        xp, np.array([[0.0, 0.0, 1.0]]), dtype=vec.dtype, device=device
    )
    edge_vec = vec * maskf + (1.0 - maskf) * z_unit
    edge_len = safe_norm(edge_vec, eps)  # (E, 1)

    # === Step 5. Envelope and radial basis, masked to zero on invalid slots ===
    # Edges with r >= rcut are not removed from the cache. Their envelope is
    # exactly zero, so messages vanish naturally while degree normalization
    # remains smooth at the cutoff boundary.
    edge_env = edge_envelope(edge_len) * maskf  # (E, 1)
    edge_rbf = radial_basis(edge_len) * maskf  # (E, n_radial)

    # === Step 6. Edge quaternion -> Wigner-D blocks ===
    D_full, Dt_full, edge_quat = _build_edge_wigner(
        edge_vec=edge_vec,
        edge_len=edge_len,
        eps=eps,
        random_gamma=random_gamma,
        wigner_calc=wigner_calc,
        gamma=gamma,
        build_full=build_wigner,
    )  # (E, D, D), (E, D, D), (E, 4)

    # === Step 7. Edge type features (src + dst), masked ===
    edge_type_feat = build_edge_type_feat(type_ebed, src, dst) * xp.astype(
        maskf, type_ebed.dtype
    )  # (E, C)

    # === Step 8. Smooth destination degrees ===
    # pt accumulates env^2 with ``index_add_`` over dst; in the padded
    # node-contiguous layout this is a plain masked sum over the nnei axis.
    # edge_env is already exactly zero on invalid slots.
    env_sq = xp.reshape(edge_env[:, 0] * edge_env[:, 0], (n_nodes, nnei))
    deg = xp.sum(env_sq, axis=1)  # (N,)
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
        edge_src_gate=None,
        edge_quat=edge_quat,
        edge_mask=mask_flat,
    )


def build_edge_cache_from_edges(
    *,
    type_ebed: Any,
    atype_flat: Any,
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
    has_exclude_types: bool,
    edge_type_keep_mask: EdgeTypeKeepMaskFn,
    random_gamma: bool,
    wigner_calc: WignerCalculatorFn,
    build_wigner: bool = True,
    gamma: Any = None,
) -> EdgeCache:
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
    gamma
        Optional per-edge roll angles with shape (E,), used only when
        ``random_gamma`` is True. pt draws gamma internally with
        ``torch.rand`` and the draw cannot be reproduced here, so callers
        needing determinism (e.g. tests) inject the angles explicitly. When
        None, angles are drawn from ``numpy.random.default_rng()`` uniformly
        in ``[0, 2*pi)``, matching pt's distribution.

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

    # === Step 1. Normalize mask and apply type exclusions ===
    edge_keep = xp.astype(edge_mask, xp.bool)
    if has_exclude_types:
        edge_keep = edge_keep & edge_type_keep_mask(atype_flat, src, dst)

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
    edge_env = edge_envelope(edge_len) * edge_keep_f  # (E, 1)
    edge_rbf = radial_basis(edge_len) * edge_keep_f  # (E, n_radial)

    # === Step 4. Edge quaternion -> Wigner-D blocks ===
    D_full, Dt_full, edge_quat = _build_edge_wigner(
        edge_vec=edge_vec,
        edge_len=edge_len,
        eps=eps,
        random_gamma=random_gamma,
        wigner_calc=wigner_calc,
        gamma=gamma,
        build_full=build_wigner,
    )  # (E, D, D), (E, D, D), (E, 4)

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
        ``random_gamma`` is True. When None, angles are drawn from
        ``numpy.random.default_rng()`` uniformly in ``[0, 2*pi)``, matching
        pt's ``torch.rand`` distribution.
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
    # pt draws the roll with ``torch.rand``; here it is injected or drawn from
    # numpy so the array-API call site stays reproducible.
    if random_gamma:
        if gamma is None:
            gamma = np.random.default_rng().uniform(
                0.0, 2.0 * math.pi, edge_quat.shape[0]
            )
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
    edge_src_gate: Any = None,
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
    edge_src_gate
        Optional per-edge SFPG weight with shape (E, 1). ``None`` in
        non-bridging mode.

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
        edge_src_gate=edge_src_gate,
        edge_quat=edge_quat,
    )


def _build_edge_mask_and_src(
    xp: Any,
    nlist: Any,
    mapping: Any,
    pair_keep_mask: Any,
    nall: int,
) -> tuple[Any, Any, Any]:
    """
    Build the padded edge validity mask and safe source-local indices.

    This reproduces the pt edge-keep rules for the padded layout:

    - padding slots (``nlist == -1``) are invalid;
    - excluded type pairs (``pair_keep_mask == False``) are invalid;
    - after mapping the neighbor's extended index to a local index, slots
      whose source falls outside ``[0, nloc)`` are invalid (pt's ``src_ok``
      filter; e.g. broken mapping or ghost-only neighbors);
    - no distance-based filtering: edges beyond ``rcut`` stay valid and are
      zeroed naturally by the smooth envelope.

    Instead of dropping invalid slots (pt's ``torch.nonzero``), they are kept
    with ``mask == False`` and safe (index 0) placeholder indices.

    Parameters
    ----------
    xp
        Array namespace.
    nlist
        Neighbor list with shape (nf, nloc, nnei); -1 marks padding.
    mapping
        Extended-to-local mapping with shape (nf, nall), or None if the
        neighbor indices are already local.
    pair_keep_mask
        Pair exclusion keep mask with shape (nf, nloc, nnei). True means keep.
    nall
        Number of atoms on the extended axis per frame.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(mask, nlist_safe, src_local_safe)``, all with shape
        (nf, nloc, nnei). ``mask`` is boolean; the two index arrays are int64
        with 0 substituted on invalid slots.
    """
    nf, nloc, nnei = nlist.shape
    nlist = xp.astype(nlist, xp.int64)
    mask = (nlist >= 0) & pair_keep_mask
    nlist_safe = xp.where(mask, nlist, xp.zeros_like(nlist))

    if mapping is None:
        # Neighbor indices are already local indices in [0, nloc).
        src_local = nlist_safe
    else:
        # Map extended index -> local index for each frame.
        mapping_flat = xp.astype(xp.reshape(mapping, (-1,)), xp.int64)
        frame_idx = xp.reshape(
            xp.arange(nf, dtype=xp.int64, device=array_api_compat.device(nlist)),
            (nf, 1, 1),
        )
        flat_idx = xp.reshape(frame_idx * nall + nlist_safe, (-1,))
        src_local = xp.reshape(xp.take(mapping_flat, flat_idx, axis=0), nlist.shape)

    # pt's src_ok filter: drop (here: mask) edges mapping outside [0, nloc).
    mask = mask & (src_local >= 0) & (src_local < nloc)
    src_local_safe = xp.where(mask, src_local, xp.zeros_like(src_local))
    # Re-zero nlist_safe after the src_ok update so coordinate gathers stay
    # in-bounds when callers pass local nlists with out-of-range entries.
    nlist_safe = xp.where(mask, nlist_safe, xp.zeros_like(nlist_safe))
    return mask, nlist_safe, src_local_safe


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
    _edge_src_gate = cache.edge_src_gate
    _edge_quat = cache.edge_quat
    D_full: Any = None
    Dt_full: Any = None
    edge_src_gate: Any = None
    edge_quat: Any = None
    if _D_full is not None:
        D_full = xp.astype(_D_full, dtype)
    if _Dt_full is not None:
        Dt_full = xp.astype(_Dt_full, dtype)
    if _edge_src_gate is not None:
        edge_src_gate = xp.astype(_edge_src_gate, dtype)
    if _edge_quat is not None:
        edge_quat = xp.astype(_edge_quat, dtype)

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
        edge_src_gate=edge_src_gate,
        edge_quat=edge_quat,
        edge_mask=cache.edge_mask,
    )
