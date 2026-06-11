# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Edge cache construction for the dpmodel DPA4/SeZM descriptor.

This module defines the :class:`EdgeCache` dataclass (the dpmodel
counterpart of the pt ``EdgeFeatureCache`` NamedTuple from
``deepmd.pt.model.descriptor.sezm_nn.edge_cache``) and
:func:`build_edge_cache`, the padded-layout counterpart of pt's
sparse ``build_edge_cache``.

Padded-edge layout
------------------
The pt implementation extracts a *sparse* edge list with ``torch.nonzero``:
only valid neighbor slots become edges, and per-edge tensors have a
data-dependent length ``E``. The dpmodel implementation instead uses a
*padded* and frame-explicit edge layout: every neighbor slot of the DeePMD
neighbor list contributes one edge, so

    ``E = nf * nloc * nnei``

with per-edge tensors flattened from ``(nf, nloc, nnei, ...)`` in row-major
order. Invalid slots (``nlist == -1`` padding, excluded type pairs) stay in
the arrays and are marked by ``edge_mask == 0``. Edge slot ``(f, i, j)``
always belongs to destination node ``f * nloc + i``, so destination
aggregation is a masked sum over the ``nnei`` axis instead of a scatter.
"""

from __future__ import (
    annotations,
)

import math
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from .utils import (
    safe_norm,
)
from .wignerd import (
    build_edge_quaternion,
    quaternion_multiply,
    quaternion_z_rotation,
)


@dataclass
class EdgeCache:
    """
    Global edge feature cache created once per forward().

    All per-edge arrays are aligned on the same padded edge axis
    (``E = nf * nloc * nnei``); see the module docstring for the layout
    contract. Node-level arrays use the local node axis ``N = nf * nloc``.

    An ``EdgeCache`` must not be reused across forward passes:
    ``D_to_m_cache``/``Dt_from_m_cache`` are keyed only by ``"lmax:mmax"``,
    not by the contents of ``D_full``, so reuse with different Wigner blocks
    would silently return stale projections.

    Parameters
    ----------
    src
        Source (neighbor) node indices with shape (E,), pointing into the
        local node axis ``N = nf * nloc``. Invalid slots must hold a safe
        in-range index (their contribution is masked out by ``edge_mask``).
    dst
        Destination (center) node indices with shape (E,). In the padded
        layout this is slot-implicit and MUST equal
        ``arange(nf * nloc)`` with each index repeated ``nnei`` consecutive
        times (i.e. ``np.repeat(np.arange(nf * nloc), nnei)``;
        node-contiguous order); aggregation code relies on this ordering.
    edge_type_feat
        Per-edge type embeddings with shape (E, C), computed as src+dst.
    edge_vec
        Edge vectors with shape (E, 3) in Å.
    edge_rbf
        Radial basis with shape (E, n_radial).
        The C^3 cutoff envelope is already baked in.
    edge_env
        C^3 cutoff envelope weights with shape (E, 1). Zero on invalid slots.
    deg
        Envelope-squared smooth degree with shape (N,), computed as the
        masked ``sum(edge_env**2)`` over each node's ``nnei`` slots.
    inv_sqrt_deg
        Inverse square root smooth degree normalization with shape (N, 1, 1).
    D_full
        Block-diagonal Wigner-D matrix with shape (E, D, D) where D=(lmax+1)^2.
        Used for efficient batched rotation. None if not available.
    Dt_full
        Transpose of D_full with shape (E, D, D). None if not available.
    D_to_m_cache
        Lazy cache for projected D matrices keyed by a normalized
        ``"lmax:mmax"`` identifier. The key does not capture the contents
        of ``D_full``, so the cache is only valid for the forward pass
        that created this ``EdgeCache`` (see the class docstring).
    Dt_from_m_cache
        Lazy cache for projected Dt matrices keyed by a normalized
        ``"lmax:mmax"`` identifier. Same single-forward-pass validity
        caveat as ``D_to_m_cache``.
    edge_src_gate
        Optional per-edge Source Freeze Propagation Gate (SFPG) weight with
        shape (E, 1). Present only in bridging mode; ``None`` otherwise.
    edge_quat
        Per-edge global-to-local quaternion used to build ``D_full`` and
        ``Dt_full`` with shape (E, 4). None if not available.
    edge_mask
        Validity mask for the padded-edge layout with shape (E,) or (E, 1);
        nonzero (1) marks a real edge, zero marks a padded/invalid slot.
        ``None`` means all slots are valid. This field has no pt counterpart:
        pt's sparse edge list contains valid edges only, while dpmodel keeps
        the padded ``nf * nloc * nnei`` slots and masks the invalid ones.
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


def _build_edge_mask_and_src(
    xp: Any,
    nlist: Any,
    mapping: Any,
    pair_keep_mask: Any,
    nall: int,
) -> tuple[Any, Any, Any]:
    """
    Build the padded edge validity mask and safe source-local indices.

    Mirrors the pt edge-keep semantics of
    ``sezm_nn.edge_cache._build_standard_edge_index`` exactly:

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
    tuple[Any, Any, Any]
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


def build_edge_cache(
    *,
    type_ebed: Any,
    extended_coord: Any,
    nlist: Any,
    mapping: Any,
    pair_keep_mask: Any,
    eps: float,
    deg_norm_floor: float,
    edge_envelope: Any,
    radial_basis: Any,
    n_radial: int,  # unused: kept for pt signature parity (pt sizes its empty cache)
    random_gamma: bool,
    wigner_calc: Any,
    gamma: Any = None,
) -> EdgeCache:
    """
    Build the global padded edge cache from a DeePMD padded neighbor list.

    Padded counterpart of pt ``sezm_nn.edge_cache.build_edge_cache``. Instead
    of extracting a sparse edge list with ``torch.nonzero`` (data-dependent
    length), every neighbor slot becomes one edge slot:
    ``E = nf * nloc * nnei`` flattened row-major, with invalid slots marked by
    ``edge_mask == 0`` (see the :class:`EdgeCache` layout contract). In
    particular ``dst == np.repeat(arange(nf * nloc), nnei)`` always, and there
    is no empty-cache special case (E is shape-determined).

    Masked-slot safety: gathered edge vectors on invalid slots are garbage
    (placeholder index 0), and could even be exactly zero (self-difference),
    which would produce a 0/0 in the normalization inside the quaternion
    construction. Although the *forward* contribution of such slots is masked
    out downstream, a NaN there would still poison the *backward* pass
    (``where`` propagates NaN gradients from the unselected branch). Invalid
    slots are therefore rewritten to the safe dummy unit vector ``+z`` BEFORE
    any norm/quaternion/Wigner evaluation, and their envelope, radial basis,
    and type features are multiplied by the mask so they are exactly zero.

    Parameters
    ----------
    type_ebed
        Per-node type embedding with shape (N, C), where N = nf * nloc.
    extended_coord
        Extended coordinates with shape (nf, nall, 3).
    nlist
        Neighbor list with shape (nf, nloc, nnei); -1 marks padding.
    mapping
        Mapping from extended to local indices with shape (nf, nall), or None
        when the neighbor indices are already local.
    pair_keep_mask
        Pair keep mask from ``PairExcludeMask`` with shape (nf, nloc, nnei).
        True means keep.
    eps
        Small positive epsilon for safe norm / quaternion construction.
    deg_norm_floor
        Floor added to the envelope-squared degree before the inverse-sqrt
        normalization.
    edge_envelope
        C^3 edge envelope callable ``(E, 1) -> (E, 1)``.
    radial_basis
        Radial basis callable ``(E, 1) -> (E, n_radial)`` (envelope baked in).
    n_radial
        Number of radial basis channels. Unused in the padded layout (kept
        for signature parity with pt, where it sizes the empty cache).
    random_gamma
        Whether to apply a random roll around the local +Z axis before
        constructing Wigner-D blocks.
    wigner_calc
        Callable converting edge quaternions (E, 4) into packed Wigner-D
        blocks ``(D_full, Dt_full)``.
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
    n_edge = n_nodes * nnei

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
    coord_flat = xp.reshape(extended_coord, (nf * nall, 3))
    neighbor_coord_index = xp.reshape(frame_idx * nall + nlist_safe, (-1,))
    loc_idx = xp.reshape(xp.arange(nloc, dtype=xp.int64, device=device), (1, nloc, 1))
    center_ext = xp.broadcast_to(frame_idx * nall + loc_idx, (nf, nloc, nnei))
    center_coord_index = xp.reshape(center_ext, (-1,))
    neighbor_pos = xp.take(coord_flat, neighbor_coord_index, axis=0)
    center_pos = xp.take(coord_flat, center_coord_index, axis=0)
    vec = neighbor_pos - center_pos  # (E, 3)

    # === Step 4. Rewrite invalid slots to the safe +z dummy vector ===
    # Gradient safety: see the function docstring.
    maskf = xp.astype(mask_flat, vec.dtype)[:, None]  # (E, 1)
    z_unit = xp.asarray(np.array([[0.0, 0.0, 1.0]]), dtype=vec.dtype, device=device)
    edge_vec = vec * maskf + (1.0 - maskf) * z_unit
    edge_len = safe_norm(edge_vec, eps)  # (E, 1)

    # === Step 5. Envelope and radial basis, masked to zero on invalid slots ===
    edge_env = edge_envelope(edge_len) * maskf  # (E, 1)
    edge_rbf = radial_basis(edge_len) * maskf  # (E, n_radial)

    # === Step 6. Edge quaternion -> Wigner-D blocks ===
    edge_quat = build_edge_quaternion(edge_vec, edge_len=edge_len, eps=eps)
    if random_gamma:
        if gamma is None:
            gamma = np.random.default_rng().uniform(0.0, 2.0 * math.pi, n_edge)
        gamma = xp.astype(xp.asarray(gamma, device=device), edge_quat.dtype)
        edge_quat = quaternion_multiply(quaternion_z_rotation(gamma), edge_quat)
    D_full, Dt_full = wigner_calc(edge_quat)

    # === Step 7. Edge type features (src + dst), masked ===
    edge_type_feat = (
        xp.take(type_ebed, src, axis=0) + xp.take(type_ebed, dst, axis=0)
    ) * xp.astype(maskf, type_ebed.dtype)

    # === Step 8. Smooth destination degrees ===
    # pt accumulates env^2 with index_add_ over dst (edge_cache.py:622); in the
    # padded node-contiguous layout this is a plain sum over the nnei axis.
    # edge_env is already exactly zero on invalid slots.
    env_sq = xp.reshape(edge_env[:, 0] * edge_env[:, 0], (n_nodes, nnei))
    deg = xp.sum(env_sq, axis=1)  # (N,)
    inv_sqrt_deg = xp.reshape(1.0 / xp.sqrt(deg + deg_norm_floor), (n_nodes, 1, 1))

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
