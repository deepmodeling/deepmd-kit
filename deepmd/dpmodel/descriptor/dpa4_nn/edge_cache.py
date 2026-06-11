# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Edge cache container for the dpmodel DPA4/SeZM descriptor.

This module currently defines only the :class:`EdgeCache` dataclass (the
dpmodel counterpart of the pt ``EdgeFeatureCache`` NamedTuple from
``deepmd.pt.model.descriptor.sezm_nn.edge_cache``). The full
``build_edge_cache`` machinery is ported in a later task.

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

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
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
