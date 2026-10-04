# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch runtime helpers for the DPA4 edge cache."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import torch

if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeCache,
    )


def cached_edge_csr(
    edge_cache: EdgeCache, endpoint: str, n_node: int | torch.SymInt
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the CSR view of one edge endpoint, built once per step.

    Several accelerated operators walk the edges of one endpoint in segment
    order: the fused convolution and the initial embedding on the CUDA path,
    the flash aggregation and the rotate-mix backward on the Triton path. They
    all share one edge set, so the sorted view is built once and kept on the
    edge cache; whichever consumer runs first pays for it.

    Parameters
    ----------
    edge_cache : EdgeCache
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
