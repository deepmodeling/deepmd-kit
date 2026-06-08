# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared utilities for ``make_fx`` / ``torch.compile`` tracing.

Used by ``deepmd.pt.model.model.sezm_model`` and
``deepmd.pt_expt.train.training``.
"""

from __future__ import (
    annotations,
)

import torch


def _is_prime(n: int) -> bool:
    """Return True when ``n`` is a prime integer (``n >= 2``)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    k = 3
    while k * k <= n:
        if n % k == 0:
            return False
        k += 2
    return True


def _next_safe_prime(start: int, forbidden: set[int]) -> int:
    """Return the smallest prime ``>= max(start, 5)`` not in ``forbidden``.

    Primes >= 5 avoid PyTorch's 0/1 specialization and the Cartesian /
    virial / charge-spin literals (2, 3, 9) that make_fx can unify with
    free-dim symbols under ``tracing_mode="symbolic"``.  Distinct values
    suppress duck-shape merging without needing ``ShapeEnv(duck_shape=False)``.
    """
    n = max(start, 5)
    while not _is_prime(n) or n in forbidden:
        n += 1
    return n


def _trace_pad_dim(t: torch.Tensor, dim: int, target: int) -> torch.Tensor:
    """Pad or trim ``t`` along ``dim`` so ``t.shape[dim] == target``.

    Padding duplicates the last slice; trimming drops trailing slices.
    Duplicating the last slice preserves valid index values inside
    index-bearing tensors (nlist, mapping) because the copied row reuses
    previously-valid entries.  Only shapes flow downstream during make_fx
    tracing, so the exact replicated values do not affect the FX graph.
    """
    cur = int(t.shape[dim])
    if cur == target:
        return t
    if cur > target:
        sl: list[slice] = [slice(None)] * t.ndim
        sl[dim] = slice(None, target)
        return t[tuple(sl)]
    sl = [slice(None)] * t.ndim
    sl[dim] = slice(-1, None)
    last = t[tuple(sl)]
    return torch.cat([t, *([last] * (target - cur))], dim=dim)


def strip_saved_tensor_detach(gm: torch.fx.GraphModule) -> None:
    """Strip all ``aten.detach`` nodes that ``make_fx`` inserts for saved tensors.

    When ``make_fx`` decomposes ``autograd.grad(..., create_graph=True)``,
    the autograd engine wraps saved forward activations in detach nodes
    (e.g. ``tanh -> detach_A -> detach_B -> tanh_backward``, or a single
    ``activation -> detach_A -> backward_op`` for attention models).
    These nodes sever the second-order gradient path from the force loss
    back to model parameters.

    All ``aten.detach.default`` nodes in the traced graph are removed.
    User-explicit ``.detach()`` calls (e.g. ``coord.detach().requires_grad_(True)``
    inside the traced function) are safe to remove because the caller
    already detaches and sets ``requires_grad`` on the input before
    invoking the compiled graph.
    """
    _DETACH = torch.ops.aten.detach.default
    for node in list(gm.graph.nodes):
        if node.op == "call_function" and node.target == _DETACH:
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
    gm.graph.lint()
    gm.recompile()


def rebuild_graph_module(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Copy all nodes into a fresh ``torch.fx.Graph``.

    After ``Graph.erase_node()`` the C-level prev/next pointers on
    neighbouring ``Node`` objects may become stale.  When ``torch.compile``
    (dynamo) later re-traces the graph it walks these pointers, which can
    cause segfaults.  Rebuilding into a new graph eliminates stale pointers.
    """
    old_graph = gm.graph
    new_graph = torch.fx.Graph()
    val_map: dict[torch.fx.Node, torch.fx.Node] = {}
    for node in old_graph.nodes:
        val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    new_graph.lint()
    return torch.fx.GraphModule(gm, new_graph)
