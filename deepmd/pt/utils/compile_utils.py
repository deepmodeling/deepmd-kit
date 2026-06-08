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
    """Strip ``aten.detach`` chains that ``make_fx`` inserts for saved tensors.

    When ``make_fx`` decomposes ``autograd.grad(..., create_graph=True)``,
    the autograd engine wraps every saved forward activation in a
    double-detach chain (e.g. ``tanh -> detach_A -> detach_B ->
    tanh_backward``).  These nodes sever the second-order gradient path
    from the force loss back to model parameters.

    User-explicit ``.detach()`` calls are preserved via three topology
    rules that identify only the make_fx-inserted chains:

    * *Chain inner*: input is another detach node.
    * *Dead node*: no downstream users.
    * *Chain head*: all users are detach nodes.

    Any detach that matches none of these is left untouched.
    """
    _DETACH = torch.ops.aten.detach.default

    def _is_detach(n: torch.fx.Node) -> bool:
        return n.op == "call_function" and n.target == _DETACH

    # Pass 1 — classify against the original graph before any mutation.
    to_remove: list[torch.fx.Node] = []
    for node in gm.graph.nodes:
        if not _is_detach(node):
            continue
        input_node = node.args[0]
        users = list(node.users.keys())
        is_chain_inner = _is_detach(input_node)
        is_dead = len(users) == 0
        is_chain_head = len(users) > 0 and all(_is_detach(u) for u in users)
        if is_chain_inner or is_dead or is_chain_head:
            to_remove.append(node)

    # Pass 2 — rewire and erase after full classification.
    for node in to_remove:
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
