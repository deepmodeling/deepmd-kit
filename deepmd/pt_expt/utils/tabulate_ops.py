# SPDX-License-Identifier: LGPL-3.0-or-later
"""Register fake tensor implementations for deepmd custom tabulate ops.

These registrations enable torch.export and make_fx to trace through the
compressed forward path, which uses C++ custom ops (tabulate_fusion_se_*).
Without fake implementations, torch.export cannot determine output shapes.

This module is imported at package init time (via utils/__init__.py) so
registrations happen before any descriptor code runs.  If the C++ custom
op library hasn't been loaded yet at that point, `ensure_fake_registered()`
can be called again later (it is idempotent) — e.g. from the compression
entry point after the ops become available.

When the C++ custom op library is loaded, the ops already have
implementations, and register_fake will raise RuntimeError. We silently
skip registration in that case.
"""

from collections.abc import (
    Callable,
)
from typing import (
    Any,
)

import torch

# Track which ops already have fake implementations so we don't re-register.
_registered: set[str] = set()


def _try_register_fake(op_name: str, fn: Callable[..., Any]) -> None:
    """Register a fake implementation, silently skipping if already registered."""
    if op_name in _registered:
        return
    try:
        torch.library.register_fake(op_name)(fn)
        _registered.add(op_name)
    except RuntimeError as e:
        if "already has" in str(e) or "already registered" in str(e):
            # Op already has an implementation (e.g. C++ library loaded).
            _registered.add(op_name)
        else:
            raise


def ensure_fake_registered() -> None:
    """Register fake implementations for all tabulate custom ops.

    Only registers for ops that exist (i.e., the custom op library is loaded).
    Idempotent — safe to call multiple times; already-registered ops are
    skipped via the ``_registered`` set.

    Called automatically at import time and should also be called from any
    code path that needs fake ops after the C++ library has been loaded
    (e.g. the compression entry point).
    """
    if not hasattr(torch.ops, "deepmd"):
        return

    # --- tabulate_fusion_se_a ---
    if hasattr(torch.ops.deepmd, "tabulate_fusion_se_a"):

        def _fake_se_a(
            table: torch.Tensor,
            table_info: torch.Tensor,
            em_x: torch.Tensor,
            em: torch.Tensor,
            last_layer_size: int,
        ) -> list[torch.Tensor]:
            return [table.new_empty([em.size(0), 4, last_layer_size])]

        _try_register_fake("deepmd::tabulate_fusion_se_a", _fake_se_a)

    # --- tabulate_fusion_se_r ---
    if hasattr(torch.ops.deepmd, "tabulate_fusion_se_r"):

        def _fake_se_r(
            table: torch.Tensor,
            table_info: torch.Tensor,
            em: torch.Tensor,
            last_layer_size: int,
        ) -> list[torch.Tensor]:
            return [table.new_empty([em.size(0), em.size(1), last_layer_size])]

        _try_register_fake("deepmd::tabulate_fusion_se_r", _fake_se_r)

    # --- tabulate_fusion_se_t ---
    if hasattr(torch.ops.deepmd, "tabulate_fusion_se_t"):

        def _fake_se_t(
            table: torch.Tensor,
            table_info: torch.Tensor,
            em_x: torch.Tensor,
            em: torch.Tensor,
            last_layer_size: int,
        ) -> list[torch.Tensor]:
            return [table.new_empty([em.size(0), last_layer_size])]

        _try_register_fake("deepmd::tabulate_fusion_se_t", _fake_se_t)

    # --- tabulate_fusion_se_t_tebd ---
    if hasattr(torch.ops.deepmd, "tabulate_fusion_se_t_tebd"):

        def _fake_se_t_tebd(
            table: torch.Tensor,
            table_info: torch.Tensor,
            em_x: torch.Tensor,
            em: torch.Tensor,
            last_layer_size: int,
        ) -> list[torch.Tensor]:
            return [
                table.new_empty([em.size(0), em.size(1), em.size(2), last_layer_size])
            ]

        _try_register_fake("deepmd::tabulate_fusion_se_t_tebd", _fake_se_t_tebd)

    # --- tabulate_fusion_se_atten ---
    if hasattr(torch.ops.deepmd, "tabulate_fusion_se_atten"):

        def _fake_se_atten(
            table: torch.Tensor,
            table_info: torch.Tensor,
            em_x: torch.Tensor,
            em: torch.Tensor,
            two_embed: torch.Tensor,
            last_layer_size: int,
            is_sorted: bool,
        ) -> list[torch.Tensor]:
            return [table.new_empty([em.size(0), 4, last_layer_size])]

        _try_register_fake("deepmd::tabulate_fusion_se_atten", _fake_se_atten)


# Best-effort at import time — ops may not be loaded yet.
ensure_fake_registered()
