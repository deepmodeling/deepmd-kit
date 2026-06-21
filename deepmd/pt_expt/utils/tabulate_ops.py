# SPDX-License-Identifier: LGPL-3.0-or-later
"""Register fake tensor implementations for deepmd custom tabulate ops.

These registrations enable torch.export and make_fx to trace through the
compressed forward path, which uses C++ custom ops (tabulate_fusion_se_*).
Without fake implementations, torch.export cannot determine output shapes.

`ensure_fake_registered()` is called explicitly (and idempotently) by the paths
that need fake ops — e.g. the compression entry point — after the C++ custom op
library has been loaded. It is deliberately NOT called at package import time:
doing so would pull custom-op registration onto the plain pt (torch.jit)
inference path (which imports this package only for the vesin neighbor list) and
crash `dp test` when the C++ op library is absent, because the pt descriptor
fallbacks monkeypatch a plain Python function onto ``torch.ops.deepmd`` and
``register_fake`` then raises "operator does not exist".

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


def _op_exists(name: str) -> bool:
    """Whether ``deepmd::<name>`` is a real (C++-registered) dispatcher op.

    A bare ``hasattr(torch.ops.deepmd, name)`` is not sufficient: when the C++
    custom-op library is absent, the pt descriptor fallbacks monkeypatch a plain
    Python function onto the ``torch.ops.deepmd`` namespace (see e.g.
    ``deepmd/pt/model/descriptor/se_a.py``). That makes ``hasattr`` return True
    while ``register_fake`` still raises "operator does not exist". Only a real
    op resolves to an ``OpOverloadPacket``.
    """
    op = getattr(torch.ops.deepmd, name, None)
    return isinstance(op, torch._ops.OpOverloadPacket)


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

    Only registers for ops that are actually loaded as real dispatcher ops
    (i.e., the C++ custom op library is present). Idempotent — safe to call
    multiple times; already-registered ops are skipped via the ``_registered``
    set. Not called at import time: the paths that need fake ops (e.g. the
    compression entry point) call this explicitly after the C++ library loads,
    so that plain pt inference never triggers custom-op registration.
    """
    # --- tabulate_fusion_se_a ---
    if _op_exists("tabulate_fusion_se_a"):

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
    if _op_exists("tabulate_fusion_se_r"):

        def _fake_se_r(
            table: torch.Tensor,
            table_info: torch.Tensor,
            em: torch.Tensor,
            last_layer_size: int,
        ) -> list[torch.Tensor]:
            return [table.new_empty([em.size(0), em.size(1), last_layer_size])]

        _try_register_fake("deepmd::tabulate_fusion_se_r", _fake_se_r)

    # --- tabulate_fusion_se_t ---
    if _op_exists("tabulate_fusion_se_t"):

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
    if _op_exists("tabulate_fusion_se_t_tebd"):

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
    if _op_exists("tabulate_fusion_se_atten"):

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
