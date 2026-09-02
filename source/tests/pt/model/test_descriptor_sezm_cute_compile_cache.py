# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Behavioral tests for device-aware CuTe compile caching."""

from __future__ import (
    annotations,
)

import importlib.util
import sys
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from unittest import (
    mock,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_PATH = REPO_ROOT / "deepmd/pt_expt/kernels/cute/sezm/compile_cache.py"


def _load_cache_module():
    assert CACHE_PATH.is_file(), f"CuTe compile cache is missing: {CACHE_PATH}"
    name = "sezm_cute_compile_cache_test"
    spec = importlib.util.spec_from_file_location(name, CACHE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_cache_separates_device_and_compute_capability() -> None:
    cache_module = _load_cache_module()
    identity = [(0, 8, 0)]
    calls: list[tuple[str, tuple[int, int, int]]] = []

    @cache_module.device_aware_lru_cache(
        maxsize=8,
        identity_getter=lambda: identity[0],
    )
    def compile_kernel(mode: str):
        calls.append((mode, identity[0]))
        return object()

    sm80_first = compile_kernel("strict-fp32")
    assert compile_kernel("strict-fp32") is sm80_first

    identity[0] = (1, 9, 0)
    sm90 = compile_kernel("strict-fp32")
    assert sm90 is not sm80_first

    identity[0] = (0, 8, 0)
    assert compile_kernel("strict-fp32") is sm80_first
    assert calls == [
        ("strict-fp32", (0, 8, 0)),
        ("strict-fp32", (1, 9, 0)),
    ]


def test_cache_exposes_standard_cache_controls() -> None:
    cache_module = _load_cache_module()

    @cache_module.device_aware_lru_cache(
        maxsize=2,
        identity_getter=lambda: (0, 8, 6),
    )
    def compile_kernel(rows: int):
        return object()

    compile_kernel(4)
    assert compile_kernel.cache_info().currsize == 1
    compile_kernel.cache_clear()
    assert compile_kernel.cache_info().currsize == 0


def test_cache_compiles_inside_the_keyed_cuda_device() -> None:
    cache_module = _load_cache_module()
    entered: list[tuple[str, int]] = []

    class DeviceContext:
        def __init__(self, index: int) -> None:
            self.index = index

        def __enter__(self) -> None:
            entered.append(("enter", self.index))

        def __exit__(self, *_args) -> None:
            entered.append(("exit", self.index))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device=lambda index: DeviceContext(index),
        )
    )

    @cache_module.device_aware_lru_cache(
        maxsize=2,
        identity_getter=lambda: (3, 9, 0),
    )
    def compile_kernel() -> object:
        entered.append(("compile", 3))
        return object()

    with mock.patch.dict(sys.modules, {"torch": fake_torch}):
        compile_kernel()

    assert entered == [("enter", 3), ("compile", 3), ("exit", 3)]
