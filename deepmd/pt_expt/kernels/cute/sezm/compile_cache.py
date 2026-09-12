# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Device-aware caching for architecture-specific CuTe compilation."""

from __future__ import (
    annotations,
)

from collections.abc import (
    Callable,
)
from contextlib import (
    nullcontext,
)
from functools import (
    lru_cache,
    wraps,
)
from typing import (
    Any,
    TypeVar,
    cast,
)

_T = TypeVar("_T", bound=Callable[..., Any])


def current_cuda_compile_identity() -> tuple[int, int, int]:
    """Return the current CUDA device and its compute capability."""
    import torch

    device_index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    return device_index, major, minor


def device_aware_lru_cache(
    *,
    maxsize: int,
    identity_getter: Callable[[], tuple[int, int, int]] = current_cuda_compile_identity,
) -> Callable[[_T], _T]:
    """Cache a compile factory separately for each CUDA device architecture."""

    def decorate(function: _T) -> _T:
        @lru_cache(maxsize=maxsize)
        def cached(
            identity: tuple[int, int, int],
            args: tuple[Any, ...],
            kwargs: tuple[tuple[str, Any], ...],
        ) -> Any:
            import torch

            device_index = identity[0]
            device_count = getattr(torch.cuda, "device_count", None)
            device_is_visible = device_count is None or device_index < device_count()
            compile_device = (
                torch.cuda.device(device_index)
                if torch.cuda.is_available() and device_is_visible
                else nullcontext()
            )
            with compile_device:
                return function(*args, **dict(kwargs))

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cached(
                identity_getter(),
                args,
                tuple(sorted(kwargs.items())),
            )

        wrapper.cache_clear = cached.cache_clear
        wrapper.cache_info = cached.cache_info
        wrapper.cache_parameters = cached.cache_parameters
        wrapper._deepmd_cute_cached = True
        return cast("_T", wrapper)

    return decorate
