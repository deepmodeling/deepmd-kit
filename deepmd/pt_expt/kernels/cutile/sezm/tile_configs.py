# SPDX-License-Identifier: LGPL-3.0-or-later
"""Launch-configuration lookup for the shape-tuned cuTile SeZM kernels.

A cuTile kernel exposes two tuning knobs, and both matter enough to be resolved
from a table rather than fixed at a constant:

``tile``
    Edges owned by a block, or -- for a kernel that walks a compressed-sparse-row
    segment -- edges consumed per iteration. The tile width is a compile-time
    constant of the generated source, so each value produces its own cached
    kernel module.
``occupancy``
    Blocks the compiler must fit per multiprocessor. Left to itself it will spend
    the entire shared-memory budget on one block. Zero delegates the choice back
    to the compiler.

The knobs are neither shape-independent nor independent of each other. On the
attention aggregation, moving the backward from a 32-edge to a 16-edge tile at
``occupancy=2`` is worth 21 %, while raising the occupancy of that same 16-edge
tile to four costs a factor of 1.9. Register pressure scales with the degree count
and the focus width, so the optimum moves with ``(lmax, focus_dim)``.

Configurations are resolved through two layers:

1. *Runtime registrations* (:func:`register_tile_configs`), which take precedence
   in the current process. A sweep installs its winners here, so a device without
   built-in coverage can still be tuned before an evaluation run. Registrations
   are process-local by design.
2. *Built-in tables* (:mod:`.tile_config_data`), keyed by an exact GPU name or by
   the longest model-name prefix ending at a space boundary, so edition suffixes
   share one architecture table without confusing names that are prefixes of one
   another.

An unresolved key falls back to the family default. Defaults are correct on any
CUDA device and merely untuned, so a new GPU or a new block layout runs correctly
on first contact and can be swept afterwards.
"""

from __future__ import annotations

import dataclasses
import functools

import torch

__all__ = [
    "TILE_CONFIG_FAMILIES",
    "LaunchConfig",
    "has_tile_config",
    "register_tile_configs",
    "tile_config",
]


@dataclasses.dataclass(frozen=True)
class LaunchConfig:
    """Tile width and occupancy hint of one kernel launch.

    Attributes
    ----------
    tile : int
        Edges per block, or per segment-walk iteration. Must be a power of two.
    occupancy : int
        Blocks per multiprocessor the compiler must accommodate; ``0`` leaves the
        choice to the compiler.
    """

    tile: int
    occupancy: int = 0

    @property
    def hints(self) -> dict[str, int]:
        """Return the keyword hints to apply to the kernel."""
        return {"occupancy": self.occupancy} if self.occupancy else {}


#: One family per launchable kernel. Forward and backward tune independently:
#: they differ in grid shape, in live-tile count and often in traversal order.
TILE_CONFIG_FAMILIES = (
    "rotate_mix_fwd",
    "rotate_mix_bwd",
    "mixing_stack_fwd",
    "mixing_stack_bwd",
    "flash_fwd",
    "flash_bwd",
    "wigner_monomials",
    "force_assembly",
)

#: Family defaults. Each is a modest tile at an occupancy of two, the setting that
#: never collapsed on any shape measured: the compiler's own choice regularly
#: over-allocates shared memory, and higher occupancies spill. The two
#: shape-independent families are latency bound rather than register bound and are
#: left to the compiler.
_DEFAULTS: dict[str, LaunchConfig] = {
    "rotate_mix_fwd": LaunchConfig(tile=64, occupancy=2),
    "rotate_mix_bwd": LaunchConfig(tile=8, occupancy=2),
    "mixing_stack_fwd": LaunchConfig(tile=32, occupancy=2),
    "mixing_stack_bwd": LaunchConfig(tile=32, occupancy=2),
    "flash_fwd": LaunchConfig(tile=32, occupancy=2),
    "flash_bwd": LaunchConfig(tile=16, occupancy=2),
    "wigner_monomials": LaunchConfig(tile=256),
    "force_assembly": LaunchConfig(tile=16),
}

# Runtime registrations, highest lookup precedence.
_RUNTIME: dict[str, dict[tuple[int, ...], LaunchConfig | None]] = {
    family: {} for family in TILE_CONFIG_FAMILIES
}


def _match_builtin_tables(
    device_name: str,
) -> dict[str, dict[tuple[int, ...], LaunchConfig | None]]:
    """Return the exact or longest whole-token-prefix table for a GPU name."""
    from .tile_config_data import (
        BUILTIN_TILE_CONFIGS,
    )

    if device_name in BUILTIN_TILE_CONFIGS:
        return BUILTIN_TILE_CONFIGS[device_name]
    prefixes = [
        model_name
        for model_name in BUILTIN_TILE_CONFIGS
        if device_name.startswith(f"{model_name} ")
    ]
    if not prefixes:
        return {}
    return BUILTIN_TILE_CONFIGS[max(prefixes, key=len)]


@functools.cache
def _builtin_tables_for_device(
    device_index: int,
) -> dict[str, dict[tuple[int, ...], LaunchConfig | None]]:
    """Resolve and cache the built-in tables for one CUDA device index."""
    return _match_builtin_tables(torch.cuda.get_device_name(device_index))


def _lookup(family: str, key: tuple[int, ...]) -> LaunchConfig | None:
    """Resolve ``key`` through the runtime and built-in layers.

    A ``None`` result folds together an explicit ``None`` entry (the sweep ran and
    the family default is the measured optimum) and an absent key (never swept on
    this GPU): the caller behaves identically in both cases.
    """
    if family not in _RUNTIME:
        raise KeyError(f"unknown tile-config family {family!r}")
    runtime = _RUNTIME[family]
    if key in runtime:
        return runtime[key]
    if not torch.cuda.is_available():
        return None
    tables = _builtin_tables_for_device(torch.cuda.current_device())
    return tables.get(family, {}).get(key)


def tile_config(family: str, key: tuple[int, ...] = ()) -> LaunchConfig:
    """Return the launch configuration of one kernel on the running GPU.

    Parameters
    ----------
    family : str
        A member of :data:`TILE_CONFIG_FAMILIES`.
    key : tuple[int, ...]
        Shape key: ``(lmax, focus_dim)`` for the SO(2) kernels, empty for the
        shape-independent ones.

    Returns
    -------
    LaunchConfig
        The registered entry, else the built-in entry for this GPU, else the
        family default.

    Raises
    ------
    KeyError
        If ``family`` is not a known family.
    """
    return _lookup(family, key) or _DEFAULTS[family]


def has_tile_config(family: str, key: tuple[int, ...] = ()) -> bool:
    """Return whether ``key`` was ever swept for ``family`` on this GPU."""
    if family not in _RUNTIME:
        raise KeyError(f"unknown tile-config family {family!r}")
    if key in _RUNTIME[family]:
        return True
    if not torch.cuda.is_available():
        return False
    tables = _builtin_tables_for_device(torch.cuda.current_device())
    return key in tables.get(family, {})


def register_tile_configs(
    family: str, key: tuple[int, ...], config: LaunchConfig | None
) -> None:
    """Install a launch configuration for the current process.

    Parameters
    ----------
    family : str
        A member of :data:`TILE_CONFIG_FAMILIES`.
    key : tuple[int, ...]
        Shape key the configuration applies to.
    config : LaunchConfig or None
        The configuration to install, or ``None`` to record that the family
        default is the measured optimum.

    Raises
    ------
    KeyError
        If ``family`` is not a known family.
    ValueError
        If the tile width is not a power of two.
    """
    if family not in _RUNTIME:
        raise KeyError(f"unknown tile-config family {family!r}")
    if config is not None and (config.tile <= 0 or config.tile & (config.tile - 1)):
        raise ValueError(f"tile width must be a power of two, got {config.tile}")
    _RUNTIME[family][key] = config
