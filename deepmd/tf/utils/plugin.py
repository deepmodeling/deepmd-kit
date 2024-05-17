# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alias for backward compatibility."""

from __future__ import (
    annotations,
)

from deepmd.utils.plugin import (
    Plugin,
    PluginVariant,
    VariantABCMeta,
    VariantMeta,
)

__all__ = [
    "Plugin",
    "VariantMeta",
    "VariantABCMeta",
    "PluginVariant",
]
