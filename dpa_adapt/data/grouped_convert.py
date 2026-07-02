# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deprecated location. Moved to :mod:`dpa_adapt.grouped` (impl in ``_convert``).

Kept as a re-export shim; use ``from dpa_adapt import add_group_markers`` or
``Grouped.mark_existing(...)`` in new code.
"""

from dpa_adapt.grouped._convert import (  # noqa: F401
    GROUP_ID_KEY,
    POOL_MASK_KEY,
    REAL_ATYPE_KEY,
    WEIGHT_KEY,
    GroupMarkerResult,
    add_group_markers,
)
