# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deprecated location. Moved to :mod:`dpa_adapt.grouped` (impl in ``_polymer``).

Kept as a re-export shim; use ``from dpa_adapt import PolymerBuilder`` or
``Grouped.from_polymer_csv(...)`` in new code.
"""

from dpa_adapt.grouped._polymer import PolymerBuilder  # noqa: F401
