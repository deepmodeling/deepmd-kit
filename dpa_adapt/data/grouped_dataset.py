# SPDX-License-Identifier: LGPL-3.0-or-later
"""Deprecated location. Moved to :mod:`dpa_adapt.grouped` (impl in ``_offline``).

Kept as a re-export shim; use ``from dpa_adapt.grouped import GroupedDataset`` in
new code.  Note: patch ``dpa_adapt.grouped._offline.load_or_extract`` (where the
name is resolved), not this shim.
"""

from dpa_adapt.grouped._offline import (  # noqa: F401
    GroupedDataset,
    has_grouped_markers,
)
