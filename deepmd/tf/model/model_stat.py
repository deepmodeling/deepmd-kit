# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alias for backward compatibility."""

from deepmd.utils.model_stat import (
    _make_all_stat_ref,
)
from deepmd.utils.model_stat import collect_batches as make_stat_input
from deepmd.utils.model_stat import (
    merge_sys_stat,
)

__all__ = [
    "_make_all_stat_ref",  # used by tests
    "make_stat_input",
    "merge_sys_stat",
]
