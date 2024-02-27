# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alias for backward compatibility."""
from deepmd.utils.model_stat import (
    _make_all_stat_ref,
    make_stat_input,
    merge_sys_stat,
)

__all__ = [
    "make_stat_input",
    "merge_sys_stat",
    "_make_all_stat_ref",  # used by tests
]
