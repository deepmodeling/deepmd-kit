# SPDX-License-Identifier: LGPL-3.0-or-later
"""Alias for backward compatibility."""

from deepmd.utils.compat import (
    convert_input_v0_v1,
    convert_input_v1_v2,
    convert_optimizer_to_new_format,
    deprecate_numb_test,
    update_deepmd_input,
)

__all__ = [
    "convert_input_v0_v1",
    "convert_input_v1_v2",
    "convert_optimizer_to_new_format",
    "deprecate_numb_test",
    "update_deepmd_input",
]
