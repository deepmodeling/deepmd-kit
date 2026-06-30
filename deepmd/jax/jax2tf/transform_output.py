# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility exports for TensorFlow output transforms."""

from deepmd.tf2.transform_output import (
    communicate_extended_output,
    get_leading_dims,
)

__all__ = ["communicate_extended_output", "get_leading_dims"]
