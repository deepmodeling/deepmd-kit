# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compatibility wrapper for the TF2 SavedModel exporter."""

from deepmd.tf2.utils.serialization import (
    deserialize_to_file,
)

__all__ = ["deserialize_to_file"]
