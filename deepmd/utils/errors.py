# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd_utils.utils.errors import (
    OutOfMemoryError,
)


class GraphTooLargeError(Exception):
    """The graph is too large, exceeding protobuf's hard limit of 2GB."""


class GraphWithoutTensorError(Exception):
    pass


__all__ = [
    "OutOfMemoryError",
    "GraphTooLargeError",
    "GraphWithoutTensorError",
]
