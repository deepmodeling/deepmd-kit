# SPDX-License-Identifier: LGPL-3.0-or-later
class GraphTooLargeError(Exception):
    """The graph is too large, exceeding protobuf's hard limit of 2GB."""


class GraphWithoutTensorError(Exception):
    pass


class OutOfMemoryError(Exception):
    """This error is caused by out-of-memory (OOM)."""
