# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hardware-accelerated DPA1 (``se_atten``) operators.

This package hosts ``make_fx``-composable Triton implementations of the DPA1
descriptor hot path: the strip / dense environment convolution ``se_conv``
(node-parallel) and the concat / graph environment convolution ``edge_conv``
(edge-parallel). Both are internal to the ``se_atten`` descriptor; the
package-level API exposes the fused operators and the shared availability flag.
"""

from .activation import (
    TRITON_AVAILABLE,
)
from .edge_conv import (
    edge_conv,
)
from .se_conv import (
    se_conv,
)

__all__ = [
    "TRITON_AVAILABLE",
    "edge_conv",
    "se_conv",
]
