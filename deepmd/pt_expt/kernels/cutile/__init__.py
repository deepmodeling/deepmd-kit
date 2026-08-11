# SPDX-License-Identifier: LGPL-3.0-or-later
"""cuTile inference kernels for the SeZM / DPA4 descriptor.

The package provides a complete, self-contained inference path written in the
``cuda.tile`` DSL. It is selected by ``DP_CUTILE_INFER`` and is mutually
exclusive with the Triton and CuTe paths: when it is enabled, no Triton kernel
executes, and a convolution whose layout it does not support falls back to the
dense reference rather than to another accelerated backend.

See :mod:`deepmd.pt_expt.kernels.cutile.common` for the properties of the tile
model that shape every kernel here, and ``doc/outisli/dpa4_cutile.md`` for the
measurements behind the design.
"""

from __future__ import annotations

from .common import (
    CUTILE_AVAILABLE,
)

__all__ = ["CUTILE_AVAILABLE"]
