# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201
"""Shared last-layer activation primitives for the DPA1 (``se_atten``) Triton
inference kernels.

Both fused environment convolutions -- the strip / dense ``se_conv``
(node-parallel) and the concat / graph ``edge_conv`` (edge-parallel) -- inline
the embedding net's last-layer activation for value and derivative. The
activation-code map, the ``triton`` availability flag and the two ``@triton.jit``
helpers live here so neither kernel module depends on the other.
"""

from __future__ import (
    annotations,
)

# Last-layer activations the fused kernels inline, mapped to the ``ACT`` code
# consumed by the Triton bodies (forward value and backward derivative). Only
# these activations are eligible for a fused path; any other keeps the dense
# reference. Both are smooth (differentiable forces), unlike relu / relu6.
ACT_CODES: dict[str, int] = {"tanh": 0, "silu": 1}

try:
    import triton
    import triton.language as tl
    from triton.language.extra import (
        libdevice,
    )

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def activation(z, ACT: tl.constexpr):
        """Last-layer activation value; ``ACT`` selects ``tanh`` (0) or ``silu`` (1)."""
        if ACT == 0:
            a = libdevice.tanh(z)
        else:
            a = z * tl.sigmoid(z)
        return a

    @triton.jit
    def activation_grad(z, ACT: tl.constexpr):
        """Activation value and its derivative for ``ACT`` in {``tanh``, ``silu``}.

        For ``tanh`` the derivative is ``1 - tanh(z)^2``; for ``silu`` it is
        ``sigmoid(z) * (1 + z * (1 - sigmoid(z)))``.
        """
        if ACT == 0:
            a = libdevice.tanh(z)
            ad = 1.0 - a * a
        else:
            s = tl.sigmoid(z)
            a = z * s
            ad = s * (1.0 + z * (1.0 - s))
        return a, ad
