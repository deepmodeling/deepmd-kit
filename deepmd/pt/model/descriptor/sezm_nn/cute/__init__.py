# SPDX-License-Identifier: LGPL-3.0-or-later
"""
CuTe-DSL accelerated SO(2) rotation operators for SeZM / DPA4.

This package provides a self-contained, ``torch.compile``-friendly implementation
of the two fused gather + batched-GEMM operators used by the SeZM SO(2) edge
convolution:

* ``rotate_to_local``  : ``out[e] = wigner[e][coeff_index] @ x[src[e]]``
* ``rotate_back``      : ``out[e] = wigner[e][:, coeff_index] @ x_local[e]``

The kernels are written with the NVIDIA CuTe DSL (``cutlass.cute``) and fuse the
Wigner-row/column gather and the source-node gather directly into the matmul, so
the large ``D_to_m`` / ``x_src`` intermediates are never materialized. They are
exposed through the modern ``torch.library.custom_op`` API (functional, with
``register_fake`` + ``register_autograd``) so that they compose correctly with
``torch.compile`` and autograd.

The top-level entry points are re-exported here for convenience.
"""

from __future__ import (
    annotations,
)

from .so2_rotation import (
    SEZM_CUTE_AVAILABLE,
    rotate_back_cute,
    rotate_to_local_cute,
)

__all__ = [
    "SEZM_CUTE_AVAILABLE",
    "rotate_back_cute",
    "rotate_to_local_cute",
]
