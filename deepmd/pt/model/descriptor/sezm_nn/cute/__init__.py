# SPDX-License-Identifier: LGPL-3.0-or-later
"""
CuTe-DSL fused SO(2) value-path operator for SeZM / DPA4.

This package hosts a single bucketed CuTe operator that folds the entire per-edge
value path of :class:`~deepmd.pt.model.descriptor.sezm_nn.so2.SO2Convolution`
(``rotate_to_local`` -> radial degree mix -> the three-layer gated SO(2) mixing
stack -> focus competition) into a fused forward kernel and a matching
recompute backward kernel, keeping the per-edge intermediates on chip. It is an
opt-in inference path enabled by ``DP_CUTE_INFER``; the final local features are
handed to the committed flash-attention aggregation for rotate-back and scatter.
Kernel entry points are internal implementation details of the SeZM descriptor;
the package-level API only exposes availability and the value-path factory.

Current limitations
-------------------
Performance
    On H20 / fp32 the operator is about 2.8x slower than the compiled Triton +
    flash-attention path (roughly 489 / 724 ms versus 174 / 262 ms per force step
    at 2000 / 4000 atoms). Peak memory is at parity with, or marginally below,
    the compiled path (about 0.5 / 0.8 GB lower) and roughly 1.68x below the
    eager path. The bottleneck is the recompute backward, which dominates the
    kernel time: its occupancy is capped by the block-diagonal weight held
    resident in shared memory, and both the forward and backward GEMMs run at the
    hand-written plateau of about 21% of fp32 peak (versus about 52% for cuBLAS).

Deployment
    This is a Python-inference-only path. The ``cutlass.cute`` kernels are
    nvcc / NVRTC JIT-compiled at runtime and do not bake into the AOTInductor
    ``.pt2`` artifact, so the operator is unavailable to the LAMMPS / GPUMD C++
    inference path. ``DP_CUTE_INFER`` is an independent path from
    ``DP_TRITON_INFER``; it engages regardless of the Triton flag and reuses the
    committed flash-attention aggregation kernel when it is active.

Correctness
    The force is bit-exact against the eager reference (energy relative error
    about 1e-9, force relative error about 5e-7 in fp32).
"""

from __future__ import (
    annotations,
)

from .forward import (
    SEZM_CUTE_AVAILABLE,
)
from .operator import (
    make_cute_value_path,
)

__all__ = [
    "SEZM_CUTE_AVAILABLE",
    "make_cute_value_path",
]
