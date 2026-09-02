# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hand-written CUDA operators for the uncompressed DPA4 / SeZM descriptor.

Modules
-------
:mod:`.so2_conv`
    The fused SO(2) convolution value path: one operator for the Wigner
    rotation, the radial degree mixer, the gated mixing stack, the inverse
    rotation, the attention weighting and the destination reduction, plus its
    analytic backward.
:mod:`.wigner_dense`
    The fused dense Wigner build: the packed block-diagonal pair
    ``(D_full, Dt_full)`` evaluated from edge quaternions as fitted sparse
    polynomials in one kernel.
:mod:`.grid_pair`
    The fused SO(3) grid pair product ``from_grid(to_grid(a) * to_grid(b))``
    with the grid field kept in registers.
:mod:`.zonal_scatter`
    The fused geometric initial embedding: the per-edge message built in
    registers and reduced through the destination CSR.
:mod:`.edge_radial`
    The fused cutoff envelope and radial basis of the pair distance.
"""

from .so2_conv import (
    SO2ConvCuda,
    make_cuda_so2_conv,
    op_available,
)

__all__ = [
    "SO2ConvCuda",
    "make_cuda_so2_conv",
    "op_available",
]
