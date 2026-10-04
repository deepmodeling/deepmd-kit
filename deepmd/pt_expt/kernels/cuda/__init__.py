# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hand-written CUDA / cuBLAS operators for graph-lower inference.

The CUDA sources live under ``source/op/pt`` and compile into
``libdeepmd_op_pt.so``; the modules here expose the resulting
``torch.ops.deepmd.*`` operators to the pt_expt graph lower together with the
backward, meta (fake) and CPU trace-time implementations that
``torch.export`` / ``make_fx`` require. Dispatch is gated by ``DP_CUDA_INFER``
(:func:`deepmd.pt_expt.kernels.utils.cuda_infer_level`).

This package holds the operators that exist only on CUDA. An operator whose
Python front end serves more than one device lives one level up, beside
:mod:`deepmd.pt_expt.kernels.utils`.

Modules
-------
:mod:`.dpa1.graph_descriptor`
    DPA1 (``se_atten``) descriptor mega kernels: environment matrix,
    embedding MLP, moment reduction and ``G^T G`` contraction in one forward
    / one backward kernel.
:mod:`.dpa4`
    DPA4 (``sezm``) operators: the fused SO(2) convolution, the SO(3) grid
    pair product, the geometric initial embedding, the Wigner-D tables and
    the fused cutoff envelope with radial basis.
"""
