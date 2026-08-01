# SPDX-License-Identifier: LGPL-3.0-or-later
"""Hand-written CUDA / cuBLAS operators for graph-lower inference.

The CUDA sources live under ``source/op/pt`` and compile into
``libdeepmd_op_pt.so`` (loaded via ``deepmd.pt.cxx_op``); the modules here
expose the resulting ``torch.ops.deepmd.*`` operators to the pt_expt graph
lower together with the backward, meta (fake) and CPU trace-time
implementations that ``torch.export`` / ``make_fx`` require. Dispatch is
gated by ``DP_CUDA_INFER`` (:func:`deepmd.kernels.utils.cuda_infer_level`).

Modules
-------
:mod:`.dpa1.graph_descriptor`
    DPA1 (``se_atten``) descriptor mega kernels: environment matrix,
    embedding MLP, moment reduction and ``G^T G`` contraction in one forward
    / one backward kernel.
:mod:`.graph_fitting`
    Descriptor-agnostic fused energy fitting network on the flat node axis
    (cuBLAS GEMMs with fused bias / activation / timestep / residual
    epilogues).
:mod:`.edge_force_virial`
    Descriptor-agnostic force / atom-virial / per-frame-virial assembly from
    the per-edge energy gradient.
"""
