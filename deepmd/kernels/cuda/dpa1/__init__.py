# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fused CUDA descriptors for the DPA1 (``se_atten``) graph lower.

:mod:`.graph_descriptor` binds the ``deepmd::dpa1_graph_descriptor`` mega
kernels (environment matrix, three-layer embedding MLP, moment and ``G^T G``).
:mod:`.graph_compress` binds ``deepmd::dpa1_graph_compress`` for the
geo-compressed strip path (quintic table lookup plus the precomputed strip
type-pair gate). Dispatch is gated by :meth:`DescrptDPA1._fused_eligible` at
``DP_CUDA_INFER >= 1``.
"""
