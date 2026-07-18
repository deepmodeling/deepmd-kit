# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt SO(2) linear, convolution, and radial mixer with opt-in fused Triton kernels.

The dpmodel SO(2) modules are array-API only.  These wrappers inject the
reference pt opt-in Triton inference path around three hot paths, mirroring
``deepmd.pt.model.descriptor.sezm_nn.so2``:

- the block-diagonal GEMM of :class:`SO2Linear`,
- the two rotation hot paths of :class:`SO2Convolution`, and
- the low-rank branch of :class:`DynamicRadialDegreeMixer`.

The kernels are sourced from the central :mod:`deepmd.kernels.triton.sezm`
package and gated by the integer inference level ``DP_TRITON_INFER`` (see
:func:`deepmd.kernels.utils.triton_infer_level`); every kernel path requires
level ``>= 1``.  The kernels run only during inference (``not self.training``),
and each kernel self-guards Triton availability and falls back to an eager
reference off CUDA / on fp64, so importing this module is safe on CPU-only
environments; training and CPU / fp64 inference use the dpmodel dense path.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from deepmd.dpmodel.common import (
    get_xp_precision,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import (
    DynamicRadialDegreeMixer as DynamicRadialDegreeMixerDP,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as SO2ConvolutionDP
from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as SO2LinearDP
from deepmd.kernels.utils import (
    triton_infer_level,
    use_cute_infer,
)
from deepmd.pt_expt.common import (
    torch_module,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeFeatureCache,
    )


@torch_module
class SO2Linear(SO2LinearDP):
    """SO(2)-equivariant linear with an opt-in fused block-diagonal Triton GEMM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Inference fast path (``DP_TRITON_INFER >= 1``): the per-|m|-block
        # batched bmm + cat of ``_block_diagonal_matmul`` is replaced by a fused
        # Triton BN=64 block-diagonal GEMM that consumes the strided operands
        # without a contiguity copy. Bound only when Triton is available and every
        # block width aligns to BN=64; otherwise the eager path is kept. The gate
        # is read once at construction so it is a compile-time constant in the
        # traced (``make_fx``) graph, and it only takes effect during inference.
        self._block_diag_gemm = None
        if triton_infer_level() >= 1:
            from deepmd.kernels.triton.sezm.so2_block_gemm import (
                SO2_BLOCK_GEMM_TRITON_AVAILABLE,
                block_diag_gemm,
                slices_supported,
            )

            if SO2_BLOCK_GEMM_TRITON_AVAILABLE and slices_supported(
                self._block_diag_slices
            ):
                self._block_diag_gemm = block_diag_gemm

    def _block_diagonal_matmul(
        self, x_flat: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        if self._block_diag_gemm is not None and not self.training:
            # The fused GEMM consumes the ``(F, D_m*Cin, D_m*Cout)`` presentation
            # directly from the strided weight, so the permute is applied here and
            # the contiguity copy the dpmodel ``bmm`` cat path would need is
            # skipped. The eager fallback permutes ``weight`` internally, so it is
            # passed the stored ``(D_m*Cin, F, D_m*Cout)`` layout untouched.
            weight = weight.permute(1, 0, 2)  # (F, D_m*Cin, D_m*Cout)
            return self._block_diag_gemm(x_flat, weight, self._block_diag_slices)
        return super()._block_diagonal_matmul(x_flat, weight)


@torch_module
class DynamicRadialDegreeMixer(DynamicRadialDegreeMixerDP):
    """Dynamic radial degree mixer with an opt-in fused Triton low-rank branch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Inference fast path (``DP_TRITON_INFER >= 1``): a fused Triton kernel
        # replaces the dense scatter and the tiny batched matmul of the
        # ``degree_channel`` low-rank branch in the ``mmax == 1`` layout. The gate
        # is read once at construction so it is a compile-time constant in the
        # traced (``make_fx``) graph, and it only takes effect during inference.
        self.use_triton_infer = triton_infer_level() >= 1
        self._radial_mix_block = None
        if (
            self.use_triton_infer
            and self.mode == "degree_channel"
            and self.rank > 0
            and self.mmax == 1
        ):
            from deepmd.kernels.triton.sezm.radial_mix import (
                radial_mix_block,
            )

            self._radial_mix_block = radial_mix_block

    def _mix_rank_compact(
        self, compact: torch.Tensor, x_local: torch.Tensor
    ) -> torch.Tensor:
        if self._radial_mix_block is not None and not self.training:
            return self._radial_mix_block(
                compact, x_local, self.channel_basis, self.lmax
            )
        return super()._mix_rank_compact(compact, x_local)


@torch_module
class SO2Convolution(SO2ConvolutionDP):
    """SO(2) convolution with opt-in fused Triton rotation kernels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # ``use_triton_infer`` is read once at construction so it is a
        # compile-time constant in the traced (``make_fx``) graph, and it only
        # takes effect during inference.
        self.use_triton_infer = triton_infer_level() >= 1

        # === Triton rotation kernels: block for mmax == 1, dense otherwise ===
        self._rotate_to_local_fn = None
        self._rotate_back_fn = None
        if self.use_triton_infer:
            from deepmd.kernels.triton.sezm.so2_rotation import (
                rotate_back_block_so2,
                rotate_back_dense,
                rotate_to_local_block,
                rotate_to_local_dense,
            )

            if self.mmax == 1:
                self._rotate_to_local_fn = lambda x, src, wigner: rotate_to_local_block(
                    x, src, wigner, self.lmax
                )
                # The block kernel reads the (E, F, D_m, Cf) focus layout directly,
                # so the rotate-back path passes ``x_local`` before the global
                # reshape and the transpose-back copy is skipped.
                self._rotate_back_fn = lambda x_local, wigner: rotate_back_block_so2(
                    x_local, wigner, self.lmax
                )
            else:
                self._rotate_to_local_fn = lambda x, src, wigner: rotate_to_local_dense(
                    x, src, wigner, self.coeff_index_m, self.ebed_dim_full
                )
                self._rotate_back_fn = lambda x_local, wigner: rotate_back_dense(
                    x_local, wigner, self.coeff_index_m, self.ebed_dim_full
                )

        # === Step 12. Optional fused flash-attention aggregation kernel ===
        # Folds the entire ``n_atten_head > 0`` value aggregation -- block-diagonal
        # rotate-back, inverse-rotation rescale, envelope-gated softmax weighting,
        # and the destination scatter -- into a single destination-segmented
        # Triton kernel, removing the transient ``x_message`` and weighted-value
        # edge tensors and the ``index_add`` round trip. It shares the
        # ``DP_TRITON_INFER`` gate with the other SeZM inference kernels and only
        # engages for the supported ``mmax == 1`` attention layout without the
        # optional focus-mix / value / output projections (the deployed DPA4
        # configuration); the op itself dispatches to an eager reference off the
        # CUDA fp32 path. The output-side head gate stays a cheap node-level
        # elementwise applied after the kernel. The supported-layout half of the
        # predicate is the dpmodel base's ``_flash_atten_layout_ok`` (the base
        # leaves ``use_flash_atten=False`` and the hooks ``None``); this re-enables
        # flash by ANDing that layout predicate with the Triton-availability gate.
        self.use_flash_atten = self.use_triton_infer and self._flash_atten_layout_ok
        if self.use_flash_atten:
            from deepmd.kernels.triton.sezm.flash_atten import (
                build_row_ptr,
                flash_atten_aggregate,
            )

            self._flash_atten_fn = flash_atten_aggregate
            self._build_row_ptr_fn = build_row_ptr

        # The rotate/flash gate above exposes only the boolean ``use_triton_infer``;
        # the fused value-path operator additionally reads the raw integer level
        # (it selects the level-3 fp16x3 mixing stack from ``self.triton_infer_level``),
        # so the level is stored here as well. ``DP_TRITON_INFER`` and
        # ``DP_CUTE_INFER`` both claim the single ``so2_message`` value path, so
        # enabling them together has no coherent meaning and is rejected here.
        self.triton_infer_level = triton_infer_level()
        if self.triton_infer_level >= 1 and use_cute_infer():
            raise ValueError(
                "DP_TRITON_INFER and DP_CUTE_INFER are mutually exclusive: both "
                "select the fused SO(2) value-path backend. Enable exactly one "
                "of them."
            )

        # === Step 13. Optional fused Triton SO(2) value-path operators ===
        # Fuses rotate-to-local, the radial degree mixing, the gated mixing
        # stack, and the focus competition of ``so2_message`` into the
        # ``sezm_triton::so2_rotate_mix`` / ``so2_mixing_stack`` operators.
        # The factory validates the block layout (``mmax == 1``, gated stack
        # with an identity final layer, supported focus widths) and returns
        # ``None`` otherwise, leaving the reference path in charge. The value
        # path resolves its launch configurations from the swept tables, so
        # it engages at ``DP_TRITON_INFER >= 2``; at level 3 the factory
        # additionally routes the mixing stack through the fp16x3 tensor-core
        # operator on shapes whose configuration passed the fp64 validation
        # sweep.
        if self.triton_infer_level >= 2:
            from deepmd.kernels.triton.sezm.so2_value_path import (
                make_triton_value_path,
            )

            self._value_path = make_triton_value_path(self)
        # === Step 14. Optional fused CuTe SO(2) value-path operator ===
        # Experimental alternative backend; mutually exclusive with the Triton
        # flag (enforced above).
        elif use_cute_infer():
            from deepmd.kernels.cute.sezm import (
                make_cute_value_path,
            )

            self._value_path = make_cute_value_path(self)

    def _rotate_to_local(
        self, x: torch.Tensor, edge_cache: EdgeFeatureCache
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_triton_infer and not self.training:
            # ``self._rotate_to_local_fn`` was bound in ``__init__`` (the block
            # kernel for the m-major ``mmax == 1`` layout, dense otherwise).
            D_full = edge_cache.D_full
            x_local = self._rotate_to_local_fn(x, edge_cache.src, D_full)
            x_dst_local: torch.Tensor | None = None
            if self.node_wise_grid_product is not None:
                x_dst_local = self._rotate_to_local_fn(x, edge_cache.dst, D_full)
            return x_local, x_dst_local
        return super()._rotate_to_local(x, edge_cache)

    def _rotate_back(
        self, x_local: torch.Tensor, edge_cache: EdgeFeatureCache, n_edge: int
    ) -> torch.Tensor:
        if self.use_triton_infer and not self.training:
            Dt_full = edge_cache.Dt_full
            if self.mmax == 1:
                # The block kernel consumes the (E, F, D_m, Cf) focus layout in
                # place, folding the inverse transpose into its channel addressing.
                return self._rotate_back_fn(x_local, Dt_full)
            # Restore reduced global layout (E, D_m, C_wide) for the dense kernel.
            x_std = (
                x_local.transpose(1, 2)
                .contiguous()
                .reshape(n_edge, self.reduced_dim, self.hidden_channels)
            )
            return self._rotate_back_fn(x_std, Dt_full)
        return super()._rotate_back(x_local, edge_cache, n_edge)

    def _flash_aggregate(
        self,
        x_local_flash: torch.Tensor,
        edge_cache: EdgeFeatureCache,
        attn_alpha: torch.Tensor,
        x_l0_node: torch.Tensor,
        n_node: int,
        compute_dtype: Any,
    ) -> torch.Tensor:
        # === Step 4.3f. Fused rotate-back + envelope-softmax-weighted
        # segment scatter. One destination-segmented Triton kernel
        # folds the block-diagonal rotate-back, the inverse-rotation
        # rescale, the per-edge ``attn_alpha`` weighting, and the
        # destination reduction into a single atomic-free pass,
        # returning the ungated aggregate ``(N, D, C_wide)``. The
        # transient rotate-back message and weighted value tensors are
        # never materialized.
        row_ptr = self._build_row_ptr_fn(edge_cache.dst, n_node)
        pre_gate = self._flash_atten_fn(
            x_local_flash,
            edge_cache.Dt_full,
            self.rotate_inv_rescale_full,
            attn_alpha,
            row_ptr,
            edge_cache.dst,
            self.lmax,
            self.n_atten_head,
        )  # (N, D, C_wide)

        # === Step 4.4f. Output-side head gate (cheap node-level) ===
        attn_output_gate = torch.sigmoid(
            torch.einsum(
                "nfi,ifo->nfo",
                self.attn_output_gate_norm(x_l0_node.to(dtype=compute_dtype)),
                self.adamw_attn_gate_w,
            )
        )  # (N, Fa, H)
        # Broadcast the per-(focus, head) gate over the head channels
        # to the packed hidden width ``c = f * Cf + h * head_dim + ch``.
        gate_full = (
            attn_output_gate.reshape(n_node, self.attn_n_focus, self.n_atten_head, 1)
            .expand(
                n_node,
                self.attn_n_focus,
                self.n_atten_head,
                self.head_dim,
            )
            .reshape(n_node, self.hidden_channels)
        )  # (N, C_wide)
        # dpmodel exposes the output precision as the string ``self.precision`` (the
        # wrapped conv has no ``self.dtype``); ``get_xp_precision`` resolves it to
        # the torch dtype the dpmodel dense branch casts to, so the fused and dense
        # aggregates share the same storage precision.
        out = (pre_gate * gate_full.unsqueeze(1)).to(
            dtype=get_xp_precision(torch, self.precision)
        )
        return out  # (N, D, C_wide)
