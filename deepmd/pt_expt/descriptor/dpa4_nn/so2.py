# SPDX-License-Identifier: LGPL-3.0-or-later
"""PT-expt SO(2) runtime bindings for accelerated kernels.

The array-API dpmodel modules define the shared execution seams. These wrappers
bind the same accelerated implementations and dispatch names used by
``deepmd.pt.model.descriptor.sezm_nn.so2``:

- the block-diagonal GEMM of :class:`SO2Linear`,
- the rotation, value, and attention paths of :class:`SO2Convolution`, and
- the low-rank branch of :class:`DynamicRadialDegreeMixer`.

Triton and cuTile are mutually exclusive complete SO(2) paths. Hand-written
CUDA operators form an independent cumulative layer, while CuTe replaces exact
Neo spans at the enclosing block. Every gate is resolved at construction so
tracing records a static dispatch choice; unsupported layouts retain the
dpmodel reference path.
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
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.kernels.utils import (
    cuda_infer_level,
    cuda_train_enabled,
    triton_infer_level,
    triton_train_level,
    use_cutile_infer,
)

from .edge_cache import (
    cached_edge_csr,
)


def active_triton_level(module: Any) -> int:
    """
    Return the Triton acceleration level that applies to a module's mode.

    Inference and training are gated independently: an operator qualifies for
    inference as soon as it reproduces the forward and the coordinate gradient
    with the parameters held fixed, whereas training additionally requires the
    gradient of every parameter it consumes and a second derivative of its own
    backward, which the force loss traverses. Both levels are captured at
    construction, so the branch this drives resolves at trace time.

    Parameters
    ----------
    module : Any
        Module carrying ``triton_infer_level`` and ``triton_train_level``.

    Returns
    -------
    int
        The level in effect for the module's current mode.
    """
    return module.triton_train_level if module.training else module.triton_infer_level


if TYPE_CHECKING:
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeCache,
    )


@torch_module
class SO2Linear(SO2LinearDP):
    """SO(2)-equivariant linear with an opt-in fused block-diagonal Triton GEMM."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Export override for the block-diagonal vs dense matmul branch below.
        # ``None`` keeps the runtime ``x_flat.is_cuda`` dispatch; the freeze sets
        # it so the AOTI graph follows the *target* device, not the CPU trace.
        self.force_block_diag_matmul: bool | None = None

        # Fast path (``DP_TRITON_INFER >= 1`` or ``DP_TRITON_TRAIN >= 1``):
        # the per-|m|-block batched bmm + cat of ``_block_diagonal_matmul`` is
        # replaced by a fused Triton BN=64 block-diagonal GEMM that consumes
        # the strided operands without a contiguity copy. Bound only when
        # Triton is available and every block width aligns to BN=64;
        # otherwise the eager path is kept. The operator carries its own
        # differentiable backward, so it serves force-loss training too. The
        # gates are read once at construction so they are compile-time
        # constants in the traced (``make_fx``) graph.
        self.triton_infer_level = triton_infer_level()
        self.triton_train_level = triton_train_level()
        self.triton_l_1_block_diag_gemm = None
        if max(self.triton_infer_level, self.triton_train_level) >= 1:
            from deepmd.pt_expt.kernels.triton.sezm.so2_block_gemm import (
                SO2_BLOCK_GEMM_TRITON_AVAILABLE,
                block_diag_gemm,
                slices_supported,
            )

            if SO2_BLOCK_GEMM_TRITON_AVAILABLE and slices_supported(
                self._block_diag_slices
            ):
                self.triton_l_1_block_diag_gemm = block_diag_gemm

    def _block_diagonal_matmul(
        self, x_flat: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        # The dense einsum is a CPU-only fallback: its block ``torch.cat`` lowering
        # trips an Inductor AVX2 C++ codegen bug, so only CPU needs it. Every other
        # device uses the block-diagonal contraction, which skips the structural
        # off-|m| zeros. ``make_fx`` resolves this Python branch at trace time, so
        # the freeze pins ``force_block_diag_matmul`` to the AOTI target device
        # (tracing always runs on CPU regardless of where the artifact will run).
        if self.force_block_diag_matmul is None:
            use_block_diag = not x_flat.is_cpu
        else:
            use_block_diag = self.force_block_diag_matmul
        if not use_block_diag:
            return torch.einsum("fei,ifo->feo", x_flat, weight)
        if (
            self.triton_l_1_block_diag_gemm is not None
            and active_triton_level(self) >= 1
        ):
            # The fused GEMM consumes the ``(F, D_m*Cin, D_m*Cout)`` presentation
            # directly from the strided weight, so the permute is applied here and
            # the contiguity copy the dpmodel ``bmm`` cat path would need is
            # skipped. The eager fallback permutes ``weight`` internally, so it is
            # passed the stored ``(D_m*Cin, F, D_m*Cout)`` layout untouched.
            weight = weight.permute(1, 0, 2)  # (F, D_m*Cin, D_m*Cout)
            return self.triton_l_1_block_diag_gemm(
                x_flat, weight, self._block_diag_slices
            )
        return super()._block_diagonal_matmul(x_flat, weight)


@torch_module
class DynamicRadialDegreeMixer(DynamicRadialDegreeMixerDP):
    """Dynamic radial degree mixer with an opt-in fused Triton low-rank branch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fast path (``DP_TRITON_INFER >= 1`` or ``DP_TRITON_TRAIN >= 1``): a
        # fused Triton kernel replaces the dense scatter and the tiny batched
        # matmul of the ``degree_channel`` low-rank branch in the ``mmax == 1``
        # layout. The operator carries its own differentiable backward with a
        # ``channel_basis`` gradient, so it serves force-loss training too.
        # The gates are read once at construction so they are compile-time
        # constants in the traced (``make_fx``) graph.
        self.triton_infer_level = triton_infer_level()
        self.triton_train_level = triton_train_level()
        self.triton_l_1_radial_mix = None
        if (
            max(self.triton_infer_level, self.triton_train_level) >= 1
            and self.mode == "degree_channel"
            and self.rank > 0
            and self.mmax == 1
        ):
            from deepmd.pt_expt.kernels.triton.sezm.radial_mix import (
                radial_mix_block,
            )

            self.triton_l_1_radial_mix = radial_mix_block

    def _mix_rank_compact(
        self, compact: torch.Tensor, x_local: torch.Tensor
    ) -> torch.Tensor:
        if self.triton_l_1_radial_mix is not None and active_triton_level(self) >= 1:
            return self.triton_l_1_radial_mix(
                compact, x_local, self.channel_basis, self.lmax
            )
        return super()._mix_rank_compact(compact, x_local)


@torch_module
class SO2Convolution(SO2ConvolutionDP):
    """SO(2) convolution with opt-in inference and training kernels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dtype = get_xp_precision(torch, self.precision)
        self.compute_dtype = get_xp_precision(torch, self.compute_precision)
        # Acceleration gates are read once at construction so they become
        # compile-time constants in the traced (``make_fx``) graph. Triton level
        # 1 replaces individual operators, level 2 binds the fused value path,
        # and inference level 3 selects its fp16x3 implementation on swept
        # shapes. ``DP_CUTILE_INFER`` selects a complete alternative SO(2) path
        # and is mutually exclusive with Triton. ``DP_CUDA_INFER`` and
        # ``DP_CUDA_TRAIN`` independently bind hand-written CUDA producers that
        # take precedence over the corresponding narrower paths. CuTe dispatch
        # is resolved at the enclosing block, where its exact-shape SO2 kernel
        # can coexist with every fallback. Unsupported entries remain ``None``.
        self.triton_infer_level = triton_infer_level()
        self.triton_train_level = triton_train_level()
        self.cuda_infer_level = cuda_infer_level()
        self.cutile_infer_enabled = use_cutile_infer()
        if self.triton_infer_level >= 1 and self.cutile_infer_enabled:
            raise ValueError(
                "DP_TRITON_INFER and DP_CUTILE_INFER are mutually exclusive: "
                "each selects a complete accelerated SO(2) inference path."
            )
        self.triton_infer_l_2_value = None
        self.cutile_infer_value = None
        self.edge_csr = cached_edge_csr

        # === Triton rotation kernels: block for mmax == 1, dense otherwise ===
        # The rotation operators carry differentiable backwards (the force
        # loss traverses them twice), so the training gate binds them as well;
        # ``active_triton_level`` then selects the path per mode.
        self.triton_l_1_rotate_to_local = None
        self.triton_l_1_rotate_back = None
        if max(self.triton_infer_level, self.triton_train_level) >= 1:
            from deepmd.pt_expt.kernels.triton.sezm.so2_rotation import (
                rotate_back_block_so2,
                rotate_back_dense,
                rotate_to_local_block,
                rotate_to_local_dense,
            )

            if self.mmax == 1:
                self.triton_l_1_rotate_to_local = lambda x, src, wigner: (
                    rotate_to_local_block(x, src, wigner, self.lmax)
                )
                # The block kernel reads the (E, F, D_m, Cf) focus layout directly,
                # so the rotate-back path passes ``x_local`` before the global
                # reshape and the transpose-back copy is skipped.
                self.triton_l_1_rotate_back = lambda x_local, wigner: (
                    rotate_back_block_so2(x_local, wigner, self.lmax)
                )
            else:
                self.triton_l_1_rotate_to_local = lambda x, src, wigner: (
                    rotate_to_local_dense(
                        x, src, wigner, self.coeff_index_m, self.ebed_dim_full
                    )
                )
                self.triton_l_1_rotate_back = lambda x_local, wigner: rotate_back_dense(
                    x_local, wigner, self.coeff_index_m, self.ebed_dim_full
                )

        # === Step 11. Optional fused flash-attention aggregation kernel ===
        # Folds the entire ``n_atten_head > 0`` value aggregation -- block-diagonal
        # rotate-back, inverse-rotation rescale, envelope-gated softmax weighting,
        # and the destination scatter -- into a single destination-segmented
        # kernel, removing the transient ``x_message`` and weighted-value edge
        # tensors and the ``index_add`` round trip; the op itself dispatches to an
        # eager reference off the CUDA fp32 path. The output-side head gate stays
        # a cheap node-level elementwise applied after the kernel.
        #
        # Layout support is a property of the block, so it is expressed
        # independently of the backend: the kernel only serves the ``mmax == 1``
        # attention layout without the optional focus-mix / value / output
        # projections (the deployed DPA4 configuration). Whichever of the
        # mutually exclusive inference gates is active then supplies the
        # implementation, and ``self.flash_attention`` being bound is what marks
        # the fused path as live.
        # The cuTile aggregation is inference-only; the Triton one also serves
        # training, so it is bound whenever either gate asks for level 1.
        self.flash_attention = None
        self.flash_attention_supports_training = False
        if self.flash_attention_layout_supported and self.cutile_infer_enabled:
            from deepmd.pt_expt.kernels.cutile.sezm.flash_atten import (
                flash_atten_aggregate,
            )

            self.flash_attention = flash_atten_aggregate
        elif (
            self.flash_attention_layout_supported
            and max(self.triton_infer_level, self.triton_train_level) >= 1
        ):
            from deepmd.pt_expt.kernels.triton.sezm.flash_atten import (
                flash_atten_aggregate,
            )

            self.flash_attention = flash_atten_aggregate
            self.flash_attention_supports_training = self.triton_train_level >= 1

        # === Step 12. Optional fused Triton SO(2) value path (inference) ===
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
        # sweep. Training never composes this path: the training value stream
        # is either the level-1 operator composition (step 13 and the dense
        # mixing stack) or the fused CUDA value path (step 15).
        if self.triton_infer_level >= 2:
            from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
                make_triton_value_path,
            )

            self.triton_infer_l_2_value = make_triton_value_path(self)

        # === Step 13. Optional fused rotate-to-local + radial degree mixing ===
        # Level-1 companion of the value path: only the rotation and the
        # degree mixing fuse into one edge-parallel operator writing the
        # focus-major mixing input directly, while the mixing stack itself
        # stays with the compiler. This removes the degree-expanded local
        # intermediate and its relayout from the traced graph; the operator's
        # backward reduces through the source CSR view and carries a
        # hand-derived second order, so it serves force-loss training.
        #
        # The operator is quadrilinear, so a force loss re-enters its forward
        # and backward several times for the second order. That fixed cost is
        # repaid only where the materialization it removes is large: the wide
        # hidden widths. Below the bound the separate rotation and radial-mix
        # kernels win end to end (a 64-wide stack loses ~10%, a 128-wide one
        # gains), so the binding follows the measured crossover.
        self.triton_l_1_rotate_mix = None
        if (
            max(self.triton_infer_level, self.triton_train_level) >= 1
            and self.hidden_channels >= 128
        ):
            from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
                make_triton_rotate_mix,
            )

            self.triton_l_1_rotate_mix = make_triton_rotate_mix(self)

        # === Step 14. Optional fused CUDA SO(2) convolution (inference) ===
        # One hand-written CUDA operator spans the complete per-edge path:
        # rotate-to-local, the radial degree mixing, the gated mixing stack, the
        # inverse rotation, the attention weighting and the destination
        # reduction. It therefore supersedes both the fused value path and the
        # flash aggregation, and takes precedence over them when the block
        # matches its supported configuration. The factory returns ``None``
        # otherwise, leaving whichever narrower path is bound in charge.
        self.cuda_infer_l_2_conv = None
        if self.cuda_infer_level >= 2 and self.flash_attention_layout_supported:
            from deepmd.pt_expt.kernels.cuda.dpa4 import (
                make_cuda_so2_conv,
            )

            self.cuda_infer_l_2_conv = make_cuda_so2_conv(self)

        # === Step 15. Optional fused CUDA SO(2) value path (training) ===
        # One CUDA operator spans the training value stream up to the attention
        # aggregation: rotate-to-local, radial degree mixing, the cross-focus
        # competition weight, the whole gated mixing stack and the final
        # identity layer. Narrow layouts stay in a resident tile kernel; wide
        # layouts compose the rotation kernels and strided cuBLASLt contractions
        # behind the same differentiable boundary. The attention span stays on
        # the Triton operator composition inside the traced graph. Bound under
        # ``DP_CUDA_TRAIN=1``; ``so2_message`` dispatches to it in training mode.
        self.cuda_train_value = None
        if cuda_train_enabled():
            from deepmd.pt_expt.kernels.cuda.dpa4.so2_conv_train import (
                make_cuda_so2_value,
            )

            self.cuda_train_value = make_cuda_so2_value(self)

        # === Step 16. Optional fused destination-segmented attention softmax ===
        # One CSR-segmented operator per direction replaces the
        # scatter/gather softmax chain of the attention weights, sharing the
        # destination-sorted view with the flash aggregation; its backward and
        # hand-derived second order keep the force-loss trace from expanding
        # the chain into materialized surfaces and serialized scatters. The
        # source-gated (SFPG) form keeps the reference path.
        self.triton_l_1_segment_softmax = None
        if (
            max(self.triton_infer_level, self.triton_train_level) >= 1
            and self.attn_n_focus * self.n_atten_head <= 16
        ):
            from deepmd.pt_expt.kernels.triton.sezm.segment_softmax import (
                SEGMENT_SOFTMAX_TRITON_AVAILABLE,
                segment_softmax,
            )

            if SEGMENT_SOFTMAX_TRITON_AVAILABLE:
                self.triton_l_1_segment_softmax = segment_softmax

        # === Step 17. Optional fused cuTile SO(2) value-path operators ===
        # Complete cuTile inference path, mutually exclusive with Triton. The
        # factory validates the block layout and returns ``None``
        # otherwise, leaving the dense reference path in charge.
        if self.cutile_infer_enabled:
            from deepmd.pt_expt.kernels.cutile.sezm.so2_value_path import (
                make_cutile_value_path,
            )

            self.cutile_infer_value = make_cutile_value_path(self)

    def _rotate_mix(
        self,
        x: torch.Tensor,
        edge_cache: EdgeCache,
        radial_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.triton_l_1_rotate_mix is not None and active_triton_level(self) >= 1:
            # The operator's backward reduces through the source CSR view,
            # built once per step and kept on the edge cache.
            cached_edge_csr(edge_cache, "src", x.shape[0])
            u0, rad_feat = self.triton_l_1_rotate_mix(x, edge_cache, radial_feat)
            x_local = u0.view(
                self.n_focus,
                edge_cache.src.shape[0],
                self.reduced_dim,
                self.so2_focus_dim,
            )  # (F, E, D_m, Cf)
            return x_local, rad_feat
        return super()._rotate_mix(x, edge_cache, radial_feat)

    def _attention_softmax(
        self,
        attn_logits: torch.Tensor,
        edge_cache: EdgeCache,
        n_nodes: int,
    ) -> torch.Tensor:
        active_level = active_triton_level(self)
        if (
            self.triton_l_1_segment_softmax is not None
            and edge_cache.edge_src_gate is None
            and attn_logits.is_cuda
            and active_level >= 1
        ):
            # The fused operator runs the whole normalization as one
            # CSR-segmented kernel per direction (forward, backward, second
            # order), sharing the destination-sorted view with the flash
            # aggregation; the scatter/gather chain and its expansion under
            # the force loss never reach the traced graph.
            n_edge = attn_logits.shape[0]
            order, row_ptr = cached_edge_csr(edge_cache, "dst", n_nodes)
            null_logit = torch.log(
                torch.nn.functional.softplus(
                    self.adamw_attn_z_bias_raw.to(dtype=torch.float32)
                )
                + float(self.eps)
            ).reshape(-1)  # (F * H,)
            n_channel = self.attn_n_focus * self.n_atten_head
            alpha = self.triton_l_1_segment_softmax(
                attn_logits.reshape(n_edge, n_channel).to(dtype=torch.float32),
                edge_cache.edge_env.reshape(n_edge).to(dtype=torch.float32),
                null_logit,
                order,
                row_ptr,
                edge_cache.dst,
            )
            return alpha.to(dtype=attn_logits.dtype).reshape(
                n_edge, self.attn_n_focus, self.n_atten_head
            )
        return super()._attention_softmax(attn_logits, edge_cache, n_nodes)

    def rotation_kernel_active(self) -> bool:
        """Return whether the bound rotation kernels serve the current mode.

        Returns
        -------
        bool
            Whether level-one Triton rotation is active.
        """
        return (
            self.triton_l_1_rotate_to_local is not None
            and active_triton_level(self) >= 1
        )

    def _rotate_to_local(
        self, x: torch.Tensor, edge_cache: EdgeCache
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.rotation_kernel_active():
            # The block kernel serves the m-major ``mmax == 1`` layout; the
            # dense kernel serves every other supported layout.
            D_full = edge_cache.D_full
            x_local = self.triton_l_1_rotate_to_local(x, edge_cache.src, D_full)
            x_dst_local: torch.Tensor | None = None
            if self.node_wise_grid_product is not None:
                x_dst_local = self.triton_l_1_rotate_to_local(x, edge_cache.dst, D_full)
            return x_local, x_dst_local
        return super()._rotate_to_local(x, edge_cache)

    def _rotate_back(
        self, x_local: torch.Tensor, edge_cache: EdgeCache, n_edge: int
    ) -> torch.Tensor:
        if self.rotation_kernel_active():
            Dt_full = edge_cache.Dt_full
            if self.mmax == 1:
                # The block kernel consumes the (E, F, D_m, Cf) focus layout in
                # place, folding the inverse transpose into its channel addressing.
                return self.triton_l_1_rotate_back(x_local, Dt_full)
            # Restore reduced global layout (E, D_m, C_wide) for the dense kernel.
            x_std = (
                x_local.transpose(1, 2)
                .contiguous()
                .reshape(n_edge, self.reduced_dim, self.hidden_channels)
            )
            return self.triton_l_1_rotate_back(x_std, Dt_full)
        return super()._rotate_back(x_local, edge_cache, n_edge)
