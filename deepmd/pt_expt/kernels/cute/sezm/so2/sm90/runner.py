# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Split-complex strict-FP32 SM90 implementation of the complete Neo SO2.

The value path uses one split-complex edge representation throughout:

* compact radial projection and direct split-complex Phase A;
* two persistent split-complex gated residual layers;
* the third residual SO2Linear commuted through Phase C and evaluated from
  64-edge destination-CSR sufficient statistics;
* the Neo output gate followed by the message-grid/readout.

No ``(E,2,10,32)`` block-real SO2 slab is constructed, and no split/block-real
pack or unpack kernel is used.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

import torch
from torch import (
    Tensor,
)

from ..linear import (
    cached_neo_so2_linear_weights,
)
from ..operation import (
    _compile_output_gate_backward,
    _equivariant_rmsnorm_backward,
    _runner_compile_identity,
    _so3_linear_backward_input,
    _x_wide_manual_backward,
)
from ..runner import (
    NeoFullCuteBackward,
)
from .final_phase_c import (
    ExpandedFinalWeights,
    prepare_expanded_final_weights,
    run_direct_statistics_forward,
)
from .message_grid_readout import (
    prepare_sm90_message_grid_state,
    run_sm90_message_grid_backward,
)
from .output_gate import (
    run_chunked_final_output_gate,
)
from .persistent import (
    NeoPersistentComplexSaved,
    NeoPersistentComplexState,
    NeoPersistentComplexWeights,
    prepare_neo_persistent_complex_weights,
)
from .phase_a import (
    run_neo_phase_a_persistent_complex_fp32,
)
from .phase_a_backward import (
    run_neo_phase_a_persistent_complex_backward_fp32,
)
from .phase_c_attention_backward import (
    allocate_grouped_expanded_final_phase_c_attention_adjoint_outputs,
    run_grouped_expanded_final_phase_c_attention_adjoint,
)
from .prefix import (
    run_persistent_prefix_forward_inplace,
    run_persistent_prefix_input_adjoint_destructive_saved,
)
from .radial import (
    project_neo_radial_input_adjoint_fp32,
    run_neo_radial_state_forward_fp32,
)

FOCUS_COUNT = 2
CHANNELS = 32
HIDDEN = FOCUS_COUNT * CHANNELS
DEGREE_COUNT = 16
M0_WIDTH = 128
M1_WIDTH = 96
GATED_LAYERS = 2
_PERSISTENT_WEIGHT_CACHE = "_deepmd_cute_neo_sm90_persistent_weights"
_FINAL_WEIGHT_CACHE = "_deepmd_cute_neo_sm90_final_weights"


def _tensor_version_key(tensor: Tensor) -> tuple[Any, ...]:
    return (
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _prepare_persistent_weights(so2: Any) -> NeoPersistentComplexWeights:
    """Pack the three SO2Linear blocks and two scalar gates once."""
    linears = tuple(so2.so2_linears)
    nonlinearities = tuple(so2.non_linearities)
    if len(linears) != 3 or len(nonlinearities) != 3:
        raise NotImplementedError("SM90 SO2 requires three SO2 layers")
    if any(type(norm).__name__ != "Identity" for norm in so2.so2_inter_norms):
        raise NotImplementedError("SM90 SO2 requires disabled inter-layer norms")
    for nonlinearity in nonlinearities[:GATED_LAYERS]:
        if getattr(nonlinearity, "layout", None) != "fndc":
            raise NotImplementedError("SM90 SO2 requires fndc gates")
        activation = getattr(nonlinearity, "activation_function", None)
        if activation is None:
            activation = getattr(
                getattr(nonlinearity, "scalar_act", None),
                "activation",
                None,
            )
        if str(activation).lower() != "silu":
            raise NotImplementedError("SM90 SO2 requires SiLU gates")
        if getattr(nonlinearity.gate_linear, "bias", None) is not None:
            raise NotImplementedError("SM90 SO2 does not support gate bias")

    sources = tuple(
        tensor
        for linear in linears
        for tensor in (linear.weight_m0, linear.weight_m[0])
    ) + tuple(
        nonlinearity.gate_linear.weight
        for nonlinearity in nonlinearities[:GATED_LAYERS]
    )
    cache_key = tuple(_tensor_version_key(tensor) for tensor in sources)
    cached = getattr(so2, _PERSISTENT_WEIGHT_CACHE, None)
    if isinstance(cached, tuple) and len(cached) == 3 and cached[0] == cache_key:
        return cached[1]

    w0_layers: list[Tensor] = []
    wp_layers: list[Tensor] = []
    for linear in linears:
        w0, wp = cached_neo_so2_linear_weights(linear)
        w0_layers.append(w0)
        wp_layers.append(wp)
    gate_layers = [
        nonlinearity.gate_linear.weight.detach()
        .view(CHANNELS, FOCUS_COUNT, 3 * CHANNELS)
        .permute(1, 0, 2)
        .contiguous()
        for nonlinearity in nonlinearities[:GATED_LAYERS]
    ]
    packed = prepare_neo_persistent_complex_weights(
        torch.stack(w0_layers, dim=0).contiguous(),
        torch.stack(wp_layers, dim=0).contiguous(),
        torch.stack(gate_layers, dim=0).contiguous(),
    )
    setattr(so2, _PERSISTENT_WEIGHT_CACHE, (cache_key, packed, sources))
    return packed


def _prepare_final_weights(so2: Any) -> ExpandedFinalWeights:
    """Cache the residual-folded final SO2Linear in node-GEMM form."""
    linear = tuple(so2.so2_linears)[-1]
    sources = (linear.weight_m0, linear.weight_m[0])
    key = tuple(_tensor_version_key(tensor) for tensor in sources)
    cached = getattr(so2, _FINAL_WEIGHT_CACHE, None)
    if isinstance(cached, tuple) and len(cached) == 3 and cached[0] == key:
        return cached[1]

    w0, wp = cached_neo_so2_linear_weights(linear)
    if tuple(w0.shape) != (FOCUS_COUNT, M0_WIDTH, M0_WIDTH):
        raise ValueError("SM90 SO2 requires final m=0 weights (2,128,128)")
    if tuple(wp.shape) != (FOCUS_COUNT, 2 * M1_WIDTH, 2 * M1_WIDTH):
        raise ValueError("SM90 SO2 requires final pair weights (2,192,192)")
    u = wp[:, :M1_WIDTH, :M1_WIDTH]
    v = wp[:, :M1_WIDTH, M1_WIDTH:]
    if not torch.equal(wp[:, M1_WIDTH:, :M1_WIDTH], -v):
        raise ValueError("final pair weight lower-left block must be -V")
    if not torch.equal(wp[:, M1_WIDTH:, M1_WIDTH:], u):
        raise ValueError("final pair weight lower-right block must be U")
    weights = prepare_expanded_final_weights(
        (w0 + torch.eye(M0_WIDTH, device=w0.device, dtype=torch.float32)).contiguous(),
        (
            torch.complex(u, v)
            + torch.eye(M1_WIDTH, device=w0.device, dtype=torch.complex64)
        ).contiguous(),
    )
    setattr(so2, _FINAL_WEIGHT_CACHE, (key, weights, sources))
    return weights


__all__ = ["NeoSm90SO2Runner"]


@dataclass(frozen=True)
class _PersistentPrefixForward:
    """Pre-final split state and exact gate preactivations for its adjoint."""

    state: NeoPersistentComplexState
    saved: NeoPersistentComplexSaved


def _run_persistent_prefix_forward(
    state: NeoPersistentComplexState,
    weights: NeoPersistentComplexWeights,
) -> _PersistentPrefixForward:
    """Run two gated layers while reusing the direct Phase-A state buffer."""
    current, saved = run_persistent_prefix_forward_inplace(state, weights)
    return _PersistentPrefixForward(
        state=current,
        saved=saved,
    )


def _run_persistent_prefix_input_adjoint(
    grad_out: NeoPersistentComplexState,
    saved: NeoPersistentComplexSaved,
    weights: NeoPersistentComplexWeights,
) -> NeoPersistentComplexState:
    """Overwrite dead gate checkpoints and the running residual in-place."""
    return run_persistent_prefix_input_adjoint_destructive_saved(
        grad_out,
        saved,
        weights,
    )


def _run_final_reverse_panels(
    *,
    grad_out: Tensor,
    weights: Any,
) -> tuple[Tensor, Tensor]:
    """Form the two strict-FP32 reverse node panels once."""
    node_count = int(grad_out.shape[0])
    grad_node = grad_out.permute(1, 2, 0, 3).contiguous()
    b0 = torch.bmm(
        grad_node.flatten(0, 1),
        weights.w0.flatten(0, 1).transpose(-1, -2),
    ).view(FOCUS_COUNT, DEGREE_COUNT, node_count, M0_WIDTH)
    grad_node1 = grad_node[:, 1:].to(torch.complex64).contiguous()
    b1 = torch.bmm(
        grad_node1.flatten(0, 1),
        weights.wc.flatten(0, 1).conj().transpose(-1, -2),
    ).view(FOCUS_COUNT, DEGREE_COUNT - 1, node_count, M1_WIDTH)
    return b0, b1


def _qk_node_input_adjoint(
    runner: NeoSm90SO2Runner,
    grad_q_node: Tensor,
    grad_k_node: Tensor,
) -> Tensor:
    """Map fused Q/K node adjoints into the wide SO2 input."""
    so2 = runner.so2
    x_wide = runner.x_wide.detach()
    x_l0 = x_wide[:, 0, :].reshape(runner.node_count, FOCUS_COUNT, CHANNELS)
    grad_x_wide = torch.empty_like(x_wide, memory_format=torch.contiguous_format)
    runner.qk_node_input_adjoint(
        x_l0.contiguous(),
        grad_q_node,
        grad_k_node,
        so2.attn_q_proj.weight.detach()
        .float()
        .view(CHANNELS, FOCUS_COUNT, CHANNELS)
        .contiguous(),
        so2.attn_k_proj.weight.detach()
        .float()
        .view(CHANNELS, FOCUS_COUNT, CHANNELS)
        .contiguous(),
        so2.attn_qk_norm.adam_scale.detach().float().contiguous(),
        grad_x_wide.view(runner.node_count, DEGREE_COUNT * HIDDEN),
    )
    return grad_x_wide


class NeoSm90SO2Runner(NeoFullCuteBackward):
    """Complete Neo SO2 runner with one native split representation."""

    uses_native_sm90_path = True

    def _build_forward_graph(self) -> None:
        torch_module = self.torch
        so2 = self.so2
        block = self.block
        node_count = self.node_count
        edge_count = self.edge_count
        if self.compute_capability != (9, 0):
            raise RuntimeError("split-complex SO2 requires SM90")

        self.structural_scratch = None
        self.use_full_node = block.node_lmax == block.lmax
        x_so2 = self.x if self.use_full_node else self.x[:, : block.mp_ebed_dim]
        x_pre = block.pre_so2_norm(x_so2)
        self.x_wide = (
            so2.pre_focus_mix(
                x_pre.reshape(
                    node_count,
                    x_so2.shape[1],
                    block.channels,
                ).unsqueeze(2)
            )
            .squeeze(2)
            .contiguous()
        )

        self.radial_compact, radial_l0 = run_neo_radial_state_forward_fp32(
            radial_feat=self.radial.detach().contiguous(),
            combined_weight=self.combined_radial,
            hidden_weight=so2.radial_hidden_proj.weight.detach().contiguous(),
        )
        phase_a_state = run_neo_phase_a_persistent_complex_fp32(
            x_wide=self.x_wide.detach(),
            src=self.src_i32,
            d_full=self.d.detach(),
            radial_compact=self.radial_compact,
            channel_basis=so2.radial_degree_mixer.channel_basis.detach()
            .view(HIDDEN)
            .contiguous(),
        )
        self.focus_gate_src = (
            phase_a_state.m0[:, :, :CHANNELS].permute(1, 0, 2).contiguous()
        )

        self.persistent_weights = _prepare_persistent_weights(so2)
        prefix = _run_persistent_prefix_forward(
            phase_a_state,
            self.persistent_weights,
        )
        self.phase_c_state = prefix.state
        self.persistent_saved = prefix.saved
        del phase_a_state, prefix

        x_l0_node = self.x_wide[:, 0, :].reshape(
            node_count,
            FOCUS_COUNT,
            CHANNELS,
        )
        self.focus_alpha = torch_module.empty(
            edge_count,
            FOCUS_COUNT,
            device=self.x.device,
            dtype=torch_module.float32,
        )
        self.q_node = torch_module.empty_like(
            x_l0_node,
            memory_format=torch_module.contiguous_format,
        )
        self.k_node = torch_module.empty_like(
            x_l0_node,
            memory_format=torch_module.contiguous_format,
        )
        self.attention_prelude_forward(
            self.focus_gate_src.view(edge_count, HIDDEN),
            x_l0_node.contiguous(),
            so2.adamw_focus_compete_w.detach().float().contiguous(),
            self.focus_norm_scale,
            so2.attn_q_proj.weight.detach()
            .float()
            .view(CHANNELS, FOCUS_COUNT, CHANNELS)
            .contiguous(),
            so2.attn_k_proj.weight.detach()
            .float()
            .view(CHANNELS, FOCUS_COUNT, CHANNELS)
            .contiguous(),
            so2.attn_qk_norm.adam_scale.detach().float().contiguous(),
            self.focus_alpha,
            self.q_node,
            self.k_node,
        )

        self.attn_logits = torch_module.empty(
            edge_count,
            FOCUS_COUNT,
            device=self.x.device,
            dtype=torch_module.float32,
        )
        self.qk_edge_forward(
            self.q_node,
            self.k_node,
            radial_l0.view(edge_count, FOCUS_COUNT, CHANNELS),
            so2.adamw_attn_logit_w.detach().contiguous(),
            self.src_i32,
            self.dst_i32,
            self.attn_logits,
        )
        del radial_l0
        self.softmax_fwd(
            self.attn_logits,
            self.edge_gate,
            self.dst_ptr_i32,
            so2.adamw_attn_z_bias_raw.detach()
            .reshape(FOCUS_COUNT)
            .float()
            .contiguous(),
            self.alpha,
            self.group_max,
            self.denom,
        )
        # First input adjoints need alpha, Q, K, and edge metadata, but not the
        # materialized logits or the null-mass parameter-gradient statistics.
        self.attn_logits = None
        self.group_max = None
        self.denom = None

        self.beta = (self.alpha * self.focus_alpha).contiguous()
        self.final_weights = _prepare_final_weights(so2)
        raw_phase_c = run_direct_statistics_forward(
            m0=self.phase_c_state.m0,
            m1=self.phase_c_state.m1,
            dt_packed=self.dt.detach(),
            beta=self.beta,
            dst_ptr=self.dst_ptr_i32,
            weights=self.final_weights,
        ).output
        self.phase_c_out = run_chunked_final_output_gate(
            raw=raw_phase_c,
            x_wide=self.x_wide.detach(),
            norm_scale=so2.attn_output_gate_norm.adam_scale.detach()
            .float()
            .reshape(FOCUS_COUNT, CHANNELS)
            .contiguous(),
            gate_weight=so2.adamw_attn_gate_w.detach()
            .float()
            .reshape(CHANNELS, FOCUS_COUNT, 1)
            .contiguous(),
            rotate_inv_rescale=so2.rotate_inv_rescale_full.detach().contiguous(),
            eps=float(so2.attn_output_gate_norm.eps),
        ).to(dtype=so2.compute_dtype)
        # The partial/node statistics and ungated output have no backward role:
        # the exact adjoint recomputes from grad_out and the pre-final state.
        del raw_phase_c

        out = self.phase_c_out.detach().to(dtype=so2.dtype)
        self.out_gate_flat = out.detach()
        self.message_grid_product = None
        self.message_grid_sm90_state = None
        if so2.message_node_grid_product is not None:
            if self.packed_message_grid:
                from ..message_grid import (
                    run_packed_message_grid_forward,
                )

                self.message_grid_sm90_state = prepare_sm90_message_grid_state(
                    so2.message_node_grid_product
                )
                grid_out, product = run_packed_message_grid_forward(
                    so2.message_node_grid_product,
                    out,
                    self.x_wide,
                    return_product=True,
                    sm90_state=self.message_grid_sm90_state,
                )
                self.message_grid_product = product.detach()
            else:
                grid_out = so2.message_node_grid_product(out, self.x_wide)
            out = out + grid_out

        self.post_mix_input = out.detach()
        out = so2.post_focus_mix(out.unsqueeze(2)).squeeze(2)
        self.post_norm_input = out.unsqueeze(2).detach()
        so2_out = block.post_so2_norm(self.post_norm_input)
        if self.use_full_node:
            self.final = so2_out
        else:
            final = self.x.new_zeros(self.x.shape)
            final[:, : block.mp_ebed_dim] = so2_out
            self.final = final

    def input_adjoint(self, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return SO2 input adjoints through the split-complex reverse path."""
        return _runner_backward(self, grad_out)


def _final_manual_backward_sm90(
    runner: NeoSm90SO2Runner,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor]:
    """Use the one-slab tiled message-grid adjoint in the final SO2 boundary."""
    if not (
        runner.packed_message_grid
        and runner.so2.message_node_grid_product is not None
        and runner.message_grid_sm90_state is not None
    ):
        raise RuntimeError("SM90 SO2 requires the packed message-grid path")

    so2 = runner.so2
    block = runner.block
    if runner.use_full_node:
        grad_so2_out = grad_out
    else:
        grad_so2_out = grad_out[:, : block.mp_ebed_dim, :, :]

    phase = runner.phase_c_out.detach()
    x_wide = runner.x_wide.detach()
    grad_post_norm_in = _equivariant_rmsnorm_backward(
        block.post_so2_norm,
        runner.post_norm_input,
        grad_so2_out,
    )
    grad_post_mix = _so3_linear_backward_input(
        so2.post_focus_mix,
        grad_post_norm_in.squeeze(2).unsqueeze(2),
    ).squeeze(2)

    message_grid_product = runner.message_grid_product
    runner.message_grid_product = None
    grad_out_gate_flat, grad_grid_context = run_sm90_message_grid_backward(
        so2.message_node_grid_product,
        runner.out_gate_flat,
        x_wide,
        grad_post_mix,
        message_grid_product,
        runner.message_grid_sm90_state,
    )
    del message_grid_product
    runner.message_grid_sm90_state = None

    grad_out_gate_flat.add_(grad_post_mix)
    # FrameExpand's input adjoint preserves its degree-major einsum stride.
    # The output-gate kernel updates one flat node panel in place, so establish
    # that writable contract once at this consumer boundary.
    grad_x_wide_down = grad_grid_context.contiguous()
    grad_phase = grad_out_gate_flat.contiguous()
    output_gate_backward = _compile_output_gate_backward(
        _runner_compile_identity(runner),
        float(so2.attn_output_gate_norm.eps),
    )
    output_gate_backward(
        grad_phase.view(runner.node_count, DEGREE_COUNT * HIDDEN),
        phase.contiguous().view(runner.node_count, DEGREE_COUNT * HIDDEN),
        x_wide.contiguous().view(runner.node_count, DEGREE_COUNT * HIDDEN),
        so2.attn_output_gate_norm.adam_scale.detach()
        .float()
        .reshape(FOCUS_COUNT, CHANNELS)
        .contiguous(),
        so2.adamw_attn_gate_w.detach()
        .float()
        .reshape(CHANNELS, FOCUS_COUNT, 1)
        .contiguous(),
        grad_phase.view(runner.node_count, DEGREE_COUNT * HIDDEN),
        grad_x_wide_down.view(runner.node_count, DEGREE_COUNT * HIDDEN),
    )
    return grad_phase.reshape_as(phase), grad_x_wide_down


def _runner_backward(
    runner: NeoSm90SO2Runner,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    so2 = runner.so2
    grad_phase, grad_x_wide_down = _final_manual_backward_sm90(runner, grad_out)
    runner.phase_c_out = None
    runner.out_gate_flat = None
    runner.post_mix_input = None
    runner.post_norm_input = None
    grad_node = (
        grad_phase.view(
            runner.node_count,
            DEGREE_COUNT,
            FOCUS_COUNT,
            CHANNELS,
        )
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    grad_node.mul_(runner.rotate.view(1, 1, DEGREE_COUNT, 1))
    b0, b1 = _run_final_reverse_panels(
        grad_out=grad_node,
        weights=runner.final_weights,
    )
    fused_outputs = allocate_grouped_expanded_final_phase_c_attention_adjoint_outputs(
        edge_count=runner.edge_count,
        node_count=runner.node_count,
        device=grad_node.device,
        grad_m0=runner.phase_c_state.m0,
        grad_m1=runner.phase_c_state.m1,
    )
    fused_adjoint = run_grouped_expanded_final_phase_c_attention_adjoint(
        b0=b0,
        b1=b1,
        m0=runner.phase_c_state.m0,
        m1=runner.phase_c_state.m1,
        dt_packed=runner.dt.detach(),
        beta=runner.beta,
        alpha=runner.alpha,
        focus_alpha=runner.focus_alpha,
        focus_src=runner.focus_gate_src,
        focus_weight=so2.adamw_focus_compete_w.detach().float().contiguous(),
        focus_scale=runner.focus_norm_scale,
        q_node=runner.q_node,
        k_node=runner.k_node,
        edge_gate=runner.edge_gate,
        src=runner.src_i32,
        dst_ptr=runner.dst_ptr_i32,
        focus_eps=runner.focus_norm_eps,
        focus_tau=float(so2.focus_softmax_tau),
        label_smoothing=float(so2.focus_label_smoothing),
        qk_scale=CHANNELS**-0.5,
        use_focus_norm=runner.focus_norm_enabled,
        outputs=fused_outputs,
    )
    grad_stack = NeoPersistentComplexState(
        fused_adjoint.grad_m0,
        fused_adjoint.grad_m1,
    )
    grad_dt = fused_adjoint.grad_dt
    grad_logits = fused_adjoint.grad_logits
    grad_edge = fused_adjoint.grad_edge
    grad_focus_src = fused_adjoint.grad_focus_src
    grad_q_node = fused_adjoint.grad_q_node
    grad_k_node = fused_adjoint.grad_k_node
    del fused_adjoint, b0, b1, grad_node
    runner.phase_c_state = None
    runner.beta = None
    runner.final_weights = None
    runner.focus_gate_src = None
    runner.alpha = None
    runner.focus_alpha = None

    grad_phase_a = _run_persistent_prefix_input_adjoint(
        grad_stack,
        runner.persistent_saved,
        runner.persistent_weights,
    )
    runner.persistent_saved = None
    # focus_gate_src aliases the first m=0 row of the direct Phase-A result.
    grad_phase_a.m0[:, :, :CHANNELS].add_(grad_focus_src)
    del grad_stack, grad_focus_src
    phase_a = run_neo_phase_a_persistent_complex_backward_fp32(
        grad_state=grad_phase_a,
        radial_compact=runner.radial_compact,
        channel_basis=so2.radial_degree_mixer.channel_basis.detach()
        .view(HIDDEN)
        .contiguous(),
        x_wide=runner.x_wide.detach(),
        source_order=runner.source_order_i32,
        source_ptr=runner.source_ptr_i32,
        d_full=runner.d.detach(),
    )
    runner.radial_compact = None
    runner.persistent_weights = None
    grad_radial = project_neo_radial_input_adjoint_fp32(
        grad_compact=phase_a.grad_radial_compact,
        grad_logits=grad_logits,
        combined_weight=runner.combined_radial,
        combined_attention_weight=runner.combined_attention_radial,
    )
    grad_x_wide = phase_a.grad_x_wide.view_as(runner.x_wide)
    grad_x_wide.add_(_qk_node_input_adjoint(runner, grad_q_node, grad_k_node))
    runner.q_node = None
    runner.k_node = None
    grad_x_wide.add_(grad_x_wide_down)
    grad_x = _x_wide_manual_backward(runner, grad_x_wide)
    runner.grad_edge = grad_edge

    return grad_x, phase_a.grad_d_full, grad_dt, grad_radial
