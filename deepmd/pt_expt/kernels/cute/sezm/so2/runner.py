# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime runner for the Neo CuTe SO2 unit."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

from ..runtime_policy import (
    FUSED_SO2_GATE_CAPABILITIES,
    SM80_PROFILE_CAPABILITIES,
    SM90_CAPABILITY,
    SUPPORTED_SO2_CAPABILITIES,
)


def _uses_packed_message_grid(compute_capability: tuple[int, int]) -> bool:
    """Return whether the packed message-grid kernels support this GPU."""
    return (
        compute_capability in SM80_PROFILE_CAPABILITIES
        or compute_capability == SM90_CAPABILITY
    )


def _validate_runtime_config(
    runtime_config: Any,
    *,
    compute_capability: tuple[int, int] | None = None,
) -> None:
    if compute_capability not in SUPPORTED_SO2_CAPABILITIES:
        raise RuntimeError("Neo SO2 requires a supported compute capability")
    if runtime_config.native_sm90_path != (compute_capability == SM90_CAPABILITY):
        raise RuntimeError("the native SM90 SO2 path must be selected only on sm_90")
    if runtime_config.per_focus_so2_fwd_pair != (
        compute_capability in SM80_PROFILE_CAPABILITIES
    ):
        raise RuntimeError("the per-focus SO2 path must be selected on sm_80/sm_86")
    if runtime_config.combined_so2_gate != (
        compute_capability in FUSED_SO2_GATE_CAPABILITIES
    ):
        raise RuntimeError(
            "the combined SO2/gate path must be selected on sm_89/sm_120"
        )


def _combined_radial_weight(
    torch: Any,
    radial_hidden_proj: Any,
    radial_degree_mixer: Any,
) -> Any:
    hidden_weight = radial_hidden_proj.weight.detach()
    mixer_weight = radial_degree_mixer.weight.detach()
    cache_key = (
        hidden_weight.data_ptr(),
        hidden_weight._version,
        mixer_weight.data_ptr(),
        mixer_weight._version,
        hidden_weight.dtype,
        hidden_weight.device,
    )
    cache = getattr(radial_degree_mixer, "_deepmd_cute_combined_weight", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1]
    blocks = []
    for degree in range(4):
        mixer_block = mixer_weight[degree * 64 : (degree + 1) * 64, :]
        blocks.append(torch.mm(hidden_weight, mixer_block))
    combined = torch.cat(blocks, dim=0).contiguous()
    radial_degree_mixer._deepmd_cute_combined_weight = (cache_key, combined)
    return combined


def _combined_attention_radial_weight(
    torch: Any,
    radial_hidden_proj: Any,
    attention_weight: Any,
) -> Any:
    hidden_weight = radial_hidden_proj.weight.detach()
    attention_weight = attention_weight.detach()
    cache_key = (
        hidden_weight.data_ptr(),
        hidden_weight._version,
        attention_weight.data_ptr(),
        attention_weight._version,
        hidden_weight.dtype,
        hidden_weight.device,
    )
    cache = getattr(radial_hidden_proj, "_deepmd_cute_attention_radial_weight", None)
    if cache is not None and cache[0] == cache_key:
        return cache[1]
    blocks = []
    for focus in range(2):
        hidden_block = hidden_weight[:, focus * 32 : (focus + 1) * 32]
        blocks.append(torch.mv(hidden_block, attention_weight[:, focus, 0]))
    combined = torch.stack(blocks, dim=1).contiguous()
    radial_hidden_proj._deepmd_cute_attention_radial_weight = (cache_key, combined)
    return combined


def _batched_radial_projection_weight(
    radial_degree_mixer: Any,
    combined_weight: Any,
    attention_radial_weight: Any,
) -> Any:
    cache_key = (
        combined_weight.data_ptr(),
        combined_weight._version,
        attention_radial_weight.data_ptr(),
        attention_radial_weight._version,
        combined_weight.dtype,
        combined_weight.device,
    )
    cache = getattr(
        radial_degree_mixer,
        "_deepmd_cute_batched_radial_projection_weight",
        None,
    )
    if cache is not None and cache[0] == cache_key:
        return cache[1]

    from .radial_phase_a import (
        prepare_batched_radial_projection_weight,
    )

    projection_weight = prepare_batched_radial_projection_weight(
        combined_weight,
        attention_radial_weight,
    )
    # Keep the derived operands alive so allocator pointer reuse cannot spoof
    # the versioned cache key after a parameter update.
    radial_degree_mixer._deepmd_cute_batched_radial_projection_weight = (
        cache_key,
        projection_weight,
        combined_weight,
        attention_radial_weight,
    )
    return projection_weight


def _edge_gate(torch: Any, edge_cache: Any) -> Any:
    gate = edge_cache.edge_env.reshape(-1).float().clamp_min(0.0)
    if edge_cache.edge_src_gate is not None:
        gate = gate * edge_cache.edge_src_gate.reshape(-1).float().clamp_min(0.0).sqrt()
    return gate.contiguous()


@dataclass
class StackCache:
    y: Any
    logits: Any | None
    non_linear: Any
    final: bool

    def __post_init__(self) -> None:
        self.y = self.y.detach()
        if self.logits is not None:
            self.logits = self.logits.detach()


class _NeoSO2BackwardWorkspaceBase:
    _EXPORTED_NAMES = (
        "grad_stack_focus",
        "grad_focus_alpha",
        "grad_dt",
        "grad_alpha",
        "grad_logits",
        "grad_edge",
        "grad_z_partial",
        "grad_z",
        "grad_x_rot",
        "grad_radial_flat",
        "grad_x_wide_phase_a",
        "grad_d",
        "grad_y",
        "grad_gate_logits",
        "grad_mixed_slab",
    )

    def attach_to(self, runner: Any) -> None:
        for name in self._EXPORTED_NAMES:
            setattr(runner, name, getattr(self, name))


def _structural_memory_views(
    torch: Any,
    *,
    edge_count: int,
    like: Any,
    phase_c_stack: Any,
    phase_c_y: Any | None,
    radial_scratch: Any,
    radial_values_per_edge: int,
    phase_c_single_input_reuse: bool,
) -> tuple[Any, Any | None, Any]:
    """Validate and expose saved stack buffers that become backward scratch."""
    expected_phase_shape = (edge_count, 2, 10, 32)
    if phase_c_single_input_reuse:
        if phase_c_y is not None:
            raise ValueError(
                "structural memory reuse requires phase_c_y to be absent in "
                "single-input mode"
            )
        phase_tensors = (("phase_c_stack", phase_c_stack, expected_phase_shape),)
    else:
        phase_tensors = (
            ("phase_c_stack", phase_c_stack, expected_phase_shape),
            ("phase_c_y", phase_c_y, expected_phase_shape),
        )
    for name, tensor, expected_shape in phase_tensors:
        if tensor is None or tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"structural memory reuse requires {name} shape {expected_shape}"
            )
        if tensor.dtype != torch.float32 or tensor.dtype != like.dtype:
            raise ValueError(f"structural memory reuse requires FP32 {name} storage")
        if tensor.device != like.device or not tensor.is_contiguous():
            raise ValueError(
                f"structural memory reuse requires contiguous {name} on {like.device}"
            )
        if tensor.storage_offset() != 0:
            raise ValueError(
                f"structural memory reuse requires zero-offset {name} storage"
            )
    required_radial_values = edge_count * radial_values_per_edge
    if radial_scratch is None or radial_scratch.numel() < required_radial_values:
        raise ValueError(
            "structural memory reuse requires radial_scratch capacity of at least "
            f"{required_radial_values} values"
        )
    if radial_scratch.dtype != torch.float32 or radial_scratch.dtype != like.dtype:
        raise ValueError("structural memory reuse requires FP32 radial_scratch storage")
    if radial_scratch.device != like.device or not radial_scratch.is_contiguous():
        raise ValueError(
            "structural memory reuse requires contiguous radial_scratch on "
            f"{like.device}"
        )
    if radial_scratch.storage_offset() != 0:
        raise ValueError(
            "structural memory reuse requires zero-offset radial_scratch storage"
        )
    tensors = (*[tensor for _, tensor, _ in phase_tensors], radial_scratch)
    storages = {tensor.untyped_storage()._cdata for tensor in tensors}
    if len(storages) != len(tensors):
        raise ValueError("structural memory reuse requires distinct storages")
    return (
        phase_c_stack.view(edge_count, 10 * 64),
        None if phase_c_y is None else phase_c_y.view(edge_count, 10 * 64),
        radial_scratch.flatten(),
    )


class NeoSO2BackwardWorkspace(_NeoSO2BackwardWorkspaceBase):
    """Lazily allocated SO2 backward scratch with lifetime-based slab reuse."""

    def __init__(
        self,
        torch: Any,
        *,
        edge_count: int,
        node_count: int,
        d_full: Any,
        dt_full: Any,
        radial: Any,
        phase_c_stack: Any | None = None,
        phase_c_y: Any | None = None,
        radial_scratch: Any | None = None,
        structural_memory_reuse: bool = False,
        phase_c_single_input_reuse: bool = False,
    ) -> None:
        opts = {"device": d_full.device, "dtype": d_full.dtype}
        d_values_per_edge = 1
        for size in d_full.shape[1:]:
            d_values_per_edge *= size
        radial_values_per_edge = 1
        for size in radial.shape[1:]:
            radial_values_per_edge *= size
        if d_values_per_edge > 10 * 64:
            raise ValueError("Neo SO2 grad_D does not fit the Phase-C scratch slab")

        if phase_c_single_input_reuse and not structural_memory_reuse:
            raise ValueError(
                "Phase-C single-input reuse requires structural memory reuse"
            )

        phase_c_stack_flat = None
        phase_c_y_flat = None
        radial_scratch_flat = None
        if structural_memory_reuse:
            phase_c_stack_flat, phase_c_y_flat, radial_scratch_flat = (
                _structural_memory_views(
                    torch,
                    edge_count=edge_count,
                    like=d_full,
                    phase_c_stack=phase_c_stack,
                    phase_c_y=phase_c_y,
                    radial_scratch=radial_scratch,
                    radial_values_per_edge=radial_values_per_edge,
                    phase_c_single_input_reuse=phase_c_single_input_reuse,
                )
            )

        self._phase_c_or_d = (
            phase_c_stack_flat
            if phase_c_stack_flat is not None
            else torch.empty(
                edge_count,
                10 * 64,
                device=opts["device"],
                dtype=opts["dtype"],
            )
        )
        self.grad_stack_focus = self._phase_c_or_d.view(edge_count, 2, 10, 32)
        self.grad_d = self._phase_c_or_d.flatten()[
            : edge_count * d_values_per_edge
        ].view_as(d_full)
        self.grad_dt = torch.empty_like(dt_full)

        # Gate backward completes before radial backward writes the final radial grad.
        if radial_scratch_flat is not None:
            self._gate_or_radial = radial_scratch_flat
        else:
            self._gate_or_radial = torch.empty(
                edge_count,
                radial_values_per_edge,
                device=opts["device"],
                dtype=opts["dtype"],
            )
        self.grad_gate_logits = None
        self.grad_radial_flat = self._gate_or_radial.flatten()[
            : edge_count * radial_values_per_edge
        ].view(edge_count, radial_values_per_edge)

        # The per-layer gate output is dead before radial backward writes grad_x_rot.
        self._gate_or_x_rot = (
            phase_c_stack_flat
            if phase_c_stack_flat is not None
            else torch.empty(
                edge_count,
                10 * 64,
                device=opts["device"],
                dtype=opts["dtype"],
            )
        )
        self.grad_y = self._gate_or_x_rot.view(edge_count * 2, 10 * 32)
        self.grad_x_rot = self._gate_or_x_rot
        if phase_c_stack_flat is None:
            self.grad_mixed_slab = None
        elif phase_c_y_flat is not None:
            self.grad_mixed_slab = phase_c_y_flat.view(edge_count, 2, 10, 32)
        else:
            self.grad_mixed_slab = torch.empty(
                edge_count,
                2,
                10,
                32,
                device=opts["device"],
                dtype=opts["dtype"],
            )

        self.grad_focus_alpha = torch.empty(
            edge_count, 2, device=opts["device"], dtype=opts["dtype"]
        )
        self.grad_alpha = torch.empty(
            edge_count, 2, device=opts["device"], dtype=opts["dtype"]
        )
        self.grad_logits = torch.empty(
            edge_count, 2, device=opts["device"], dtype=opts["dtype"]
        )
        self.grad_edge = torch.empty(
            edge_count, device=opts["device"], dtype=opts["dtype"]
        )
        self.grad_z_partial = torch.empty(
            node_count, 2, device=opts["device"], dtype=opts["dtype"]
        )
        self.grad_z = torch.empty(2, device=opts["device"], dtype=opts["dtype"])
        self.grad_x_wide_phase_a = torch.empty(
            node_count,
            16 * 64,
            device=opts["device"],
            dtype=opts["dtype"],
        )


class NeoFullCuteBackward:
    def __init__(
        self,
        torch: Any,
        block: Any,
        record: Any,
        x: Any,
        d_full: Any,
        dt_full: Any,
        radial_feat: Any,
        dst_ptr: Any,
        source_order: Any,
        source_ptr: Any,
        *,
        runtime_config: Any,
    ) -> None:
        from .kernels.envelope_softmax import (
            compile_envelope_softmax_forward,
        )
        from .kernels.phase_a_radial_forward import (
            run_neo_phase_a_radial_forward_packed_direct,
        )
        from .linear import (
            run_neo_so2_linear_manual,
        )
        from .wigner_layout import (
            PACKED_VALUE_COUNT,
        )

        self.torch = torch
        self.block = block
        self.so2 = block.so2_conv
        self.record = record
        self.config = runtime_config
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        compute_capability = tuple(torch.cuda.get_device_capability(device_index))
        self.device_index = device_index
        self.compute_capability = compute_capability
        self.packed_message_grid = _uses_packed_message_grid(compute_capability)
        self.compile_identity = (device_index, *compute_capability)
        _validate_runtime_config(
            runtime_config,
            compute_capability=compute_capability,
        )
        self.x = x
        self.d = d_full
        self.dt = dt_full
        expected_shape = (
            record.edge_cache.src.numel(),
            PACKED_VALUE_COUNT,
        )
        if (
            tuple(d_full.shape) != expected_shape
            or tuple(dt_full.shape) != expected_shape
        ):
            raise ValueError(
                f"packed Neo SO2 Wigner tensors must have shape {expected_shape}"
            )
        if d_full.data_ptr() != dt_full.data_ptr():
            raise ValueError("packed Neo SO2 D and Dt must share one storage")
        self.radial = radial_feat
        self.dst_ptr_i32 = dst_ptr.to(device=x.device, dtype=torch.int32).contiguous()
        self.edge_count = record.edge_cache.src.numel()
        self.node_count = x.shape[0]
        self.src_i32 = record.edge_cache.src.to(torch.int32).contiguous()
        self.src_i64 = record.edge_cache.src.contiguous()
        self.dst_i32 = record.edge_cache.dst.to(torch.int32).contiguous()
        self.dst_i64 = record.edge_cache.dst.contiguous()
        if (
            source_order.numel() == self.edge_count
            and source_ptr.numel() == self.node_count + 1
        ):
            self.source_order_i32 = source_order.to(
                device=x.device,
                dtype=torch.int32,
            ).contiguous()
            self.source_ptr_i32 = source_ptr.to(
                device=x.device,
                dtype=torch.int32,
            ).contiguous()
        else:
            from .radial_phase_a import (
                build_source_csr,
            )

            source_csr = build_source_csr(self.src_i32, self.node_count)
            self.source_order_i32 = source_csr.source_order
            self.source_ptr_i32 = source_csr.source_ptr
        self.rotate = self.so2.rotate_inv_rescale_full.contiguous()
        self.edge_gate = _edge_gate(torch, record.edge_cache)
        self.focus_norm_enabled = bool(self.so2.focus_norm)
        if self.focus_norm_enabled:
            focus_norm = self.so2.focus_compete_norm
            self.focus_norm_eps = float(focus_norm.eps)
            self.focus_norm_scale = focus_norm.adam_scale.detach().float().contiguous()
        else:
            self.focus_norm_eps = 0.0
            # The no-norm specialization does not read this fixed-ABI argument;
            # reuse an existing shape-compatible tensor instead of allocating one.
            self.focus_norm_scale = (
                self.so2.attn_qk_norm.adam_scale.detach().float().contiguous()
            )
        from .kernels.focus_source_backward import (
            compile_neo_attention_prelude_forward,
        )

        with torch.cuda.device(self.device_index):
            self.attention_prelude_forward = compile_neo_attention_prelude_forward(
                self.focus_norm_eps,
                float(self.so2.attn_qk_norm.eps),
                float(self.so2.focus_softmax_tau),
                float(self.so2.focus_label_smoothing),
                self.compile_identity,
                use_focus_norm=self.focus_norm_enabled,
            )
        from .kernels.qk_edge import (
            compile_neo_qk_edge_backward,
            compile_neo_qk_edge_forward,
        )

        qk_scale = float(self.so2.head_dim**-0.5)
        with torch.cuda.device(self.device_index):
            self.qk_edge_forward = compile_neo_qk_edge_forward(
                qk_scale,
                self.compile_identity,
            )
            self.qk_edge_backward = compile_neo_qk_edge_backward(
                qk_scale,
                self.compile_identity,
            )
        from .kernels.qk_edge import (
            compile_neo_qk_node_input_adjoint,
        )

        with torch.cuda.device(self.device_index):
            self.qk_node_input_adjoint = compile_neo_qk_node_input_adjoint(
                float(self.so2.attn_qk_norm.eps),
                self.compile_identity,
            )
        from .phase_c import (
            CuteNeoPhaseCBackwardLayout,
        )

        self.phase_c_layout_backward = CuteNeoPhaseCBackwardLayout(
            focus_eps=self.focus_norm_eps,
            focus_tau=float(self.so2.focus_softmax_tau),
            focus_label_smoothing=float(self.so2.focus_label_smoothing),
            use_focus_norm=self.focus_norm_enabled,
        )
        with torch.cuda.device(self.device_index):
            self.softmax_fwd = compile_envelope_softmax_forward(
                128,
                float(self.so2.eps),
            )
        self.structural_gate_forward = None
        self.structural_gate_backward = None
        self.combined_gate_backward = None
        self._focus_major_gate_linear_forward = None
        self._focus_major_gate_linear_backward_add = None
        self._run_structural_gate_forward = None
        self._run_structural_gate_backward = None
        if runtime_config.combined_so2_gate:
            from .kernels.gate_linear_residual_backward import (
                compile_neo_gate_linear_residual_backward_fused,
            )

            with torch.cuda.device(self.device_index):
                self.combined_gate_backward = (
                    compile_neo_gate_linear_residual_backward_fused()
                )
        elif not runtime_config.native_sm90_path:
            from .kernels.structural_gate_sm80 import (
                compile_neo_gate_split_structural_vec4_sm80_backward,
                compile_neo_gate_split_structural_vec4_sm80_forward,
            )
            from .structural_gate import (
                focus_major_gate_linear_backward_add_,
                focus_major_gate_linear_forward,
                run_structural_gate_backward,
                run_structural_gate_forward,
            )

            self.structural_gate_forward = (
                compile_neo_gate_split_structural_vec4_sm80_forward(
                    self.compile_identity,
                )
            )
            self.structural_gate_backward = (
                compile_neo_gate_split_structural_vec4_sm80_backward(
                    self.compile_identity,
                )
            )
            self._focus_major_gate_linear_forward = focus_major_gate_linear_forward
            self._focus_major_gate_linear_backward_add = (
                focus_major_gate_linear_backward_add_
            )
            self._run_structural_gate_forward = run_structural_gate_forward
            self._run_structural_gate_backward = run_structural_gate_backward
        opts = {"device": x.device, "dtype": x.dtype}
        self._backward_workspace = None
        self.alpha = torch.empty(
            self.edge_count, 2, device=opts["device"], dtype=opts["dtype"]
        )
        self.group_max = torch.empty(
            self.node_count, 2, device=opts["device"], dtype=opts["dtype"]
        )
        self.denom = torch.empty(
            self.node_count, 2, device=opts["device"], dtype=opts["dtype"]
        )

        self._run_cute_phase_a_radial_forward = (
            run_neo_phase_a_radial_forward_packed_direct
        )
        self._run_neo_so2_linear_manual = run_neo_so2_linear_manual

        self.combined_radial = _combined_radial_weight(
            torch, self.so2.radial_hidden_proj, self.so2.radial_degree_mixer
        )
        self.combined_attention_radial = _combined_attention_radial_weight(
            torch,
            self.so2.radial_hidden_proj,
            self.so2.adamw_attn_logit_w,
        )
        self.batched_radial_projection_weight = _batched_radial_projection_weight(
            self.so2.radial_degree_mixer,
            self.combined_radial,
            self.combined_attention_radial,
        )

        self._build_forward_graph()

    def ensure_backward_workspace(
        self,
    ) -> NeoSO2BackwardWorkspace:
        if self._backward_workspace is None:
            radial_scratch = getattr(self, "structural_scratch", None)
            if radial_scratch is None:
                radial_scratch = next(
                    (
                        cache.logits
                        for cache in self.stack_caches
                        if cache.logits is not None
                    ),
                    None,
                )
            if radial_scratch is None:
                raise RuntimeError(
                    "Neo SO2 backward requires a reusable radial scratch buffer; "
                    "no stack layer stored gate logits"
                )
            if self.phase_c_stack.device.type == "cuda":
                stream = self.torch.cuda.current_stream(self.phase_c_stack.device)
                self.phase_c_stack.record_stream(stream)
                radial_scratch.record_stream(stream)
            workspace = NeoSO2BackwardWorkspace(
                self.torch,
                edge_count=self.edge_count,
                node_count=self.node_count,
                d_full=self.d,
                dt_full=self.dt,
                radial=self.radial,
                phase_c_stack=self.phase_c_stack,
                phase_c_y=None,
                radial_scratch=radial_scratch,
                structural_memory_reuse=True,
                phase_c_single_input_reuse=True,
            )
            workspace.attach_to(self)
            self._backward_workspace = workspace
        return self._backward_workspace

    def _build_forward_graph(self) -> None:
        torch = self.torch
        so2 = self.so2
        block = self.block
        n_node = self.node_count
        n_edge = self.edge_count
        self.structural_scratch = None
        if self.config.combined_so2_gate:
            self.structural_scratch = self.x.new_empty(self.radial.shape)

        use_full_node = block.node_lmax == block.lmax
        self.use_full_node = use_full_node
        x_so2 = self.x if use_full_node else self.x[:, : block.mp_ebed_dim, :, :]
        x_pre = block.pre_so2_norm(x_so2)
        self.x_wide = so2.pre_focus_mix(
            x_pre.reshape(n_node, x_so2.shape[1], block.channels).unsqueeze(2)
        ).squeeze(2)

        cur, rad_l0, radial_compact = self._run_cute_phase_a_radial_forward(
            radial_hidden_proj=so2.radial_hidden_proj,
            radial_degree_mixer=so2.radial_degree_mixer,
            x_wide=self.x_wide.detach(),
            src=self.src_i32,
            D_full=self.d.detach(),
            radial_feat_m0=self.radial.detach(),
        )
        self.radial_compact = radial_compact.detach()

        self.focus_gate_src = cur[:, :, 0, :].detach().contiguous()
        self.stack_caches: list[StackCache] = []

        if self.config.combined_so2_gate:
            from .kernels.combined_gate_forward import (
                CuteNeoSO2GateCombinedFwdRunner,
                prepare_neo_so2_gate_combined_weights,
            )
            from .linear import (
                cached_neo_so2_linear_weights,
            )

        stack_layers = zip(
            so2.so2_linears,
            so2.so2_inter_norms,
            so2.non_linearities,
            strict=True,
        )
        for layer_idx, (so2_linear, _inter_norm, non_linear) in enumerate(stack_layers):
            x_layer = cur.detach()
            final = layer_idx == so2.mixing_layers - 1
            logits = None
            if not final and self.config.combined_so2_gate:
                w0, wpair = cached_neo_so2_linear_weights(so2_linear)
                gate_parameter = non_linear.gate_linear.weight
                gate_weight = gate_parameter.detach().view(32, 2, 3 * 32).contiguous()
                pack_key = (
                    so2_linear.weight_m0.data_ptr(),
                    so2_linear.weight_m0._version,
                    so2_linear.weight_m[0].data_ptr(),
                    so2_linear.weight_m[0]._version,
                    gate_parameter.data_ptr(),
                    gate_parameter._version,
                    x_layer.device,
                )
                pack_cache = getattr(
                    so2_linear,
                    "_deepmd_cute_neo_combined_gate_weights",
                    None,
                )
                if (
                    not isinstance(pack_cache, tuple)
                    or len(pack_cache) != 6
                    or pack_cache[0] != pack_key
                ):
                    with torch.cuda.device(x_layer.device):
                        pack_stream = torch.cuda.current_stream(x_layer.device)
                        packed_weights = prepare_neo_so2_gate_combined_weights(
                            w0,
                            wpair,
                            gate_weight,
                        )
                        ready_event = torch.cuda.Event()
                        ready_event.record(pack_stream)
                    packed_weights_ready = (
                        ready_event,
                        pack_stream.cuda_stream,
                    )
                    pack_cache = (
                        pack_key,
                        packed_weights,
                        packed_weights_ready,
                        # Retain source parameters so allocator pointer reuse
                        # cannot spoof the versioned cache key.
                        so2_linear.weight_m0,
                        so2_linear.weight_m[0],
                        gate_parameter,
                    )
                    so2_linear._deepmd_cute_neo_combined_gate_weights = pack_cache
                else:
                    packed_weights = pack_cache[1]
                    packed_weights_ready = pack_cache[2]
                y = torch.empty_like(x_layer)
                # Each CTA reads its residual tile before storing it, and the
                # m=0 and m>0 regions are disjoint. Reuse x_layer for output to
                # retain the optimized two-buffer stack footprint.
                combined_forward = CuteNeoSO2GateCombinedFwdRunner(
                    x_layer,
                    x_layer,
                    y,
                    x_layer,
                    packed_weights=packed_weights,
                    packed_weights_ready=packed_weights_ready,
                )
                cur = combined_forward()
            else:
                y = self._run_neo_so2_linear_manual(
                    so2_linear,
                    x_layer,
                    add_residual=final,
                    per_focus_pair=self.config.per_focus_so2_fwd_pair,
                )
            if not final and not self.config.combined_so2_gate:
                gate_src = y[:, :, 0, :]
                logits = self._focus_major_gate_linear_forward(
                    gate_src,
                    non_linear.gate_linear.weight.detach(),
                )
                self._run_structural_gate_forward(
                    self.structural_gate_forward,
                    x_layer,
                    y,
                    logits,
                    out=x_layer,
                )
                cur = x_layer
            elif final:
                self.phase_c_stack = y.detach()
                self.phase_c_y = None
                cur = None
            self.stack_caches.append(
                StackCache(
                    y=y,
                    logits=logits,
                    non_linear=non_linear,
                    final=final,
                )
            )

        x_wide_qk = self.x_wide.detach()
        rad_l0_qk = rad_l0.detach().view(n_edge, 2, 32)
        x_l0_node = x_wide_qk[:, 0, :].reshape(n_node, 2, 32)
        focus_alpha = torch.empty(
            n_edge,
            2,
            device=self.focus_gate_src.device,
            dtype=torch.float32,
        )
        q_node = torch.empty_like(
            x_l0_node,
            memory_format=torch.contiguous_format,
        )
        k_node = torch.empty_like(
            x_l0_node,
            memory_format=torch.contiguous_format,
        )
        self.attention_prelude_forward(
            self.focus_gate_src.view(n_edge, 64),
            x_l0_node.contiguous(),
            so2.adamw_focus_compete_w.detach().float().contiguous(),
            self.focus_norm_scale,
            so2.attn_q_proj.weight.detach().float().view(32, 2, 32).contiguous(),
            so2.attn_k_proj.weight.detach().float().view(32, 2, 32).contiguous(),
            so2.attn_qk_norm.adam_scale.detach().float().contiguous(),
            focus_alpha,
            q_node,
            k_node,
        )
        self.focus_alpha = focus_alpha.detach()
        self.q_node = q_node.detach()
        self.k_node = k_node.detach()
        self.attn_logits = torch.empty(
            n_edge,
            2,
            device=self.q_node.device,
            dtype=self.q_node.dtype,
        )
        self.qk_edge_forward(
            self.q_node,
            self.k_node,
            rad_l0_qk.contiguous(),
            so2.adamw_attn_logit_w.detach().contiguous(),
            self.src_i32,
            self.dst_i32,
            self.attn_logits,
        )

        self.softmax_fwd(
            self.attn_logits.detach().contiguous(),
            self.edge_gate,
            self.dst_ptr_i32,
            so2.adamw_attn_z_bias_raw.detach().reshape(2).float().contiguous(),
            self.alpha,
            self.group_max,
            self.denom,
        )
        self.attn_logits = None
        x_wide_down = self.x_wide.detach()
        from .kernels.phase_c_forward import (
            run_neo_phase_c_onepass_output_gate,
        )

        self.phase_c_out = run_neo_phase_c_onepass_output_gate(
            x_local_flat=self.phase_c_stack,
            Dt_full=self.dt.detach(),
            alpha_focus=self.alpha,
            focus_compete_alpha=self.focus_alpha,
            dst_ptr=self.dst_ptr_i32,
            rotate_inv_rescale=so2.rotate_inv_rescale_full,
            x_wide=x_wide_down,
            output_gate_norm_scale=so2.attn_output_gate_norm.adam_scale.detach()
            .float()
            .reshape(2, 32)
            .contiguous(),
            output_gate_weight=so2.adamw_attn_gate_w.detach()
            .float()
            .reshape(32, 2, 1)
            .contiguous(),
            output_gate_eps=float(so2.attn_output_gate_norm.eps),
        ).to(dtype=so2.compute_dtype)
        out = self.phase_c_out.detach().to(dtype=so2.dtype)
        self.out_gate_flat = out.detach()
        self.message_grid_product = None
        if so2.message_node_grid_product is not None:
            if self.packed_message_grid:
                from .message_grid import (
                    run_packed_message_grid_forward,
                )

                grid_out = run_packed_message_grid_forward(
                    so2.message_node_grid_product,
                    out,
                    x_wide_down,
                )
            else:
                grid_out = so2.message_node_grid_product(out, x_wide_down)
            out = out + grid_out
        self.post_mix_input = out.detach()
        out = so2.post_focus_mix(out.unsqueeze(2)).squeeze(2)
        self.post_norm_input = out.unsqueeze(2).detach()
        so2_out = block.post_so2_norm(self.post_norm_input)
        if use_full_node:
            self.final = so2_out
        else:
            final = self.x.new_zeros(self.x.shape)
            final[:, : block.mp_ebed_dim, :, :] = so2_out
            self.final = final
