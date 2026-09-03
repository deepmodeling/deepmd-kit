# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Opt-in CuTe Neo SO2 custom op for SeZM/DPA4 inference."""

from __future__ import (
    annotations,
)

import threading
import weakref
from dataclasses import (
    dataclass,
    field,
)
from functools import (
    lru_cache,
)
from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)

import torch
from torch import (
    Tensor,
)

from .. import (
    runtime_policy,
)
from .message_grid import (
    is_supported_message_grid,
)


@dataclass(frozen=True)
class NeoSO2RuntimeConfig:
    """Architecture-specific choices for the compact Neo SO(2) path."""

    native_sm90_path: bool = False
    per_focus_so2_fwd_pair: bool = False
    combined_so2_gate: bool = False


@dataclass(frozen=True)
class NeoSO2Spec:
    """Shape and branch contract for one Neo SO(2) unit replacement."""

    lmax: int
    node_lmax: int
    mmax: int
    full_dim: int
    reduced_dim: int
    channels: int
    n_focus: int
    focus_dim: int
    hidden_channels: int
    so2_layers: int
    n_atten_head: int
    radial_so2_mode: str
    radial_so2_rank: int
    so2_norm: bool
    focus_compete: bool
    message_node_so3: bool
    atten_f_mix: bool
    atten_v_proj: bool
    atten_o_proj: bool
    mlp_bias: bool
    layer_scale: bool
    use_so2_attn_res: bool
    has_pre_so2_norm: bool
    has_post_so2_norm: bool

    @property
    def is_current_neo_target(self) -> bool:
        return (
            self.lmax == 3
            and self.node_lmax == self.lmax
            and self.mmax == 1
            and self.full_dim == 16
            and self.reduced_dim == 10
            and self.channels == 32
            and self.n_focus == 2
            and self.focus_dim == 32
            and self.hidden_channels == 64
            and self.so2_layers == 3
            and self.n_atten_head == 1
            and self.radial_so2_mode == "degree_channel"
            and self.radial_so2_rank == 1
            and not self.so2_norm
            and self.focus_compete
            and self.message_node_so3
            and not self.atten_f_mix
            and not self.atten_v_proj
            and not self.atten_o_proj
            and not self.mlp_bias
            and not self.layer_scale
            and not self.use_so2_attn_res
            and not self.has_pre_so2_norm
            and self.has_post_so2_norm
        )


def _is_identity(module: Any) -> bool:
    return (
        module.__class__.__name__ == "Identity"
        or module.__class__.__name__ == "_Identity"
    )


def get_neo_so2_spec(block: Any) -> NeoSO2Spec:
    """Extract the SO2 contract from a SeZM interaction block."""
    so2 = block.so2_conv
    return NeoSO2Spec(
        lmax=int(so2.lmax),
        node_lmax=int(block.node_lmax),
        mmax=int(so2.mmax),
        full_dim=int(so2.ebed_dim_full),
        reduced_dim=int(so2.reduced_dim),
        channels=int(so2.channels),
        n_focus=int(so2.n_focus),
        focus_dim=int(so2.so2_focus_dim),
        hidden_channels=int(so2.hidden_channels),
        so2_layers=int(so2.mixing_layers),
        n_atten_head=int(so2.n_atten_head),
        radial_so2_mode=str(so2.radial_so2_mode),
        radial_so2_rank=int(so2.radial_so2_rank),
        so2_norm=bool(so2.so2_norm),
        focus_compete=bool(so2.focus_compete),
        message_node_so3=getattr(so2, "message_node_grid_product", None) is not None,
        atten_f_mix=bool(so2.atten_f_mix),
        atten_v_proj=getattr(so2, "attn_v_proj", None) is not None,
        atten_o_proj=getattr(so2, "attn_o_proj", None) is not None,
        mlp_bias=bool(so2.mlp_bias),
        layer_scale=bool(so2.layer_scale),
        use_so2_attn_res=bool(so2.use_so2_attn_res),
        has_pre_so2_norm=not _is_identity(block.pre_so2_norm),
        has_post_so2_norm=not _is_identity(block.post_so2_norm),
    )


def has_equivariant_rms_norm_contract(module: Any) -> bool:
    """Return whether a norm exposes the exact state consumed by Neo SO2.

    Parameters
    ----------
    module : Any
        Candidate equivariant normalization module.

    Returns
    -------
    bool
        Whether the module's layout and state match the fused kernel contract.
    """
    return (
        type(module).__name__ == "EquivariantRMSNorm"
        and int(getattr(module, "lmax", -1)) == 3
        and int(getattr(module, "channels", -1)) == 32
        and int(getattr(module, "n_focus", -1)) == 1
        and tuple(getattr(getattr(module, "adam_scale", None), "shape", ()))
        == (4, 1, 32)
        and tuple(getattr(getattr(module, "bias", None), "shape", ())) == (1, 32)
        and tuple(getattr(getattr(module, "balance_weight", None), "shape", ()))
        == (16,)
        and tuple(getattr(getattr(module, "expand_index", None), "shape", ())) == (16,)
        and hasattr(module, "eps")
    )


def is_supported_neo_so2_block(block: Any) -> bool:
    """Return whether this block can use the current Neo CuTe SO2 path."""
    so2 = block.so2_conv
    return (
        get_neo_so2_spec(block).is_current_neo_target
        and bool(so2.focus_norm)
        and not bool(so2.edge_cartesian)
        and getattr(so2, "node_cartesian_tp", None) is None
        and has_equivariant_rms_norm_contract(block.post_so2_norm)
        and is_supported_message_grid(so2.message_node_grid_product)
    )


def _module_floating_state_uses_strict_fp32(
    module: Any,
    *,
    require_floating_tensor: bool,
) -> bool:
    """Check live floating parameters and buffers without reading device data."""
    saw_floating_tensor = False
    for getter_name in ("parameters", "buffers"):
        getter = getattr(module, getter_name, None)
        if getter is None:
            return False
        for tensor in getter():
            if tensor.is_floating_point():
                saw_floating_tensor = True
                if tensor.dtype != torch.float32:
                    return False
    return saw_floating_tensor or not require_floating_tensor


def _module_uses_strict_fp32(module: Any) -> bool:
    return _module_floating_state_uses_strict_fp32(
        module,
        require_floating_tensor=True,
    )


def _module_is_frozen(module: Any) -> bool:
    """Return whether autograd cannot request gradients for module state."""
    parameters = getattr(module, "parameters", None)
    return parameters is not None and not any(
        parameter.requires_grad for parameter in parameters()
    )


def _tensor_is_aligned(tensor: Tensor, alignment: int = 16) -> bool:
    return tensor.data_ptr() % alignment == 0


def _module_state_is_aligned(module: Any, alignment: int = 16) -> bool:
    """Check CuTe's declared alignment contract for parameters and buffers."""
    return all(
        not tensor.is_floating_point() or _tensor_is_aligned(tensor, alignment)
        for getter_name in ("parameters", "buffers")
        for tensor in getattr(module, getter_name)()
    )


def _gate_expand_index_is_supported(block: Any) -> bool:
    """Check the degree-to-gate map assumed by fused Neo gate kernels."""
    non_linearities = getattr(
        getattr(block, "so2_conv", None),
        "non_linearities",
        None,
    )
    if non_linearities is None:
        return True
    buffers = tuple(
        expand_index
        for non_linear in non_linearities
        if (expand_index := getattr(non_linear, "expand_index", None)) is not None
        and expand_index.numel() > 0
    )
    signature = tuple(
        (
            tensor.data_ptr(),
            tensor._version,
            tensor.dtype,
            tensor.device,
            tuple(tensor.shape),
        )
        for tensor in buffers
    )
    cached = getattr(block, "_deepmd_cute_gate_expand_contract", None)
    if cached is not None and cached[0] == signature:
        return bool(cached[1])
    expected = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=torch.long,
        device="cpu",
    )
    supported = all(
        torch.equal(
            expand_index.detach().to(device="cpu", dtype=torch.long),
            expected,
        )
        for expand_index in buffers
    )
    block._deepmd_cute_gate_expand_contract = (signature, supported)
    return supported


def _gate_expand_index_structure_is_supported(block: Any) -> bool:
    """Check graph-visible gate-index metadata without reading tensor values."""
    non_linearities = getattr(
        getattr(block, "so2_conv", None),
        "non_linearities",
        None,
    )
    if non_linearities is None:
        return True
    return all(
        expand_index.dtype == torch.long and tuple(expand_index.shape) == (9,)
        for non_linear in non_linearities
        if (expand_index := getattr(non_linear, "expand_index", None)) is not None
        and expand_index.numel() > 0
    )


def _aligned_contiguous(tensor: Tensor, alignment: int = 16) -> Tensor:
    """Return canonical storage satisfying CuTe's assumed alignment."""
    if tensor.is_contiguous() and _tensor_is_aligned(tensor, alignment):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _producer_modules_use_strict_fp32(modules: Any) -> bool:
    return all(
        _module_floating_state_uses_strict_fp32(
            module,
            require_floating_tensor=False,
        )
        for module in modules
    )


def _dtypes_use_strict_fp32(dtypes: Any) -> bool:
    return all(dtype == torch.float32 for dtype in dtypes)


def is_supported_so2_compute_capability(
    compute_capability: tuple[int, int],
) -> bool:
    """Return whether SO2 supports this compute capability."""
    return runtime_policy.is_supported_so2_capability(compute_capability)


def _device_compute_capability(device: torch.device) -> tuple[int, int]:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _cuda_compute_capability(device_index)


def _tensor_compute_capability(tensor: Any) -> tuple[int, int] | None:
    """Resolve dispatch capability from an operand instead of global CUDA state."""
    device = getattr(tensor, "device", None)
    if device is None or device.type != "cuda":
        return None
    return _device_compute_capability(device)


def _device_is_supported_for_so2(device: torch.device) -> bool:
    # Metadata-only eligibility checks may use a CUDA device on a host without
    # a CUDA runtime. Concrete CUDA dispatch always validates the capability.
    return not torch.cuda.is_available() or is_supported_so2_compute_capability(
        _device_compute_capability(device)
    )


def packed_wigner_edges_eligible(
    *,
    candidate: bool,
    edge_count: int,
    node_count: int,
    destinations_sorted: bool,
    runtime_dtypes: Any = (),
) -> bool:
    """Finish packed eligibility from host-side edge-order provenance."""
    return (
        candidate
        and edge_count > 0
        and destinations_sorted
        and _dtypes_use_strict_fp32(runtime_dtypes)
        and runtime_policy.so2_int32_indexing_is_safe(
            edge_count=edge_count,
            node_count=node_count,
        )
    )


def is_neo_so2_static_eligible(
    block: Any,
    *,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    """Check SO2 conditions that are stable during one descriptor forward."""
    return (
        runtime_policy.is_cute_infer_enabled()
        and not training
        and not bool(getattr(block, "training", False))
        and device.type == "cuda"
        and _device_is_supported_for_so2(device)
        and dtype == torch.float32
        and not torch.is_autocast_enabled(device.type)
        and runtime_policy.uses_strict_fp32_matmul()
        and _module_uses_strict_fp32(block)
        and _module_is_frozen(block)
        and _module_state_is_aligned(block)
        and _gate_expand_index_structure_is_supported(block)
        and getattr(block, "_deepmd_cute_so2_state", None) is not False
        and is_supported_neo_so2_block(block)
    )


def is_neo_so2_runtime_eligible(
    block: Any,
    *,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
    edge_count: int,
    node_count: int,
    destinations_sorted: bool,
) -> bool:
    """Return the exact strict-FP32 inference contract for SO2 dispatch."""
    return (
        edge_count > 0
        and destinations_sorted
        and is_neo_so2_static_eligible(
            block,
            training=training,
            device=device,
            dtype=dtype,
        )
    )


def is_packed_wigner_candidate(
    *,
    blocks: Any,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
    producer_modules: Any = (),
    producer_dtypes: Any = (),
    has_edge_src_gate: bool = False,
) -> bool:
    """Check packed-Wigner conditions known before edge construction."""
    if has_edge_src_gate:
        # SO2 currently falls back for SFPG so eager must receive dense Wigner data.
        return False
    block_tuple = tuple(blocks)
    if device.type == "cuda":
        try:
            compute_capability = _device_compute_capability(device)
        except (AssertionError, RuntimeError):
            packed_wigner_enabled = False
        else:
            packed_wigner_enabled = (
                runtime_policy.is_cute_infer_enabled()
                and runtime_policy.is_supported_so2_capability(compute_capability)
            )
    else:
        packed_wigner_enabled = False
    if (
        not block_tuple
        or not packed_wigner_enabled
        or not _producer_modules_use_strict_fp32(producer_modules)
        or not _dtypes_use_strict_fp32(producer_dtypes)
        or torch.is_autocast_enabled(device.type)
    ):
        return False
    return all(
        is_neo_so2_static_eligible(
            block,
            training=training,
            device=device,
            dtype=dtype,
        )
        for block in block_tuple
    )


class _RegistryEntry:
    """Weakly retain a block so discarded models do not leak the registry."""

    def __init__(
        self,
        block: Any,
        config: Any,
        *,
        on_collect: Any | None = None,
    ) -> None:
        try:
            self._block_ref = weakref.ref(block, on_collect)
        except TypeError:
            self._block_ref = lambda: block
        self.config = config

    @property
    def block(self) -> Any:
        block = self._block_ref()
        if block is None:
            raise RuntimeError("the registered Neo SO2 block has been released")
        return block


@dataclass(frozen=True)
class _RegisteredSO2State:
    device_index: int
    handle: int
    config: NeoSO2RuntimeConfig


@dataclass
class _RunnerState:
    runner: Any | None
    backward_calls: int = 0
    reservation_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )


_REGISTRY: dict[int, _RegistryEntry] = {}
_NEXT_HANDLE = 1
_REGISTRY_LOCK = threading.Lock()
_PACKED_RUNNER_CACHE: dict[tuple[str, int | None, int], _RunnerState] = {}
_PACKED_RUNNER_CACHE_LOCK = threading.Lock()


def _runner_compile_identity(runner: Any) -> tuple[int, int, int]:
    """Return the runner's device-specific CuTe compilation identity."""
    return getattr(runner, "compile_identity", (-1, 0, 0))


def _compile_on_runner_device(
    compile_identity: tuple[int, int, int],
    compiler: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Compile under the CUDA device represented by the cache key."""
    device_index = int(compile_identity[0])
    if device_index < 0:
        return compiler(*args, **kwargs)
    with torch.cuda.device(device_index):
        return compiler(*args, **kwargs)


@lru_cache(maxsize=8)
def _compile_output_gate_backward(
    compile_identity: tuple[int, int, int], eps: float
) -> Any:
    from .kernels.output_gate_backward import (
        compile_neo_output_gate_backward,
    )

    return _compile_on_runner_device(
        compile_identity,
        compile_neo_output_gate_backward,
        eps,
    )


def register_cute_so2_block(block: Any, config: Any) -> int:
    """Register a DeePMD block/config pair and return a stable integer handle."""
    global _NEXT_HANDLE

    def remove_collected_entry(block_ref: weakref.ReferenceType[Any]) -> None:
        with _REGISTRY_LOCK:
            entry = _REGISTRY.get(handle)
            if entry is not None and entry._block_ref is block_ref:
                _REGISTRY.pop(handle, None)

    with _REGISTRY_LOCK:
        handle = _NEXT_HANDLE
        _NEXT_HANDLE += 1
        _REGISTRY[handle] = _RegistryEntry(
            block=block,
            config=config,
            on_collect=remove_collected_entry,
        )
    return handle


def invalidate_cute_so2_state(block: Any) -> None:
    """Release a block's registered CuTe state after its modules change."""
    state = getattr(block, "_deepmd_cute_so2_state", None)
    if isinstance(state, _RegisteredSO2State):
        with _REGISTRY_LOCK:
            _REGISTRY.pop(state.handle, None)
    if hasattr(block, "_deepmd_cute_so2_state"):
        delattr(block, "_deepmd_cute_so2_state")
    if hasattr(block, "_deepmd_cute_gate_expand_contract"):
        delattr(block, "_deepmd_cute_gate_expand_contract")


def _validate_gate_expand_index(block: Any) -> None:
    """Pin the degree-to-gate map assumed by fused Neo gate kernels."""
    if not _gate_expand_index_is_supported(block):
        raise ValueError(
            "Neo SO2 fused gate kernels require expand_index=[0,1,2,0,1,2,0,1,2]"
        )


@torch.compiler.disable
def _register_cute_so2_state(
    block: Any,
    device_index: int,
    config: NeoSO2RuntimeConfig,
) -> _RegisteredSO2State | None:
    """Publish cold SO2 state eagerly before its handle enters a custom op."""
    _validate_gate_expand_index(block)
    old_state = getattr(block, "_deepmd_cute_so2_state", None)
    if isinstance(old_state, _RegisteredSO2State):
        with _REGISTRY_LOCK:
            old_entry = _REGISTRY.get(old_state.handle)
            if (
                old_state.device_index == device_index
                and old_state.config == config
                and old_entry is not None
                and old_entry.block is block
            ):
                return old_state
            _REGISTRY.pop(old_state.handle, None)
    if not _module_state_is_aligned(block):
        # Module parameters and buffers are frozen for this inference path, so
        # cache a failed static contract until explicit state invalidation.
        block._deepmd_cute_so2_state = False
        return None
    state = _RegisteredSO2State(
        device_index=device_index,
        handle=register_cute_so2_block(block, config),
        config=config,
    )
    block._deepmd_cute_so2_state = state
    return state


@torch.compiler.disable
def prepare_cute_so2_blocks(
    blocks: Any,
    *,
    training: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> bool:
    """Validate and register SO2 state before model graph capture begins."""
    if training or device.type != "cuda" or dtype != torch.float32:
        return False
    block_tuple = tuple(blocks)
    if not block_tuple:
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compute_capability = _cuda_compute_capability(device_index)
    if not is_supported_so2_compute_capability(compute_capability):
        return False
    config = _architecture_default_config(compute_capability)
    if not all(
        is_neo_so2_static_eligible(
            block,
            training=training,
            device=device,
            dtype=dtype,
        )
        for block in block_tuple
    ):
        return False
    return all(
        _register_cute_so2_state(block, device_index, config) is not None
        for block in block_tuple
    )


def _runner_token_key(
    runner_token: Tensor, *, path: str
) -> tuple[str, int | None, int]:
    if runner_token.dtype != torch.uint8 or runner_token.numel() != 1:
        raise ValueError(f"{path} Neo SO2 runner token must be one uint8 value")
    return (
        runner_token.device.type,
        runner_token.device.index,
        int(runner_token.data_ptr()),
    )


def _packed_runner_key(runner_token: Tensor) -> tuple[str, int | None, int]:
    return _runner_token_key(runner_token, path="packed")


def _release_packed_runner(
    key: tuple[str, int | None, int],
    state: _RunnerState,
) -> None:
    with _PACKED_RUNNER_CACHE_LOCK:
        if _PACKED_RUNNER_CACHE.get(key) is state:
            _PACKED_RUNNER_CACHE.pop(key, None)


def _store_packed_runner(runner_token: Tensor, runner: Any) -> None:
    key = _packed_runner_key(runner_token)
    state = _RunnerState(runner=runner)
    with _PACKED_RUNNER_CACHE_LOCK:
        old_state = _PACKED_RUNNER_CACHE.get(key)
        if old_state is not None:
            raise RuntimeError("packed Neo SO2 runner token is already outstanding")
        else:
            _PACKED_RUNNER_CACHE[key] = state
    weakref.finalize(runner_token, _release_packed_runner, key, state)


def _borrow_packed_runner(runner_token: Tensor) -> Any | None:
    key = _packed_runner_key(runner_token)
    with _PACKED_RUNNER_CACHE_LOCK:
        state = _PACKED_RUNNER_CACHE.get(key)
    if state is None:
        raise RuntimeError("packed Neo SO2 runner token is not outstanding")
    with state.reservation_lock:
        # Transfer the forward runner once; retained VJPs rebuild isolated workspaces.
        runner = state.runner
        state.runner = None
        state.backward_calls += 1
    return runner


def _edge_src_gate_arg(edge_src_gate: Tensor) -> Tensor | None:
    return None if edge_src_gate.numel() == 0 else edge_src_gate


def _layout_like(tensor: Tensor, like: Tensor) -> Tensor:
    if tensor.shape == like.shape and tensor.stride() == like.stride():
        return tensor
    out = torch.empty_strided(
        like.shape,
        like.stride(),
        device=like.device,
        dtype=like.dtype,
    )
    out.copy_(tensor)
    return out


def _grad_layout_like(tensor: Tensor, like: Tensor, *, skip: bool) -> Tensor:
    if skip:
        return tensor
    return _layout_like(tensor, like)


def _assert_grad_meta_contract(
    actual: Tensor,
    expected_like: Tensor,
    *,
    name: str,
    expected_stride: tuple[int, ...] | None = None,
) -> None:
    """Fail before AOT consumes a gradient that contradicts register_fake."""
    stride = expected_like.stride() if expected_stride is None else expected_stride
    if (
        actual.shape != expected_like.shape
        or actual.dtype != expected_like.dtype
        or actual.device != expected_like.device
        or actual.stride() != stride
    ):
        raise RuntimeError(
            f"Neo SO2 {name} gradient violates the custom-op meta contract: "
            f"got shape={tuple(actual.shape)} stride={actual.stride()}, expected "
            f"shape={tuple(expected_like.shape)} stride={stride}"
        )


def _assert_not_cuda_graph_capturing(path: str, tensor: Tensor) -> None:
    """Reject stateful paths that cannot be safely captured and replayed."""
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"Neo SO2 {path} cannot run inside direct CUDA graph capture; "
            "use torch.compile without wrapping this stateful custom op in "
            "torch.cuda.graph"
        )


def _fake_x_wide_grad_like(x: Tensor, *, skip: bool) -> Tensor:
    """Fake the native SO2 grad-x layout when layout restore is skipped.

    The manual SO3 pre-mix backward returns a degree-major view with shape
    ``(N, D, 1, C)`` and stride ``(C, N*C, C, 1)``.  If the fake kernel
    promises ``empty_like(x)`` instead, AOTAutograd emits stride assertions for
    a contiguous ``(N, D, 1, C)`` gradient and rejects the runtime result.
    """
    if not skip:
        return torch.empty_like(x)
    if x.ndim != 4:
        return torch.empty_like(x)
    n_node = x.shape[0]
    channels = x.shape[-1]
    return torch.empty_strided(
        x.shape,
        (channels, n_node * channels, channels, 1),
        device=x.device,
        dtype=x.dtype,
    )


def _equivariant_rmsnorm_backward(norm: Any, x: Tensor, grad_out: Tensor) -> Tensor:
    # Matches EquivariantRMSNorm.forward for the inference case. Parameters are
    # frozen; only the input gradient is needed for forces/stress.
    in_dtype = x.dtype
    norm_dtype = norm.adam_scale.dtype
    xf = x.to(dtype=norm_dtype)
    gf = grad_out.to(dtype=norm_dtype)
    x0_in = xf[:, :1, :, :]
    xt = xf[:, 1:, :, :]
    x0 = x0_in - x0_in.mean(dim=-1, keepdim=True)

    mean_variance = x0.square().sum(dim=(1, 3)) * norm.balance_weight[0]
    if xt.numel() > 0:
        mean_variance = mean_variance + torch.einsum(
            "ndfc,d->nf", xt * xt, norm.balance_weight[1:]
        )
    eps = getattr(norm, "eps_tensor", norm.eps)
    inv = torch.rsqrt(mean_variance + eps).unsqueeze(1).unsqueeze(-1)
    expanded_scale = torch.index_select(
        norm.adam_scale, dim=0, index=norm.expand_index
    ).unsqueeze(0)

    grad_pre = gf * expanded_scale
    grad_x0 = grad_pre[:, :1, :, :]
    grad_xt = grad_pre[:, 1:, :, :]

    dinv = (grad_x0 * x0).sum(dim=(1, 3), keepdim=True)
    if xt.numel() > 0:
        dinv = dinv + (grad_xt * xt).sum(dim=(1, 3), keepdim=True)
    dvar = -0.5 * dinv * inv.pow(3)

    grad_centered = grad_x0 * inv + dvar * (2.0 * norm.balance_weight[0] * x0)
    grad_x0_in = grad_centered - grad_centered.mean(dim=-1, keepdim=True)
    if xt.numel() == 0:
        return grad_x0_in.to(dtype=in_dtype)
    grad_xt_in = grad_xt * inv + dvar * (
        2.0 * norm.balance_weight[1:].view(1, -1, 1, 1) * xt
    )
    return torch.cat([grad_x0_in, grad_xt_in], dim=1).to(dtype=in_dtype)


def _so3_linear_backward_input(linear: Any, x: Tensor, grad_out: Tensor) -> Tensor:
    del x
    weight = linear.weight.view(
        linear.lmax + 1,
        linear.in_channels,
        linear.n_focus,
        linear.out_channels,
    )
    weight_expanded = torch.index_select(weight, dim=0, index=linear.expand_index)
    return torch.einsum("ndfo,difo->ndfi", grad_out, weight_expanded)


def _focus_linear_forward(linear: Any, x: Tensor) -> Tensor:
    weight = linear.weight.view(linear.in_channels, linear.n_focus, linear.out_channels)
    out = torch.einsum("bfi,ifo->bfo", x, weight)
    if linear.use_bias:
        out = out + linear.bias.view(linear.n_focus, linear.out_channels).unsqueeze(0)
    return out


def _focus_linear_backward_input(linear: Any, grad_out: Tensor) -> Tensor:
    weight = linear.weight.view(linear.in_channels, linear.n_focus, linear.out_channels)
    return torch.einsum("bfo,ifo->bfi", grad_out, weight)


def _swiglu_forward(x: Tensor) -> Tensor:
    gate, value = torch.chunk(x, chunks=2, dim=-1)
    return gate * torch.sigmoid(gate) * value


def _swiglu_backward_input(x: Tensor, grad_out: Tensor) -> Tensor:
    gate, value = torch.chunk(x, chunks=2, dim=-1)
    sig = torch.sigmoid(gate)
    grad_gate = grad_out * value * (sig + gate * sig * (1.0 - sig))
    grad_value = grad_out * gate * sig
    return torch.cat([grad_gate, grad_value], dim=-1)


def _frame_expand_forward(module: Any, coeff: Tensor) -> Tensor:
    weight = module.weight.index_select(0, module.degree_index)
    return torch.einsum("ndfi,dio->ndfo", coeff, weight)


def _frame_expand_backward_input(module: Any, grad_out: Tensor) -> Tensor:
    weight = module.weight.index_select(0, module.degree_index)
    return torch.einsum("ndfo,dio->ndfi", grad_out, weight)


def _frame_contract_backward_input(module: Any, grad_out: Tensor) -> Tensor:
    weight = module.weight.index_select(0, module.degree_index)
    return torch.einsum("ndfo,dio->ndfi", grad_out, weight)


def _neo_so2_linear_backward_input_with_residual(
    so2_linear: Any,
    grad_out: Tensor,
    residual: Tensor,
    *,
    inplace_residual: bool = False,
    out: Tensor | None = None,
) -> Tensor:
    from .linear import (
        cached_neo_so2_linear_weights,
    )

    w0, wpair = cached_neo_so2_linear_weights(so2_linear)
    cache = getattr(so2_linear, "_deepmd_cute_neo_manual_weights_t", None)
    cache_key = (w0.data_ptr(), wpair.data_ptr(), w0.dtype, w0.device)
    if (
        cache is None
        or cache[0] is not w0
        or cache[1] is not wpair
        or cache[2] != cache_key
    ):
        w0_t = w0.transpose(1, 2).contiguous()
        wpair_t = wpair.transpose(1, 2).contiguous()
        so2_linear._deepmd_cute_neo_manual_weights_t = (
            w0,
            wpair,
            cache_key,
            w0_t,
            wpair_t,
        )
    else:
        w0_t, wpair_t = cache[3], cache[4]
    if inplace_residual:
        if out is not None:
            raise ValueError("SO2 backward cannot select both in-place and out storage")
        from .linear import (
            neo_so2_linear_backward_residual_inplace,
        )

        return neo_so2_linear_backward_residual_inplace(
            residual,
            grad_out,
            w0_t,
            wpair_t,
        )
    if grad_out is residual:
        from .structural_gate import (
            focus_major_so2_backward_with_folded_residual_out,
        )

        folded_cache = getattr(
            so2_linear,
            "_deepmd_cute_neo_manual_folded_weights_t",
            None,
        )
        folded_key = (
            w0_t.data_ptr(),
            wpair_t.data_ptr(),
            w0_t.dtype,
            w0_t.device,
        )
        if (
            folded_cache is None
            or folded_cache[0] is not w0_t
            or folded_cache[1] is not wpair_t
            or folded_cache[2] != folded_key
        ):
            w0_folded_t = w0_t.clone()
            wpair_folded_t = wpair_t.clone()
            w0_folded_t.diagonal(dim1=-2, dim2=-1).add_(1.0)
            wpair_folded_t.diagonal(dim1=-2, dim2=-1).add_(1.0)
            folded_cache = (
                w0_t,
                wpair_t,
                folded_key,
                w0_folded_t,
                wpair_folded_t,
            )
            so2_linear._deepmd_cute_neo_manual_folded_weights_t = folded_cache
        if out is None:
            out = grad_out.new_empty(grad_out.shape)
        return focus_major_so2_backward_with_folded_residual_out(
            grad_out,
            folded_cache[3],
            folded_cache[4],
            out=out,
        )

    raise RuntimeError("Neo SO2 backward reached an unsupported storage pattern")


def _so3_grid_cross_glu_flat_backward(
    net: Any,
    query_flat: Tensor,
    context_flat: Tensor,
    grad_out_flat: Tensor,
) -> tuple[Tensor, Tensor]:
    """Input-gradient for Neo's flat cross SO3GridNet GLU branch."""
    if (
        net.layout != "flat"
        or net.mode != "cross"
        or net.op_type != "glu"
        or net.frame_expand is None
        or net.frame_contract is None
    ):
        raise NotImplementedError("manual grid backward supports Neo cross/flat/glu")

    q_dtype = query_flat.dtype
    c_dtype = context_flat.dtype
    n_batch, coeff_dim, _ = query_flat.shape
    n_focus = net.n_focus
    channels = net.channels
    n_frames = net.n_frames
    expanded = net.expanded_channels

    query = query_flat.reshape(n_batch, coeff_dim, n_focus, channels)
    context = context_flat.reshape(n_batch, coeff_dim, n_focus, channels)
    scalar_pair = torch.cat(
        [query[:, 0, :, :], context[:, 0, :, :]],
        dim=-1,
    ).to(dtype=net.dtype)

    left = _frame_expand_forward(net.frame_expand, query).to(dtype=net.dtype)
    right = _frame_expand_forward(net.frame_expand, context).to(dtype=net.dtype)
    left_view = left.reshape(n_batch, coeff_dim, n_focus, n_frames, channels)
    right_view = right.reshape(n_batch, coeff_dim, n_focus, n_frames, channels)
    to_grid = net.projector.to_grid_mat.reshape(
        net.projector.grid_size,
        coeff_dim,
        n_frames,
    )
    from_grid = net.projector.from_grid_mat.reshape(
        coeff_dim,
        n_frames,
        net.projector.grid_size,
    )
    left_grid = torch.einsum("gdk,ndfkc->ngfc", to_grid, left_view)
    right_grid = torch.einsum("gdk,ndfkc->ngfc", to_grid, right_view)
    coeff = torch.einsum("dkg,ngfc->ndfkc", from_grid, left_grid * right_grid)
    coeff_flat = coeff.reshape(n_batch, coeff_dim, n_focus, expanded)

    scalar_out = _swiglu_forward(scalar_pair)
    scalar_logits = _focus_linear_forward(net.scalar_gate, scalar_pair)
    scalar_gate = torch.sigmoid(scalar_logits)
    coeff_view = coeff_flat.reshape(n_batch, coeff_dim, n_focus, n_frames, channels)
    scalar_path = coeff_view * scalar_gate[:, None, :, None, :]
    scalar_path = scalar_path.clone()
    scalar_path[:, 0, :, net.frame_zero_index, :].add_(scalar_out)
    scalar_path_flat = scalar_path.reshape(n_batch, coeff_dim, n_focus, expanded)

    grad = grad_out_flat.reshape(n_batch, coeff_dim, n_focus, channels).to(
        dtype=net.dtype
    )
    if net.residual_scale is not None:
        grad = grad * net.residual_scale.reshape(1, 1, n_focus, channels)
    grad_scalar_flat = _frame_contract_backward_input(net.frame_contract, grad)
    grad_scalar_view = grad_scalar_flat.reshape(
        n_batch,
        coeff_dim,
        n_focus,
        n_frames,
        channels,
    )

    grad_coeff = grad_scalar_view * scalar_gate[:, None, :, None, :]
    grad_scalar_gate = (grad_scalar_view * coeff_view).sum(dim=(1, 3))
    grad_scalar_out = grad_scalar_view[:, 0, :, net.frame_zero_index, :]
    grad_scalar_logits = grad_scalar_gate * scalar_gate * (1.0 - scalar_gate)
    grad_scalar_pair = _focus_linear_backward_input(net.scalar_gate, grad_scalar_logits)
    grad_scalar_pair = grad_scalar_pair + _swiglu_backward_input(
        scalar_pair,
        grad_scalar_out,
    )

    grad_grid = torch.einsum("dkg,ndfkc->ngfc", from_grid, grad_coeff)
    grad_left_grid = grad_grid * right_grid
    grad_right_grid = grad_grid * left_grid
    grad_left = torch.einsum("gdk,ngfc->ndfkc", to_grid, grad_left_grid).reshape(
        n_batch,
        coeff_dim,
        n_focus,
        expanded,
    )
    grad_right = torch.einsum("gdk,ngfc->ndfkc", to_grid, grad_right_grid).reshape(
        n_batch,
        coeff_dim,
        n_focus,
        expanded,
    )
    grad_query = _frame_expand_backward_input(net.frame_expand, grad_left)
    grad_context = _frame_expand_backward_input(net.frame_expand, grad_right)
    grad_query[:, 0, :, :].add_(grad_scalar_pair[:, :, :channels])
    grad_context[:, 0, :, :].add_(grad_scalar_pair[:, :, channels:])
    del scalar_path_flat
    return (
        grad_query.reshape_as(query_flat).to(dtype=q_dtype),
        grad_context.reshape_as(context_flat).to(dtype=c_dtype),
    )


def _final_manual_backward(runner: Any, grad_out: Tensor) -> tuple[Tensor, Tensor]:
    so2 = runner.so2
    block = runner.block
    n_node = runner.node_count
    if runner.use_full_node:
        grad_so2_out = grad_out
    else:
        grad_so2_out = grad_out[:, : block.mp_ebed_dim, :, :]

    phase = runner.phase_c_out.detach()
    x_wide = runner.x_wide.detach()
    out_gate_flat = runner.out_gate_flat
    post_in = runner.post_mix_input.unsqueeze(2)
    post_norm_in = runner.post_norm_input

    grad_post_norm_in = _equivariant_rmsnorm_backward(
        block.post_so2_norm,
        post_norm_in,
        grad_so2_out,
    )
    grad_post_mix = _so3_linear_backward_input(
        so2.post_focus_mix,
        post_in,
        grad_post_norm_in.squeeze(2).unsqueeze(2),
    ).squeeze(2)

    if so2.message_node_grid_product is not None:
        if runner.packed_message_grid:
            message_grid_product = runner.message_grid_product
            runner.message_grid_product = None
            from .message_grid import (
                run_packed_message_grid_backward,
            )

            grad_out_gate_flat, grad_grid_context = run_packed_message_grid_backward(
                so2.message_node_grid_product,
                out_gate_flat,
                x_wide,
                grad_post_mix,
                product_flat=message_grid_product,
            )
            del message_grid_product
        else:
            grad_out_gate_flat, grad_grid_context = _so3_grid_cross_glu_flat_backward(
                so2.message_node_grid_product,
                out_gate_flat,
                x_wide,
                grad_post_mix,
            )
        grad_out_gate_flat.add_(grad_post_mix)
        grad_x_wide_down = torch.zeros(
            n_node,
            16 * 64,
            device=x_wide.device,
            dtype=x_wide.dtype,
        ).view(n_node, 16, 64)
        grad_x_wide_down.add_(grad_grid_context)
    else:
        grad_out_gate_flat = grad_post_mix
        grad_x_wide_down = torch.zeros(
            n_node,
            16 * 64,
            device=x_wide.device,
            dtype=x_wide.dtype,
        ).view(n_node, 16, 64)

    grad_phase = grad_out_gate_flat.contiguous()
    output_gate_backward = _compile_output_gate_backward(
        _runner_compile_identity(runner),
        float(so2.attn_output_gate_norm.eps),
    )
    output_gate_backward(
        grad_phase.view(n_node, 16 * 64),
        phase.contiguous().view(n_node, 16 * 64),
        x_wide.contiguous().view(n_node, 16 * 64),
        so2.attn_output_gate_norm.adam_scale.detach()
        .float()
        .reshape(2, 32)
        .contiguous(),
        so2.adamw_attn_gate_w.detach().float().reshape(32, 2, 1).contiguous(),
        grad_phase.view(n_node, 16 * 64),
        grad_x_wide_down.view(n_node, 16 * 64),
    )
    return grad_phase.reshape_as(phase), grad_x_wide_down


def _qk_manual_backward(
    runner: Any,
    grad_logits: Tensor,
) -> Tensor:
    so2 = runner.so2
    n_node = runner.node_count
    n_edge = runner.edge_count
    x_wide = runner.x_wide.detach()
    x_l0 = x_wide[:, 0, :].reshape(n_node, 2, 32)
    q_node = runner.q_node
    k_node = runner.k_node
    grad_q_node = getattr(runner, "grad_q_node", None)
    grad_k_node = getattr(runner, "grad_k_node", None)
    if grad_q_node is None:
        grad_q_node = torch.empty_like(
            q_node,
            memory_format=torch.contiguous_format,
        )
        grad_k_node = torch.empty_like(
            k_node,
            memory_format=torch.contiguous_format,
        )
        runner.grad_q_node = grad_q_node
        runner.grad_k_node = grad_k_node
    if not grad_q_node.is_contiguous() or not grad_k_node.is_contiguous():
        raise RuntimeError("Neo Q/K backward buffers must be compact N x 2 x 32")
    grad_q_node.zero_()
    grad_k_node.zero_()
    runner.qk_edge_backward(
        grad_logits.contiguous(),
        q_node,
        k_node,
        runner.src_i32,
        runner.dst_i32,
        grad_q_node,
        grad_k_node,
    )
    grad_x_wide = getattr(runner, "grad_x_wide_qk", None)
    if grad_x_wide is None or grad_x_wide.shape != x_wide.shape:
        grad_x_wide = torch.empty(
            x_wide.shape,
            device=x_wide.device,
            dtype=x_wide.dtype,
        )
        runner.grad_x_wide_qk = grad_x_wide
    runner.qk_node_input_adjoint(
        x_l0.contiguous(),
        grad_q_node,
        grad_k_node,
        so2.attn_q_proj.weight.detach().float().view(32, 2, 32).contiguous(),
        so2.attn_k_proj.weight.detach().float().view(32, 2, 32).contiguous(),
        so2.attn_qk_norm.adam_scale.detach().float().contiguous(),
        grad_x_wide.view(n_node, 16 * 64),
    )
    return grad_x_wide


def _native_edge_major_stack_grad(grad_stack_out: Tensor) -> Tensor:
    """Validate the exact in-place Phase-C adjoint consumed by final SO2."""
    edge_count = grad_stack_out.shape[0]
    expected_shape = (edge_count, 2, 10, 32)
    expected_stride = (2 * 10 * 32, 10 * 32, 32, 1)
    if (
        grad_stack_out.shape != expected_shape
        or grad_stack_out.stride() != expected_stride
        or grad_stack_out.dtype != torch.float32
    ):
        raise RuntimeError(
            "in-place Phase-C adjoint requires compact edge-major storage: "
            f"got shape={tuple(grad_stack_out.shape)} "
            f"stride={grad_stack_out.stride()} dtype={grad_stack_out.dtype}, "
            f"expected shape={expected_shape} stride={expected_stride}"
        )
    return grad_stack_out


def _phase_c_layout_grad_stack(runner: Any) -> Tensor:
    """Select aliased edge-major or ordinary Phase-C output storage."""
    if runner.phase_c_y is not None:
        raise RuntimeError(
            "in-place Phase-C adjoint requires the folded single-input stack"
        )
    grad_mixed = runner.grad_mixed_slab
    if grad_mixed is None or (
        grad_mixed.untyped_storage()._cdata
        == runner.phase_c_stack.untyped_storage()._cdata
    ):
        raise RuntimeError(
            "in-place Phase-C adjoint must retain a distinct grad_mixed_slab"
        )
    # Phase C owns every destination edge and stores G at the same
    # edge/focus address only after its complete 10x32 input fragment is
    # register-resident. Final SO2 consumes G before gate scratch reuses y2.
    return _native_edge_major_stack_grad(runner.phase_c_stack)


def _stack_backward_manual(runner: Any, grad_stack_out: Tensor) -> Tensor:
    cur_grad = _native_edge_major_stack_grad(grad_stack_out)
    for cache_index in range(len(runner.stack_caches) - 1, -1, -1):
        cache = runner.stack_caches[cache_index]
        residual_grad = cur_grad
        if cache.final:
            grad_y = cur_grad
        else:
            if runner.config.combined_so2_gate:
                assert runner.combined_gate_backward is not None
                runner.combined_gate_backward(
                    cur_grad.view(-1, 10 * 32),
                    cache.y.detach().view(-1, 10 * 32),
                    cache.non_linear.gate_linear.weight.detach()
                    .view(32, 2, 3 * 32)
                    .contiguous(),
                    runner.grad_y,
                )
                grad_y = runner.grad_y.view_as(cache.y)
            else:
                grad_gate_logits = runner._run_structural_gate_backward(
                    runner.structural_gate_backward,
                    cur_grad,
                    cache.y,
                    cache.logits,
                    runner.grad_y.view_as(cache.y),
                    grad_logits=runner.grad_gate_logits,
                    overwrite_logits=True,
                )
                grad_y = runner.grad_y.view_as(cache.y)
                runner._focus_major_gate_linear_backward_add(
                    grad_y,
                    grad_gate_logits,
                    cache.non_linear.gate_linear.weight.detach(),
                )
        linear = runner.so2.so2_linears[cache_index]
        so2_out = runner.grad_mixed_slab if cache.final else None
        cur_grad = _neo_so2_linear_backward_input_with_residual(
            linear,
            grad_y,
            residual_grad,
            inplace_residual=not cache.final,
            out=so2_out,
        )
    return cur_grad


def _x_wide_manual_backward(runner: Any, grad_x_wide_total: Tensor) -> Tensor:
    block = runner.block
    so2 = runner.so2
    n_node = runner.node_count
    x_so2 = runner.x if runner.use_full_node else runner.x[:, : block.mp_ebed_dim, :, :]
    x_pre = block.pre_so2_norm(x_so2)
    x_pre_flat = x_pre.reshape(n_node, x_so2.shape[1], block.channels).unsqueeze(2)
    grad_x_pre_flat = _so3_linear_backward_input(
        so2.pre_focus_mix,
        x_pre_flat,
        grad_x_wide_total.unsqueeze(2),
    )
    grad_x_pre = grad_x_pre_flat.squeeze(2).reshape_as(x_so2)
    if type(block.pre_so2_norm).__name__ != "Identity":
        raise NotImplementedError(
            "manual x-wide backward currently expects Identity pre norm"
        )
    if runner.use_full_node:
        grad_x = grad_x_pre
    else:
        grad_x = torch.zeros_like(runner.x)
        grad_x[:, : block.mp_ebed_dim, :, :] = grad_x_pre
    expected_stride = (
        grad_x.shape[-1],
        grad_x.shape[0] * grad_x.shape[-1],
        grad_x.shape[-1],
        1,
    )
    if grad_x.stride() != expected_stride:
        grad_x = grad_x.permute(1, 0, 2, 3).contiguous().permute(1, 0, 2, 3)
    return grad_x


def _make_edge_cache(
    *,
    src: Tensor,
    dst: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    edge_env: Tensor,
    edge_src_gate: Tensor,
) -> Any:
    return SimpleNamespace(
        src=src,
        dst=dst,
        D_full=d_full,
        Dt_full=dt_full,
        edge_env=edge_env,
        edge_src_gate=_edge_src_gate_arg(edge_src_gate),
        D_to_m_cache={},
        Dt_from_m_cache={},
    )


def _build_runner(
    handle: int,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    edge_src_gate: Tensor,
) -> Any:
    """Build a runner after enforcing CuTe's runtime pointer contract.

    This function executes below the custom-op boundary, including for the
    compile-visible thin path.  Exact ``data_ptr`` checks are therefore safe
    here and repair contiguous offset views without introducing a Dynamo graph
    break above the op.
    """
    with _REGISTRY_LOCK:
        entry = _REGISTRY[int(handle)]
    config = entry.config
    if config.native_sm90_path:
        from .sm90.runner import NeoSm90SO2Runner as Runner
    else:
        from .runner import NeoFullCuteBackward as Runner

    x = _aligned_contiguous(x)
    shared_wigner_storage = d_full.data_ptr() == dt_full.data_ptr()
    d_full = _aligned_contiguous(d_full)
    dt_full = d_full if shared_wigner_storage else _aligned_contiguous(dt_full)
    radial_feat = _aligned_contiguous(radial_feat)
    edge_env = _aligned_contiguous(edge_env)
    src = _aligned_contiguous(src)
    dst = _aligned_contiguous(dst)
    dst_ptr = _aligned_contiguous(dst_ptr)
    source_order = _aligned_contiguous(source_order)
    source_ptr = _aligned_contiguous(source_ptr)
    edge_src_gate = _aligned_contiguous(edge_src_gate)
    edge_cache = _make_edge_cache(
        src=src,
        dst=dst,
        d_full=d_full,
        dt_full=dt_full,
        edge_env=edge_env,
        edge_src_gate=edge_src_gate,
    )
    record = SimpleNamespace(edge_cache=edge_cache)
    with torch.cuda.device(x.device), torch.no_grad():
        runner = Runner(
            torch,
            entry.block,
            record,
            x,
            d_full,
            dt_full,
            radial_feat,
            dst_ptr,
            source_order,
            source_ptr,
            runtime_config=config,
        )
    return runner


def _so2_packed_direct_forward_impl(
    handle: int,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    edge_src_gate: Tensor,
) -> tuple[Tensor, Tensor]:
    _assert_not_cuda_graph_capturing("packed forward", x)
    runner = _build_runner(
        handle,
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
    )
    runner_token = torch.empty(
        (1,),
        device=x.device,
        dtype=torch.uint8,
    )
    _store_packed_runner(runner_token, runner)
    return _layout_like(runner.final.detach(), x), runner_token


def _so2_backward_from_runner_current_device(
    handle: int,
    grad_out: Tensor,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    edge_src_gate: Tensor,
    runner: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del handle
    grad_x, grad_d, grad_dt, grad_radial = _runner_backward_manual(
        runner,
        grad_out,
    )

    grad_edge_env = runner.grad_edge.view_as(edge_env)
    if edge_src_gate.numel() != 0:
        factor = edge_src_gate.reshape_as(edge_env).float().clamp_min(0.0).sqrt()
        grad_edge_env = grad_edge_env * factor.to(dtype=grad_edge_env.dtype)
    grad_edge_env = (
        grad_edge_env.clone() if grad_edge_env._base is not None else grad_edge_env
    )
    grad_edge_env.masked_fill_(edge_env <= 0, 0)
    grad_x_out = _grad_layout_like(grad_x, x, skip=True)
    grad_d_out = _grad_layout_like(grad_d, d_full, skip=True)
    grad_dt_out = _grad_layout_like(grad_dt, dt_full, skip=True)
    grad_radial_out = _grad_layout_like(grad_radial, radial_feat, skip=True)
    grad_edge_env_out = _grad_layout_like(grad_edge_env, edge_env, skip=True)
    if x.ndim == 4:
        n_node = x.shape[0]
        channels = x.shape[-1]
        grad_x_stride = (channels, n_node * channels, channels, 1)
    else:
        grad_x_stride = x.stride()
    for name, actual, expected_like, expected_stride in (
        ("x", grad_x_out, x, grad_x_stride),
        ("D", grad_d_out, d_full, d_full.stride()),
        ("Dt", grad_dt_out, dt_full, dt_full.stride()),
        ("radial", grad_radial_out, radial_feat, radial_feat.stride()),
        ("edge_env", grad_edge_env_out, edge_env, edge_env.stride()),
    ):
        _assert_grad_meta_contract(
            actual,
            expected_like,
            name=name,
            expected_stride=expected_stride,
        )
    return (
        grad_x_out,
        grad_d_out,
        grad_dt_out,
        grad_radial_out,
        grad_edge_env_out,
    )


def _so2_backward_from_runner(
    handle: int,
    grad_out: Tensor,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    edge_src_gate: Tensor,
    runner: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run all compilation and launches on the operand's CUDA device."""
    with torch.cuda.device(grad_out.device):
        return _so2_backward_from_runner_current_device(
            handle,
            grad_out,
            x,
            d_full,
            dt_full,
            radial_feat,
            edge_env,
            edge_src_gate,
            runner,
        )


def _so2_packed_direct_backward_impl(
    handle: int,
    grad_out: Tensor,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    edge_src_gate: Tensor,
    runner_token: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    _assert_not_cuda_graph_capturing("packed backward", grad_out)
    runner = _borrow_packed_runner(runner_token)
    if runner is None:
        runner = _build_runner(
            handle,
            x,
            d_full,
            dt_full,
            radial_feat,
            edge_env,
            src,
            dst,
            dst_ptr,
            source_order,
            source_ptr,
            edge_src_gate,
        )
    return _so2_backward_from_runner(
        handle,
        grad_out,
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        edge_src_gate,
        runner,
    )


def _runner_backward_manual(
    runner: Any,
    grad_out: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if getattr(runner, "uses_native_sm90_path", False):
        return runner.input_adjoint(grad_out)

    so2 = runner.so2
    n_edge = runner.edge_count

    grad_phase_c_out, grad_x_wide_down = _final_manual_backward(runner, grad_out)
    # Keep the large SO2 slabs out of the readout-backward allocation crest.
    runner.ensure_backward_workspace()

    from .phase_c import (
        NeoPhaseCBackwardLayoutOutputs,
    )

    layout_outputs = getattr(runner, "phase_c_layout_outputs", None)
    if layout_outputs is None:
        layout_outputs = NeoPhaseCBackwardLayoutOutputs(
            grad_stack=_phase_c_layout_grad_stack(runner),
            grad_wigner_dt=runner.grad_dt,
            grad_logits=runner.grad_logits,
            grad_edge=runner.grad_edge,
            grad_z_partial=runner.grad_z_partial,
            grad_z=runner.grad_z,
            grad_focus_src=torch.empty(
                2,
                runner.edge_count,
                32,
                device=runner.x.device,
                dtype=torch.float32,
            ),
        )
        runner.phase_c_layout_outputs = layout_outputs
    runner.phase_c_layout_backward(
        grad_phase_c_out.contiguous(),
        runner.phase_c_stack.detach().contiguous(),
        runner.dt.detach().contiguous(),
        runner.alpha,
        runner.focus_alpha,
        runner.dst_ptr_i32,
        runner.rotate,
        runner.edge_gate,
        so2.adamw_attn_z_bias_raw.detach().reshape(2).float().contiguous(),
        runner.group_max,
        runner.denom,
        runner.focus_gate_src.detach().contiguous(),
        so2.adamw_focus_compete_w.detach().float().contiguous(),
        so2.focus_compete_norm.adam_scale.detach().float().reshape(2, 32).contiguous(),
        layout_outputs,
    )
    grad_stack_out = _native_edge_major_stack_grad(layout_outputs.grad_stack)
    grad_focus_src_focus = layout_outputs.grad_focus_src

    grad_x_wide_qk = _qk_manual_backward(
        runner,
        runner.grad_logits.view(n_edge, 2),
    )
    grad_mixed = _stack_backward_manual(runner, grad_stack_out)

    grad_mixed_focus = grad_mixed
    if not grad_mixed_focus.is_contiguous():
        grad_mixed_focus = grad_mixed_focus.contiguous()
    x_wide_flat = runner.x_wide.detach().contiguous().view(runner.node_count, 16 * 64)
    radial_state = runner.radial_compact
    from .radial_phase_a import (
        run_neo_radial_phase_a_backward_node_tiled,
    )

    run_neo_radial_phase_a_backward_node_tiled(
        grad_mixed_focus.view(runner.edge_count, 2 * 10 * 32),
        runner.grad_logits,
        radial_state,
        runner.so2.radial_degree_mixer.channel_basis.detach().view(64).contiguous(),
        x_wide_flat,
        runner.source_order_i32,
        runner.source_ptr_i32,
        runner.d.detach(),
        grad_focus_src_focus=grad_focus_src_focus,
        batched_radial_projection_weight=runner.batched_radial_projection_weight,
        grad_x_wide=runner.grad_x_wide_phase_a,
        grad_d_full=runner.grad_d,
        grad_radial_m0=runner.grad_radial_flat,
        validate_csr=runtime_policy.is_cute_strict_enabled(),
    )
    grad_radial = runner.grad_radial_flat.view_as(runner.radial)
    grad_x_wide_phase_a = runner.grad_x_wide_phase_a.view_as(runner.x_wide)
    grad_d = runner.grad_d

    grad_x_wide_total = grad_x_wide_phase_a
    grad_x_wide_total.add_(grad_x_wide_qk)
    grad_x_wide_total.add_(grad_x_wide_down)
    grad_x = _x_wide_manual_backward(runner, grad_x_wide_total)
    return grad_x, grad_d, runner.grad_dt, grad_radial


def _stateful_custom_op_tags() -> tuple[Any, ...] | None:
    """Tag hidden-state runner ops as unsafe for direct CUDA graph capture."""
    tag_type = getattr(getattr(torch, "_C", None), "Tag", None)
    cudagraph_unsafe = getattr(tag_type, "cudagraph_unsafe", None)
    if cudagraph_unsafe is None:
        return None
    return (cudagraph_unsafe,)


_SO2_CUSTOM_OP_TAGS = _stateful_custom_op_tags()
_so2_packed_direct_op = torch.library.custom_op(
    "sezm_cute::so2_packed_direct", mutates_args=(), tags=_SO2_CUSTOM_OP_TAGS
)(_so2_packed_direct_forward_impl)
_so2_packed_direct_bwd_op = torch.library.custom_op(
    "sezm_cute::so2_packed_direct_bwd", mutates_args=(), tags=_SO2_CUSTOM_OP_TAGS
)(_so2_packed_direct_backward_impl)


@_so2_packed_direct_op.register_fake
def _(
    handle: int,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    edge_src_gate: Tensor,
) -> tuple[Tensor, Tensor]:
    del handle
    del d_full, dt_full, radial_feat, edge_env, src, dst, dst_ptr
    del source_order, source_ptr, edge_src_gate
    return (
        torch.empty_like(x),
        torch.empty((1,), device=x.device, dtype=torch.uint8),
    )


@_so2_packed_direct_bwd_op.register_fake
def _(
    handle: int,
    grad_out: Tensor,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    edge_src_gate: Tensor,
    runner_token: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    del handle, grad_out, src, dst, dst_ptr, source_order, source_ptr
    del edge_src_gate, runner_token
    return (
        _fake_x_wide_grad_like(x, skip=True),
        torch.empty_like(d_full),
        torch.empty_like(dt_full),
        torch.empty_like(radial_feat),
        torch.empty_like(edge_env),
    )


def _so2_packed_direct_setup_context(
    ctx: Any,
    inputs: tuple[Any, ...],
    output: tuple[Tensor, Tensor],
) -> None:
    _, runner_token = output
    (
        handle,
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
    ) = inputs
    ctx.handle = int(handle)
    ctx.save_for_backward(
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
        runner_token,
    )


def _so2_packed_direct_registered_backward_impl(
    ctx: Any,
    grad_out: Tensor,
) -> tuple[Any, ...]:
    (
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
        runner_token,
    ) = ctx.saved_tensors
    grad_x, grad_d, grad_dt, grad_radial, grad_edge_env = _so2_packed_direct_bwd_op(
        ctx.handle,
        grad_out,
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
        runner_token,
    )
    return (
        None,
        grad_x,
        grad_d,
        grad_dt,
        grad_radial,
        grad_edge_env,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _so2_packed_direct_backward(
    ctx: Any,
    grad_out: Tensor,
    grad_runner_token: Tensor | None,
) -> tuple[Any, ...]:
    """Run packed-direct backward with its custom op visible to compilation."""
    del grad_runner_token
    return _so2_packed_direct_registered_backward_impl(ctx, grad_out)


_so2_packed_direct_op.register_autograd(
    _so2_packed_direct_backward,
    setup_context=_so2_packed_direct_setup_context,
)


def _cute_so2_impl(
    handle: int,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    edge_src_gate: Tensor,
) -> Tensor:
    output, _runner_token = _so2_packed_direct_op(
        int(handle),
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
    )
    return output


def cute_so2(
    handle: int,
    x: Tensor,
    d_full: Tensor,
    dt_full: Tensor,
    radial_feat: Tensor,
    edge_env: Tensor,
    src: Tensor,
    dst: Tensor,
    dst_ptr: Tensor,
    edge_src_gate: Tensor,
    source_order: Tensor | None = None,
    source_ptr: Tensor | None = None,
) -> Tensor:
    """Run SO2 through its registered custom-op boundary."""
    if source_order is None:
        source_order = src.new_empty((0,), dtype=torch.int32)
    if source_ptr is None:
        source_ptr = src.new_empty((0,), dtype=torch.int32)
    return _cute_so2_impl(
        handle,
        x,
        d_full,
        dt_full,
        radial_feat,
        edge_env,
        src,
        dst,
        dst_ptr,
        source_order,
        source_ptr,
        edge_src_gate,
    )


def _dst_ptr_from_sorted(
    torch_module: Any,
    dst: Tensor,
    n_node: int,
    *,
    destinations_sorted: bool,
) -> Tensor | None:
    if not destinations_sorted:
        return None
    if runtime_policy.is_cute_strict_enabled() and dst.numel() > 1:
        torch_module._assert_async(
            torch_module.all(dst[1:] >= dst[:-1]),
            "Neo SO2 destinations_sorted=True requires monotonically "
            "nondecreasing destination indices",
        )
    boundaries = torch_module.arange(
        n_node + 1,
        device=dst.device,
        dtype=torch_module.int64,
    )
    return torch_module.searchsorted(dst.contiguous(), boundaries)


def _validated_sorted_edge_metadata_args(
    edge_cache: Any,
    *,
    node_count: int,
    dst_ptr: Tensor | None,
    source_order: Tensor | None,
    source_ptr: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Return validated invocation-local CSR tensors for this edge cache."""
    if not getattr(edge_cache, "destinations_sorted", False):
        return None
    if dst_ptr is None or source_order is None or source_ptr is None:
        return None
    device = edge_cache.src.device
    if (
        dst_ptr.device != device
        or source_order.device != device
        or source_ptr.device != device
        or dst_ptr.dtype != torch.int32
        or source_order.dtype != torch.int32
        or source_ptr.dtype != torch.int32
        or dst_ptr.numel() != node_count + 1
        or source_order.numel() != edge_cache.src.numel()
        or source_ptr.numel() != node_count + 1
    ):
        return None
    return (
        dst_ptr.contiguous(),
        source_order.contiguous(),
        source_ptr.contiguous(),
    )


def _cuda_compute_capability(device_index: int) -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability(device_index))


def _architecture_default_config(
    compute_capability: tuple[int, int],
) -> NeoSO2RuntimeConfig:
    return NeoSO2RuntimeConfig(
        native_sm90_path=compute_capability == runtime_policy.SM90_CAPABILITY,
        per_focus_so2_fwd_pair=(
            compute_capability in runtime_policy.SM80_PROFILE_CAPABILITIES
        ),
        combined_so2_gate=(
            compute_capability in runtime_policy.FUSED_SO2_GATE_CAPABILITIES
        ),
    )


def _maybe_run_prepared_cute_so2(
    block: Any,
    x: Tensor,
    edge_cache: Any,
    radial_feat: Tensor,
    dst_ptr: Tensor | None = None,
    source_order: Tensor | None = None,
    source_ptr: Tensor | None = None,
) -> Tensor | None:
    """Dispatch prevalidated packed SO2 state without a Python graph break."""
    state = getattr(block, "_deepmd_cute_so2_state", None)
    if not isinstance(state, _RegisteredSO2State):
        return None
    with _REGISTRY_LOCK:
        entry = _REGISTRY.get(state.handle)
        if entry is None or entry.block is not block:
            return None
    d_full = edge_cache.D_packed
    dt_full = d_full
    destinations_sorted = bool(getattr(edge_cache, "destinations_sorted", False))
    device_index = x.device.index
    if (
        block.training
        or x.device.type != "cuda"
        or device_index is None
        or device_index != state.device_index
        or x.dtype != torch.float32
        or torch.is_autocast_enabled(x.device.type)
        or d_full is None
        or dt_full is None
        or d_full is not dt_full
        or d_full.dim() != 2
        or dt_full.dim() != 2
        or edge_cache.edge_src_gate is not None
        or not destinations_sorted
        or not _dtypes_use_strict_fp32(
            (
                d_full.dtype,
                dt_full.dtype,
                radial_feat.dtype,
                edge_cache.edge_env.dtype,
            )
        )
    ):
        return None
    edge_count = edge_cache.src.numel()
    if not runtime_policy.so2_int32_indexing_is_safe(
        edge_count=edge_count,
        node_count=x.shape[0],
    ):
        return None
    metadata_args = _validated_sorted_edge_metadata_args(
        edge_cache,
        node_count=x.shape[0],
        dst_ptr=dst_ptr,
        source_order=source_order,
        source_ptr=source_ptr,
    )
    if metadata_args is None:
        dst_ptr = _dst_ptr_from_sorted(
            torch,
            edge_cache.dst,
            x.shape[0],
            destinations_sorted=destinations_sorted,
        )
        if dst_ptr is None:
            return None
        source_order = edge_cache.src.new_empty((0,), dtype=torch.int32)
        source_ptr = edge_cache.src.new_empty((0,), dtype=torch.int32)
    else:
        dst_ptr, source_order, source_ptr = metadata_args
    # The opaque custom-op implementation performs exact pointer-alignment
    # canonicalization in ``_build_runner``. Keep this wrapper graph-visible.
    empty_edge_src_gate = edge_cache.edge_env.new_empty((0,))
    d_arg = d_full.contiguous()
    output = cute_so2(
        state.handle,
        x.contiguous(),
        d_arg,
        d_arg,
        radial_feat.contiguous(),
        edge_cache.edge_env.contiguous(),
        edge_cache.src.contiguous(),
        edge_cache.dst.contiguous(),
        dst_ptr.contiguous(),
        empty_edge_src_gate,
        source_order=source_order.contiguous(),
        source_ptr=source_ptr.contiguous(),
    )
    return _layout_like(output, x)


@torch.compiler.disable
def _maybe_run_cute_so2_fallback(
    block: Any,
    x: Tensor,
    edge_cache: Any,
    radial_feat: Tensor,
    dst_ptr: Tensor | None = None,
    source_order: Tensor | None = None,
    source_ptr: Tensor | None = None,
) -> Tensor | None:
    """Run the opt-in Neo CuTe SO2 path, or return ``None`` for fallback."""
    if edge_cache.D_packed is None:
        return None
    # The optimized backward does not expose the differentiable SFPG
    # source-gate adjoint. Preserve force/stress correctness via eager fallback.
    if edge_cache.edge_src_gate is not None:
        return None
    destinations_sorted = bool(getattr(edge_cache, "destinations_sorted", False))
    if not is_neo_so2_runtime_eligible(
        block,
        training=bool(block.training),
        device=x.device,
        dtype=x.dtype,
        edge_count=edge_cache.src.numel(),
        node_count=x.shape[0],
        destinations_sorted=destinations_sorted,
    ):
        return None
    if not _dtypes_use_strict_fp32(
        (
            edge_cache.D_packed.dtype,
            radial_feat.dtype,
            edge_cache.edge_env.dtype,
        )
    ):
        return None

    state = getattr(block, "_deepmd_cute_so2_state", None)
    if state is False:
        return None
    if state is not None and not isinstance(state, _RegisteredSO2State):
        state = None
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compute_capability = _cuda_compute_capability(device_index)
    if not is_supported_so2_compute_capability(compute_capability):
        return None
    config = _architecture_default_config(compute_capability)
    if not runtime_policy.so2_int32_indexing_is_safe(
        edge_count=edge_cache.src.numel(),
        node_count=x.shape[0],
    ):
        return None
    if edge_cache.D_packed.dim() != 2:
        return None
    if state is None or state.device_index != device_index or state.config != config:
        state = _register_cute_so2_state(block, device_index, config)
        if state is None:
            return None

    metadata_args = _validated_sorted_edge_metadata_args(
        edge_cache,
        node_count=x.shape[0],
        dst_ptr=dst_ptr,
        source_order=source_order,
        source_ptr=source_ptr,
    )
    if metadata_args is None:
        dst_ptr = _dst_ptr_from_sorted(
            torch,
            edge_cache.dst,
            x.shape[0],
            destinations_sorted=destinations_sorted,
        )
        if dst_ptr is None:
            return None
        source_order = edge_cache.src.new_empty((0,), dtype=torch.int32)
        source_ptr = edge_cache.src.new_empty((0,), dtype=torch.int32)
    else:
        dst_ptr, source_order, source_ptr = metadata_args
    x_arg = _aligned_contiguous(x)
    d_arg = _aligned_contiguous(edge_cache.D_packed)
    dt_arg = d_arg
    radial_arg = _aligned_contiguous(radial_feat)
    edge_env_arg = _aligned_contiguous(edge_cache.edge_env)
    src_arg = _aligned_contiguous(edge_cache.src)
    dst_arg = _aligned_contiguous(edge_cache.dst)
    dst_ptr_arg = _aligned_contiguous(dst_ptr)
    source_order_arg = _aligned_contiguous(source_order)
    source_ptr_arg = _aligned_contiguous(source_ptr)
    edge_src_gate = edge_cache.edge_src_gate
    if edge_src_gate is None:
        edge_src_gate = edge_cache.edge_env.new_empty((0,))
    edge_src_gate_arg = _aligned_contiguous(edge_src_gate)
    output = cute_so2(
        state.handle,
        x_arg,
        d_arg,
        dt_arg,
        radial_arg,
        edge_env_arg,
        src_arg,
        dst_arg,
        dst_ptr_arg,
        edge_src_gate_arg,
        source_order=source_order_arg,
        source_ptr=source_ptr_arg,
    )
    return _layout_like(output, x)


def maybe_run_cute_so2(
    block: Any,
    x: Tensor,
    edge_cache: Any,
    radial_feat: Tensor,
    dst_ptr: Tensor | None = None,
    source_order: Tensor | None = None,
    source_ptr: Tensor | None = None,
) -> Tensor | None:
    """Use prevalidated opaque dispatch, or the conservative eager fallback."""
    state = getattr(block, "_deepmd_cute_so2_state", None)
    use_prepared = isinstance(
        state,
        _RegisteredSO2State,
    ) or runtime_policy.is_so2_thin_wrapper_enabled(_tensor_compute_capability(x))
    if use_prepared:
        output = _maybe_run_prepared_cute_so2(
            block,
            x,
            edge_cache,
            radial_feat,
            dst_ptr,
            source_order,
            source_ptr,
        )
        if output is not None:
            return output
    return _maybe_run_cute_so2_fallback(
        block,
        x,
        edge_cache,
        radial_feat,
        dst_ptr,
        source_order,
        source_ptr,
    )
