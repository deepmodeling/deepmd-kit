# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contracts for the opt-in CuTe Neo K1 inference path."""

from __future__ import (
    annotations,
)

import importlib
from dataclasses import (
    fields,
)
from types import (
    SimpleNamespace,
)

import pytest
import torch

from deepmd.kernels.cute.neo import k1 as _K1
from deepmd.kernels.cute.neo import k1_runner as _K1_RUNNER
from deepmd.kernels.cute.neo import (
    runtime_policy,
)
from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
    EdgeFeatureCache,
    _build_edge_wigner,
    _separate_packed_wigner,
)
from deepmd.pt.model.descriptor.sezm_nn.norm import (
    EquivariantRMSNorm,
)

NeoK1BackwardWorkspace = _K1_RUNNER.NeoK1BackwardWorkspace
NeoK1RuntimeConfig = _K1.NeoK1RuntimeConfig
NeoFullCuteBackward = _K1_RUNNER.NeoFullCuteBackward
StackCache = _K1_RUNNER.StackCache
_validate_runtime_config = _K1_RUNNER._validate_runtime_config
_uses_packed_message_grid = _K1_RUNNER._uses_packed_message_grid


class _Identity:
    pass


def _neo_like_block(**overrides):
    frame_contract = {
        "coefficient_layout": "packed",
        "n_frames": 3,
        "channels": 32,
    }
    message_node_grid_product = SimpleNamespace(
        layout="flat",
        mode="cross",
        op_type="glu",
        n_focus=2,
        n_frames=3,
        channels=32,
        dtype=torch.float32,
        frame_expand=SimpleNamespace(**frame_contract),
        frame_contract=SimpleNamespace(**frame_contract),
    )
    so2 = SimpleNamespace(
        lmax=3,
        mmax=1,
        ebed_dim_full=16,
        reduced_dim=10,
        channels=32,
        n_focus=2,
        so2_focus_dim=32,
        hidden_channels=64,
        mixing_layers=3,
        n_atten_head=1,
        head_dim=32,
        radial_so2_mode="degree_channel",
        radial_so2_rank=1,
        so2_norm=False,
        focus_compete=True,
        focus_norm=True,
        edge_cartesian=False,
        node_cartesian_tp=None,
        message_node_grid_product=message_node_grid_product,
        atten_f_mix=False,
        attn_v_proj=None,
        attn_o_proj=None,
        mlp_bias=False,
        layer_scale=False,
        use_so2_attn_res=False,
    )
    for key, value in overrides.items():
        setattr(so2, key, value)
    block = torch.nn.Module()
    block.so2_conv = so2
    block.lmax = 3
    block.node_lmax = 3
    block.pre_so2_norm = _Identity()
    block.post_so2_norm = EquivariantRMSNorm(
        3,
        32,
        dtype=torch.float32,
        trainable=False,
    )
    block.runtime_weight = torch.nn.Parameter(
        torch.ones(1, dtype=torch.float32, device="cpu"),
        requires_grad=False,
    )
    return block


def test_exact_neo_contract_is_supported() -> None:
    block = _neo_like_block()

    assert _K1.get_neo_k1_spec(block).is_current_neo_target
    assert _K1.is_supported_neo_k1_block(block)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lmax", 4),
        ("mmax", 2),
        ("channels", 64),
        ("n_focus", 1),
        ("mixing_layers", 2),
        ("focus_compete", False),
        ("edge_cartesian", True),
    ],
)
def test_non_neo_contracts_fall_back(field: str, value: object) -> None:
    assert not _K1.is_supported_neo_k1_block(_neo_like_block(**{field: value}))


def test_unsupported_message_grid_contract_falls_back() -> None:
    block = _neo_like_block()
    block.so2_conv.message_node_grid_product.op_type = "mlp"

    assert not _K1.is_supported_neo_k1_block(block)


def test_unsupported_post_norm_falls_back() -> None:
    block = _neo_like_block()
    block.post_so2_norm = torch.nn.LayerNorm(32, device="cpu")

    assert not _K1.is_supported_neo_k1_block(block)


def test_gate_expand_contract_cache_tracks_buffer_versions() -> None:
    block = _neo_like_block()
    expand_index = torch.tensor(
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=torch.long,
        device="cpu",
    )
    block.so2_conv.non_linearities = [SimpleNamespace(expand_index=expand_index)]

    assert _K1._gate_expand_index_structure_is_supported(block)
    assert _K1._gate_expand_index_is_supported(block)
    expand_index[0] = 2
    assert _K1._gate_expand_index_structure_is_supported(block)
    assert not _K1._gate_expand_index_is_supported(block)


@pytest.mark.parametrize(
    "capability",
    sorted(runtime_policy.SUPPORTED_K1_CAPABILITIES),
)
def test_every_supported_architecture_has_a_validated_config(
    capability: tuple[int, int],
) -> None:
    config = _K1._architecture_default_config(capability)

    assert _validate_runtime_config(config, compute_capability=capability) is None


@pytest.mark.parametrize("capability", [(8, 0), (8, 6)])
def test_sm80_family_uses_per_focus_so2_forward(
    capability: tuple[int, int],
) -> None:
    config = _K1._architecture_default_config(capability)

    assert config.per_focus_so2_fwd_pair
    assert not config.native_sm90_path
    assert not config.combined_so2_gate


def test_sm90_uses_native_split_complex_path() -> None:
    config = _K1._architecture_default_config((9, 0))

    assert config.native_sm90_path
    assert not config.per_focus_so2_fwd_pair
    assert not config.combined_so2_gate


def test_sm100_uses_shared_default_profile() -> None:
    config = _K1._architecture_default_config((10, 0))

    assert not config.native_sm90_path
    assert not config.per_focus_so2_fwd_pair
    assert not config.combined_so2_gate


@pytest.mark.parametrize("capability", [(8, 9), (12, 0)])
def test_sm89_and_sm120_use_combined_so2_gate(
    capability: tuple[int, int],
) -> None:
    config = _K1._architecture_default_config(capability)

    assert config.combined_so2_gate
    assert not config.native_sm90_path
    assert not config.per_focus_so2_fwd_pair


def test_combined_so2_gate_rejects_misaligned_contiguous_tensor() -> None:
    pytest.importorskip("cutlass.cute")
    pytest.importorskip("cuda.bindings.driver")
    from deepmd.kernels.cute.neo.k1_kernels.cute_neo_so2_gate_combined_fwd import (
        _require_16_byte_alignment,
    )

    storage = torch.empty(9, dtype=torch.float32, device="cpu")
    aligned = storage[:8]
    misaligned = storage[1:9]
    assert aligned.data_ptr() % 16 == 0
    assert misaligned.is_contiguous()
    _require_16_byte_alignment((aligned,))
    with pytest.raises(ValueError, match="16-byte aligned"):
        _require_16_byte_alignment((misaligned,))


def test_runtime_config_contains_only_reached_selectors() -> None:
    assert {field.name for field in fields(NeoK1RuntimeConfig)} == {
        "native_sm90_path",
        "per_focus_so2_fwd_pair",
        "combined_so2_gate",
    }


def test_runtime_config_rejects_incompatible_capability() -> None:
    config = _K1._architecture_default_config((8, 0))

    with pytest.raises(RuntimeError, match="supported compute capability"):
        _validate_runtime_config(config, compute_capability=(7, 5))


def test_runtime_config_rejects_wrong_architecture_profile() -> None:
    sm80_config = _K1._architecture_default_config((8, 0))
    with pytest.raises(RuntimeError, match="per-focus SO2"):
        _validate_runtime_config(sm80_config, compute_capability=(12, 0))

    sm90_config = _K1._architecture_default_config((9, 0))
    with pytest.raises(RuntimeError, match="native SM90 K1"):
        _validate_runtime_config(sm90_config, compute_capability=(8, 0))

    sm120_config = _K1._architecture_default_config((12, 0))
    with pytest.raises(RuntimeError, match="combined SO2/gate"):
        _validate_runtime_config(sm120_config, compute_capability=(10, 0))
    with pytest.raises(RuntimeError, match="combined SO2/gate"):
        _validate_runtime_config(NeoK1RuntimeConfig(), compute_capability=(12, 0))


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((8, 0), True),
        ((8, 6), True),
        ((8, 9), False),
        ((9, 0), True),
        ((10, 0), False),
        ((12, 0), False),
    ],
)
def test_packed_message_grid_architecture_contract(
    capability: tuple[int, int],
    expected: bool,
) -> None:
    assert _uses_packed_message_grid(capability) is expected


def test_packed_edge_eligibility_requires_sorted_strict_fp32() -> None:
    kwargs = {
        "candidate": True,
        "edge_count": 12,
        "node_count": 4,
        "destinations_sorted": True,
        "runtime_dtypes": (torch.float32, torch.float32),
    }

    assert _K1.packed_wigner_edges_eligible(**kwargs)
    assert not _K1.packed_wigner_edges_eligible(
        **{**kwargs, "destinations_sorted": False}
    )
    assert not _K1.packed_wigner_edges_eligible(
        **{**kwargs, "runtime_dtypes": (torch.float64,)}
    )
    assert _K1.packed_wigner_edges_eligible(
        **{**kwargs, "edge_count": 3, "node_count": 4}
    )


def test_packed_wigner_has_a_separate_backward_compatible_cache_field() -> None:
    dense = torch.empty((2, 16, 16), device="cpu")
    dense_t = torch.empty_like(dense)
    actual_dense, actual_dense_t, packed = _separate_packed_wigner(dense, dense_t)
    assert actual_dense is dense
    assert actual_dense_t is dense_t
    assert packed is None

    panel = torch.empty((2, 46), device="cpu")
    actual_dense, actual_dense_t, packed = _separate_packed_wigner(panel, panel)
    assert actual_dense is None
    assert actual_dense_t is None
    assert packed is panel
    assert EdgeFeatureCache._fields[-2:] == ("destinations_sorted", "D_packed")


def test_ineligible_wigner_build_does_not_import_cute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    edge_vec = torch.tensor([[1.0, 0.0, 0.0]], device="cpu")
    edge_len = torch.linalg.vector_norm(edge_vec, dim=-1, keepdim=True)

    def eager_wigner(edge_quat: torch.Tensor):
        dense = edge_quat.new_zeros((edge_quat.shape[0], 1, 1))
        return dense, dense.transpose(-1, -2)

    dense, dense_t, edge_quat = _build_edge_wigner(
        edge_vec=edge_vec,
        edge_len=edge_len,
        eps=1.0e-8,
        random_gamma=False,
        wigner_calc=eager_wigner,
        packed_wigner=False,
    )

    assert dense is not None and tuple(dense.shape) == (1, 1, 1)
    assert dense_t is not None and tuple(dense_t.shape) == (1, 1, 1)
    assert tuple(edge_quat.shape) == (1, 4)


@pytest.mark.parametrize(
    ("training", "dtype", "edge_count", "node_count", "destinations_sorted"),
    [
        (True, torch.float32, 12, 4, True),
        (False, torch.float64, 12, 4, True),
        (False, torch.float32, 0, 4, True),
        (False, torch.float32, 12, 4, False),
    ],
)
def test_ineligible_runtime_contracts_fall_back(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    dtype: torch.dtype,
    edge_count: int,
    node_count: int,
    destinations_sorted: bool,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    block = _neo_like_block().eval()

    assert not _K1.is_neo_k1_runtime_eligible(
        block,
        training=training,
        device=torch.device("cuda", 0),
        dtype=dtype,
        edge_count=edge_count,
        node_count=node_count,
        destinations_sorted=destinations_sorted,
    )


def test_runtime_contract_allows_fewer_edges_than_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    block = _neo_like_block().eval()

    assert _K1.is_neo_k1_runtime_eligible(
        block,
        training=False,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
        edge_count=3,
        node_count=4,
        destinations_sorted=True,
    )


def _cute_cuda_runtime_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception:  # pragma: no cover - runtime dependent
        return False
    return True


@pytest.mark.skipif(
    not _cute_cuda_runtime_available(),
    reason="CuTe attention-prelude regression requires CUDA and CuTe DSL",
)
def test_attention_prelude_initializes_all_qk_rows_when_edges_are_sparse() -> None:
    from deepmd.kernels.cute.neo.k1_kernels.cute_neo_focus_src_backward import (
        compile_neo_attention_prelude_forward,
    )

    torch.manual_seed(20260810)
    device = torch.device("cuda")
    edge_count = 2
    node_count = 3
    focus = torch.randn(edge_count, 64, device=device)
    x_l0 = torch.randn(node_count, 2, 32, device=device)
    focus_weight = torch.randn(32, 2, device=device)
    focus_scale = torch.randn(2, 32, device=device)
    q_weight = torch.randn(32, 2, 32, device=device)
    k_weight = torch.randn_like(q_weight)
    qk_scale = torch.randn(2, 32, device=device)
    focus_alpha = torch.full((edge_count, 2), torch.nan, device=device)
    q_node = torch.full((node_count, 2, 32), torch.nan, device=device)
    k_node = torch.full_like(q_node, torch.nan)
    focus_eps = 1.0e-5
    qk_eps = 1.0e-5
    tau = 0.7
    label_smoothing = 0.1

    focus_view = focus.view(edge_count, 2, 32)
    focus_norm = (
        focus_view
        * torch.rsqrt(focus_view.square().mean(dim=-1, keepdim=True) + focus_eps)
        * focus_scale.unsqueeze(0)
    )
    focus_logits = torch.stack(
        [
            (focus_norm[:, focus_idx] * focus_weight[:, focus_idx]).sum(dim=-1)
            for focus_idx in range(2)
        ],
        dim=-1,
    )
    expected_alpha = torch.softmax(focus_logits / tau, dim=-1)
    expected_alpha = expected_alpha * (1.0 - label_smoothing) + (label_smoothing / 2.0)
    x_norm = (
        x_l0
        * torch.rsqrt(x_l0.square().mean(dim=-1, keepdim=True) + qk_eps)
        * qk_scale.unsqueeze(0)
    )
    expected_q = torch.einsum("nfi,ifo->nfo", x_norm, q_weight)
    expected_k = torch.einsum("nfi,ifo->nfo", x_norm, k_weight)

    run = compile_neo_attention_prelude_forward(
        focus_eps,
        qk_eps,
        tau,
        label_smoothing,
    )
    run(
        focus,
        x_l0,
        focus_weight,
        focus_scale,
        q_weight,
        k_weight,
        qk_scale,
        focus_alpha,
        q_node,
        k_node,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(q_node).all()
    assert torch.isfinite(k_node).all()
    torch.testing.assert_close(focus_alpha, expected_alpha, atol=5.0e-5, rtol=5.0e-5)
    torch.testing.assert_close(q_node, expected_q, atol=5.0e-5, rtol=5.0e-5)
    torch.testing.assert_close(k_node, expected_k, atol=5.0e-5, rtol=5.0e-5)


def test_exact_runtime_contract_is_eligible_without_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    monkeypatch.setattr(runtime_policy, "uses_strict_fp32_matmul", lambda: True)
    block = _neo_like_block().eval()
    kwargs = {
        "training": False,
        "device": torch.device("cuda", 0),
        "dtype": torch.float32,
        "edge_count": 12,
        "node_count": 4,
        "destinations_sorted": True,
    }

    assert _K1.is_neo_k1_runtime_eligible(block, **kwargs)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device_type: True)
    assert not _K1.is_neo_k1_runtime_eligible(block, **kwargs)


def test_runtime_contract_rejects_tf32_matmul_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device_type: False)
    block = _neo_like_block().eval()
    kwargs = {
        "training": False,
        "device": torch.device("cuda", 0),
        "dtype": torch.float32,
        "edge_count": 12,
        "node_count": 4,
        "destinations_sorted": True,
    }

    monkeypatch.setattr(runtime_policy, "uses_strict_fp32_matmul", lambda: True)
    assert _K1.is_neo_k1_runtime_eligible(block, **kwargs)

    monkeypatch.setattr(runtime_policy, "uses_strict_fp32_matmul", lambda: False)
    assert not _K1.is_neo_k1_runtime_eligible(block, **kwargs)


def test_fake_native_gradient_layout_matches_runtime_contract() -> None:
    x = torch.empty(5, 16, 1, 32, device="cpu")

    grad = _K1._fake_x_wide_grad_like(x, skip=True)

    assert grad.shape == x.shape
    assert grad.stride() == (32, 5 * 32, 32, 1)


def test_custom_op_runtime_canonicalizes_misaligned_contiguous_view() -> None:
    base = torch.arange(17, dtype=torch.float32, device="cpu")
    offset_view = base[1:]

    assert offset_view.is_contiguous()
    assert offset_view.storage_offset() == 1
    actual = _K1._aligned_contiguous(offset_view)

    torch.testing.assert_close(actual, offset_view)
    assert actual.is_contiguous()
    assert actual.storage_offset() == 0
    assert actual.data_ptr() % 16 == 0


def test_custom_op_runtime_preserves_aligned_compact_tensor() -> None:
    tensor = torch.arange(16, dtype=torch.float32, device="cpu")

    assert _K1._aligned_contiguous(tensor) is tensor


def test_stack_cache_retains_only_backward_state() -> None:
    assert {field.name for field in fields(StackCache)} == {
        "y",
        "logits",
        "non_linear",
        "final",
    }
