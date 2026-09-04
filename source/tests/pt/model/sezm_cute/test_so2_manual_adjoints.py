# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the real-module adjoints used by Neo SO2 backward."""

from __future__ import (
    annotations,
)

import importlib
from types import (
    SimpleNamespace,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.activation import (
    SwiGLU,
)
from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
    SO3GridNet,
)
from deepmd.pt.model.descriptor.sezm_nn.norm import (
    EquivariantRMSNorm,
)
from deepmd.pt.model.descriptor.sezm_nn.so3 import (
    FocusLinear,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt_expt.kernels.cute.sezm.so2 import operation as so2


@pytest.fixture(autouse=True)
def _construct_modules_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "DEVICE", torch.device("cpu"))


def _randn(*shape: int, requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(
        *shape,
        dtype=torch.float64,
        device="cpu",
        requires_grad=requires_grad,
    )


def test_equivariant_rmsnorm_manual_adjoint_matches_real_module() -> None:
    norm = EquivariantRMSNorm(
        lmax=2,
        channels=4,
        n_focus=2,
        eps=2.0e-6,
        dtype=torch.float64,
        trainable=False,
    )
    with torch.no_grad():
        norm.adam_scale.copy_(
            torch.linspace(
                0.5,
                1.5,
                norm.adam_scale.numel(),
                dtype=torch.float64,
                device="cpu",
            ).reshape_as(norm.adam_scale)
        )
        norm.bias.copy_(
            torch.linspace(
                -0.2,
                0.2,
                norm.bias.numel(),
                dtype=torch.float64,
                device="cpu",
            ).reshape_as(norm.bias)
        )
    x = _randn(3, 9, 2, 4, requires_grad=True)
    grad_out = _randn(*x.shape)

    reference_grad = torch.autograd.grad(norm(x), x, grad_out)[0]
    actual_grad = so2._equivariant_rmsnorm_backward(
        norm,
        x.detach(),
        grad_out,
    )

    torch.testing.assert_close(actual_grad, reference_grad, atol=1.0e-12, rtol=1.0e-12)


def test_focus_linear_manual_adjoint_matches_real_module() -> None:
    linear = FocusLinear(
        in_channels=6,
        out_channels=5,
        n_focus=2,
        dtype=torch.float64,
        bias=True,
        trainable=False,
        seed=None,
    )
    x = _randn(7, 2, 6, requires_grad=True)
    grad_out = _randn(7, 2, 5)

    reference = linear(x)
    reference_grad = torch.autograd.grad(reference, x, grad_out)[0]

    torch.testing.assert_close(so2._focus_linear_forward(linear, x), reference)
    torch.testing.assert_close(
        so2._focus_linear_backward_input(linear, grad_out),
        reference_grad,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_swiglu_manual_adjoint_matches_real_module() -> None:
    activation = SwiGLU()
    x = _randn(6, 3, 10, requires_grad=True)
    grad_out = _randn(6, 3, 5)

    reference = activation(x)
    reference_grad = torch.autograd.grad(reference, x, grad_out)[0]

    torch.testing.assert_close(so2._swiglu_forward(x), reference)
    torch.testing.assert_close(
        so2._swiglu_backward_input(x.detach(), grad_out),
        reference_grad,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_grid_cross_glu_flat_manual_adjoint_matches_real_module() -> None:
    net = SO3GridNet(
        lmax=3,
        kmax=1,
        channels=4,
        n_focus=2,
        mode="cross",
        op_type="glu",
        dtype=torch.float64,
        layout="flat",
        coefficient_layout="packed",
        residual_scale_init=0.25,
        trainable=False,
        seed=None,
    )
    coeff_dim = net.projector.coeff_dim // net.n_frames
    query = _randn(3, coeff_dim, net.n_focus * net.channels, requires_grad=True)
    context = _randn(3, coeff_dim, net.n_focus * net.channels, requires_grad=True)
    grad_out = _randn(3, coeff_dim, net.n_focus * net.channels)

    reference_query, reference_context = torch.autograd.grad(
        net(query, context),
        (query, context),
        grad_out,
    )
    actual_query, actual_context = so2._so3_grid_cross_glu_flat_backward(
        net,
        query.detach(),
        context.detach(),
        grad_out,
    )

    torch.testing.assert_close(
        actual_query,
        reference_query,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    torch.testing.assert_close(
        actual_context,
        reference_context,
        atol=1.0e-11,
        rtol=1.0e-11,
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
    reason="SM90 message-grid differential requires CUDA and CuTe DSL",
)
def test_sm90_message_grid_forward_and_adjoint_match_real_module() -> None:
    if tuple(torch.cuda.get_device_capability()) != (9, 0):
        pytest.skip("SM90 message-grid differential requires an SM90 GPU")

    from deepmd.pt_expt.kernels.cute.sezm.so2.message_grid import (
        run_packed_message_grid_forward,
    )
    from deepmd.pt_expt.kernels.cute.sezm.so2.sm90.message_grid_readout import (
        prepare_sm90_message_grid_state,
        run_sm90_message_grid_backward,
    )

    prior_precision = torch.get_float32_matmul_precision()
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        net = SO3GridNet(
            lmax=3,
            kmax=1,
            channels=32,
            n_focus=2,
            mode="cross",
            op_type="glu",
            dtype=torch.float32,
            layout="flat",
            coefficient_layout="packed",
            residual_scale_init=0.25,
            trainable=False,
            seed=None,
        ).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(20260816)
        query = (
            0.1
            * torch.randn(
                5,
                16,
                64,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            )
        ).requires_grad_(True)
        context = (
            0.1
            * torch.randn(
                16,
                5,
                64,
                generator=generator,
                device="cuda",
                dtype=torch.float32,
            ).permute(1, 0, 2)
        ).requires_grad_(True)
        assert context.stride() == (64, 5 * 64, 1)
        grad_out = torch.randn(
            query.shape,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )

        reference = net(query, context)
        reference_query, reference_context = torch.autograd.grad(
            reference,
            (query, context),
            grad_out,
        )
        state = prepare_sm90_message_grid_state(net)
        actual, product = run_packed_message_grid_forward(
            net,
            query.detach(),
            context.detach(),
            return_product=True,
            sm90_state=state,
        )
        actual_query, actual_context = run_sm90_message_grid_backward(
            net,
            query.detach(),
            context.detach(),
            grad_out,
            product,
            state,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, reference, atol=5.0e-5, rtol=5.0e-5)
        torch.testing.assert_close(
            actual_query,
            reference_query,
            atol=5.0e-5,
            rtol=5.0e-5,
        )
        torch.testing.assert_close(
            actual_context,
            reference_context,
            atol=5.0e-5,
            rtol=5.0e-5,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
        torch.set_float32_matmul_precision(prior_precision)


@pytest.mark.skipif(
    not _cute_cuda_runtime_available(),
    reason="Q/K CuTe adjoint regression requires CUDA and CuTe DSL",
)
def test_qk_manual_adjoint_matches_autograd_with_sparse_edges() -> None:
    from deepmd.pt_expt.kernels.cute.sezm.so2.kernels.qk_edge import (
        compile_neo_qk_edge_backward,
        compile_neo_qk_node_input_adjoint,
    )

    torch.manual_seed(20260813)
    device = torch.device("cuda", torch.cuda.current_device())
    node_count = 7
    src = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    dst = torch.tensor([2, 2, 3], dtype=torch.int32, device=device)
    edge_count = src.numel()
    eps = 1.0e-5
    scale = 32.0**-0.5
    x_wide = torch.randn(
        node_count,
        16,
        64,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    q_weight = torch.randn(32, 2, 32, dtype=torch.float32, device=device)
    k_weight = torch.randn_like(q_weight)
    norm_scale = torch.randn(2, 32, dtype=torch.float32, device=device)
    grad_logits = torch.randn(edge_count, 2, dtype=torch.float32, device=device)

    x_l0 = x_wide[:, 0, :].reshape(node_count, 2, 32)
    x_norm = (
        x_l0
        * torch.rsqrt(x_l0.square().mean(dim=-1, keepdim=True) + eps)
        * norm_scale.unsqueeze(0)
    )
    q_node = torch.einsum("nfi,ifo->nfo", x_norm, q_weight)
    k_node = torch.einsum("nfi,ifo->nfo", x_norm, k_weight)
    logits = (q_node[dst.long()] * k_node[src.long()]).sum(dim=-1) * scale
    reference = torch.autograd.grad(logits, x_wide, grad_logits)[0]
    compile_identity = (
        torch.cuda.current_device(),
        *torch.cuda.get_device_capability(device),
    )

    runner = SimpleNamespace(
        so2=SimpleNamespace(
            attn_q_proj=SimpleNamespace(weight=q_weight.reshape(32, 64)),
            attn_k_proj=SimpleNamespace(weight=k_weight.reshape(32, 64)),
            attn_qk_norm=SimpleNamespace(adam_scale=norm_scale, eps=eps),
        ),
        node_count=node_count,
        edge_count=edge_count,
        x_wide=x_wide.detach(),
        q_node=q_node.detach().contiguous(),
        k_node=k_node.detach().contiguous(),
        src_i32=src,
        dst_i32=dst,
        qk_edge_backward=compile_neo_qk_edge_backward(scale, compile_identity),
        qk_node_input_adjoint=compile_neo_qk_node_input_adjoint(
            eps,
            compile_identity,
        ),
    )
    actual = so2._qk_manual_backward(runner, grad_logits)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, reference, atol=5.0e-5, rtol=5.0e-5)
