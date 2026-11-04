# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generated-input checks for frozen Neo edge-cohort inference."""

from __future__ import (
    annotations,
)

from types import (
    SimpleNamespace,
)

import pytest
import torch

pytest.importorskip("triton")

from deepmd.pt_expt.kernels.triton.sezm import so2_edge_cohort as ecs
from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as sv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def operands(edges: int, mode: str = "random") -> tuple:
    """Generate well-conditioned operands without a downloaded checkpoint."""
    torch.manual_seed(719)
    u = torch.randn(2, edges, 320, device="cuda") * 0.1
    if mode == "zero":
        u.zero_()
    alpha = torch.rand(edges, 2, device="cuda") * 0.8 + 0.1
    if mode == "tiny":
        alpha.fill_(1e-5)
    w0 = torch.randn(3, 2, 128, 128, device="cuda") * 0.01
    w1 = torch.randn(3, 2, 192, 192, device="cuda") * 0.01
    gw = torch.randn(2, 2, 32, 96, device="cuda") * 0.01
    out, z, uf = sv._mixing_stack_impl(u, alpha, w0, w1, gw, 3, 32, True)
    go = torch.randn_like(out) * 0.1
    return (
        go,
        out,
        z,
        uf,
        alpha,
        w0.transpose(2, 3).contiguous(),
        w1.transpose(2, 3).contiguous(),
        gw,
        gw.transpose(2, 3).contiguous(),
        3,
        32,
        True,
    )


@pytest.mark.parametrize("edges", [0, 1, 37, 129, 1025, 8193])
@pytest.mark.parametrize("cap", [64, 8192])
@pytest.mark.parametrize("mode", ["random", "zero", "tiny"])
def test_cohort_matches_native(edges: int, cap: int, mode: str) -> None:
    """Cover tail cohorts, two focus strides, legal tiny alpha and no mutation."""
    args = operands(edges, mode)
    copies = [x.clone() for x in args[:9]]
    expected_u, expected_a = sv._mixing_stack_bwd_impl(*args)
    go, out, z, _, alpha, w0t, w1t, gw, gwt, *_ = args
    actual_u = ecs.cohort_input_backward_impl(go, z, alpha, w0t, w1t, gw, gwt, cap)
    actual_a = ecs._cohort_alpha_backward(go, out, alpha)
    for actual, expected in ((actual_u, expected_u), (actual_a, expected_a)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for before, after in zip(copies, args[:9], strict=True):
        assert torch.equal(before, after)


def test_compiled_dynamic_cohort() -> None:
    """The opaque runtime edge loop supports online dynamic graph execution."""
    compiled = torch.compile(ecs.cohort_backward, fullgraph=True, dynamic=True)
    for edges in (37, 129):
        args = operands(edges)
        actual = compiled(*args)
        expected = sv._mixing_stack_bwd_impl(*args)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_make_fx_retains_runtime_cohorts(monkeypatch) -> None:
    """A small tracing example must not disable batching on larger inputs."""
    from torch.fx.experimental.proxy_tensor import (
        make_fx,
    )

    graph = make_fx(ecs.cohort_backward, tracing_mode="symbolic")(*operands(37))
    assert any(
        node.op == "call_function"
        and node.target == torch.ops.sezm_triton.so2_cohort_input_bwd.default
        for node in graph.graph.nodes
    )
    calls = []
    original = ecs.cohort_input_backward_impl

    def observed(go, z, alpha, w0t, w1t, gw, gwt, cap):
        calls.append((go.shape[0], cap))
        return original(go, z, alpha, w0t, w1t, gw, gwt, cap)

    monkeypatch.setattr(ecs, "cohort_input_backward_impl", observed)
    compiled = torch.compile(graph, fullgraph=True, dynamic=True)
    for edges in (37, 8192, 8193, 16384, 16385):
        args = operands(edges)
        actual = compiled(*args)
        expected = sv._mixing_stack_bwd_impl(*args)
        assert calls[-1] == (edges, ecs.COHORT_EDGES)
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.parametrize("dynamic", [False, True])
def test_compiled_native_backward(dynamic: bool) -> None:
    """Optional strides must preserve the default native compilation path."""
    compiled = torch.compile(sv._mixing_stack_bwd_op, fullgraph=True, dynamic=dynamic)
    for edges in (37, 129):
        args = operands(edges)
        expected = sv._mixing_stack_bwd_impl(*args)
        actual = compiled(*args)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize(
    "lmax,cf,apply", [(2, 32, True), (3, 64, True), (3, 32, False)]
)
def test_unsupported_layout(lmax: int, cf: int, apply: bool) -> None:
    """Untested layouts must not enter the experimental implementation."""
    args = operands(1)
    assert not ecs.supports_edge_cohort(args[0], args[2], lmax, cf, apply)
    assert not ecs.supports_edge_cohort(args[0].cpu(), args[2].cpu(), 3, 32, True)


@pytest.mark.parametrize("reason", ["weights", "saved_z", "saved_u", "create_graph"])
def test_training_dispatch_preserved(monkeypatch, reason: str) -> None:
    """Weight/state/higher-order requests bypass automatic cohort selection."""
    args = operands(1)
    go, out, z, uf, alpha, w0t, w1t, gw, *_ = args
    u = torch.zeros_like(uf)
    w0 = w0t.transpose(2, 3).contiguous()
    w1 = w1t.transpose(2, 3).contiguous()
    ctx = SimpleNamespace(
        saved_tensors=(u, alpha, out, z, uf, w0, w1, gw),
        needs_input_grad=(
            True,
            True,
            reason == "weights",
            False,
            False,
            False,
            False,
            False,
        ),
        lmax=3,
        focus_dim=32,
        apply_alpha=True,
    )
    calls = []

    def training(*unused):
        calls.append(True)
        return torch.zeros_like(u), torch.zeros_like(alpha), w0, w1, gw

    def forbidden(*unused):
        raise AssertionError("ECS entered a training/higher-order request")

    monkeypatch.setattr(sv, "_mixing_stack_train_bwd_op", training)
    monkeypatch.setattr(ecs, "cohort_backward", forbidden)
    with torch.set_grad_enabled(reason == "create_graph"):
        sv._mixing_stack_backward(
            ctx,
            go,
            torch.zeros_like(z) if reason == "saved_z" else None,
            torch.zeros_like(uf) if reason == "saved_u" else None,
        )
    assert calls == [True]


def test_frozen_dispatch(monkeypatch) -> None:
    """The production dispatcher must enter ECS, not an equal native fallback."""
    args = operands(37)
    go, out, z, uf, alpha, w0t, w1t, gw, *_ = args
    ctx = SimpleNamespace(
        saved_tensors=(
            uf,
            alpha,
            out,
            z,
            uf,
            w0t.transpose(2, 3).contiguous(),
            w1t.transpose(2, 3).contiguous(),
            gw,
        ),
        needs_input_grad=(True, True, False, False, False, False, False, False),
        lmax=3,
        focus_dim=32,
        apply_alpha=True,
    )
    expected = sv._mixing_stack_bwd_impl(*args)
    original = ecs.cohort_backward
    calls = []

    def observed(*inputs):
        calls.append(True)
        return original(*inputs)

    def forbidden(*unused):
        raise AssertionError("supported inference silently used the native fallback")

    monkeypatch.setattr(ecs, "cohort_backward", observed)
    monkeypatch.setattr(sv, "_mixing_stack_bwd_op", forbidden)
    with torch.no_grad():
        actual = sv._mixing_stack_backward(ctx, go, None, None)
    assert calls == [True]
    for a, b in zip(actual[:2], expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
def test_production_value_path_enters_cohort(monkeypatch, backend):
    """A real level-2 convolution must use cohorts on frozen first backward."""
    for name in (
        "DP_TRITON_TRAIN",
        "DP_CUDA_INFER",
        "DP_CUDA_TRAIN",
        "DP_CUTE_INFER",
        "DP_CUTILE_INFER",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("DP_TRITON_INFER", "2")
    if backend == "pt":
        from deepmd.pt.model.descriptor.sezm_nn.so2 import (
            SO2Convolution,
        )

        precision = {"dtype": torch.float32}
    else:
        from deepmd.pt_expt.descriptor.dpa4_nn.so2 import (
            SO2Convolution,
        )

        precision = {"precision": "float32"}
    conv = (
        SO2Convolution(
            lmax=3,
            mmax=1,
            channels=32,
            n_focus=2,
            focus_dim=32,
            mixing_layers=3,
            radial_so2_mode="degree_channel",
            radial_so2_rank=1,
            n_atten_head=1,
            seed=7,
            trainable=False,
            **precision,
        )
        .to("cuda")
        .eval()
    )
    x = (torch.randn(7, 16, 64, device="cuda") * 0.1).requires_grad_()
    radial = torch.randn(17, 4, 32, device="cuda") * 0.1
    cache = SimpleNamespace(
        src=torch.arange(17, device="cuda") % 7,
        D_full=torch.eye(16, device="cuda").repeat(17, 1, 1),
    )
    calls = []
    original = ecs.cohort_backward

    def observe(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(ecs, "cohort_backward", observe)
    output = conv._triton_value_path(x, cache, radial)[0]
    gradient = torch.autograd.grad(output.square().sum(), x)[0]
    assert calls == [True]
    assert torch.isfinite(gradient).all()


def test_short_force_loss_training_matches_reference(monkeypatch):
    """Three SGD steps preserve loss, gradients and weights on the Neo layout."""
    torch.manual_seed(719)
    actual = [
        (torch.randn(2, 17, 320, device="cuda") * 0.1).requires_grad_(),
        (torch.rand(17, 2, device="cuda") * 0.8 + 0.1).requires_grad_(),
        (torch.randn(3, 2, 128, 128, device="cuda") * 0.01).requires_grad_(),
        (torch.randn(3, 2, 192, 192, device="cuda") * 0.01).requires_grad_(),
        (torch.randn(2, 2, 32, 96, device="cuda") * 0.01).requires_grad_(),
    ]
    reference = [t.detach().clone().requires_grad_() for t in actual]
    optimizers = [
        torch.optim.SGD(values[2:], lr=0.01) for values in (actual, reference)
    ]

    def forbidden(*unused):
        raise AssertionError("training entered inference-only ECS")

    monkeypatch.setattr(ecs, "cohort_backward", forbidden)
    for _ in range(3):
        results = []
        for operation, values, optimizer in zip(
            (sv._mixing_stack_op, sv._mixing_stack_reference),
            (actual, reference),
            optimizers,
            strict=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            output = operation(*values, 3, 32, True)[0]
            energy = output.square().mean()
            force = torch.autograd.grad(energy, values[0], create_graph=True)[0]
            loss = energy + 0.01 * force.square().sum()
            gradients = torch.autograd.grad(loss, values)
            results.append((loss.detach(), *[g.detach() for g in gradients]))
            for parameter, gradient in zip(values[2:], gradients[2:], strict=True):
                parameter.grad = gradient
            optimizer.step()
        for got, expected in zip(*results, strict=True):
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
        for got, expected in zip(actual[2:], reference[2:], strict=True):
            torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)
