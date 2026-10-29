# SPDX-License-Identifier: LGPL-3.0-or-later
"""Independent scalar storage emitted by the existing rotation producer."""

from types import (
    SimpleNamespace,
)

import pytest
import torch

from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as vp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not vp.SO2_VALUE_PATH_TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)


@pytest.fixture(autouse=True)
def _strict_fp32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def _inputs(edges: int, rank: int, zero: bool = False) -> tuple:
    torch.manual_seed(42)
    # Noncontiguous node stride; the existing kernel requires channel stride 1.
    x = (torch.randn(14, 16, 64, device="cuda") * 0.1)[::2].detach()
    if zero:
        x.zero_()
    x.requires_grad_()
    src = torch.arange(edges, device="cuda") % 7
    order = src.argsort(stable=True)
    counts = torch.bincount(src, minlength=7)
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    w = torch.randn(edges, 16, 16, device="cuda") * 0.1
    mask = torch.zeros(16, 16, device="cuda")
    for degree in range(4):
        mask[degree**2 : (degree + 1) ** 2, degree**2 : (degree + 1) ** 2] = 1
    w = (w * mask).requires_grad_()
    shape = (edges, 4, 64) if rank == 0 else (edges, 25 * rank)
    kc = (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
    cb = torch.randn(1 if rank == 0 else rank * 64, device="cuda") * 0.1
    cb.requires_grad_(rank > 0)
    return x, src, order, ptr, w, kc, cb, 3, 2, rank


@pytest.mark.parametrize("edges,rank", [(0, 1), (1, 0), (17, 1), (257, 2)])
@pytest.mark.parametrize("branch", ["both", "main", "scalar"])
@pytest.mark.parametrize("zero", [False, True])
def test_capture_and_vjp(edges: int, rank: int, branch: str, zero: bool) -> None:
    args = _inputs(edges, rank, zero)
    original = vp._rotate_mix_op(*args)
    expected = (original, original[:, :, :32].permute(1, 0, 2))
    actual = vp._rotate_mix_scalar_op(*args)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    if edges:
        assert (
            actual[0].untyped_storage().data_ptr()
            != actual[1].untyped_storage().data_ptr()
        )
    assert actual[1].stride() == (64, edges * 64, 1)
    chosen = {"both": (0, 1), "main": (0,), "scalar": (1,)}[branch]
    upstream = tuple(torch.randn_like(expected[i]) for i in chosen)
    leaves = (args[0], args[4], args[5]) + ((args[6],) if rank else ())
    ref_grad = torch.autograd.grad(tuple(expected[i] for i in chosen), leaves, upstream)
    got_grad = torch.autograd.grad(tuple(actual[i] for i in chosen), leaves, upstream)
    for a, b in zip(got_grad, ref_grad, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_vjp_can_be_differentiated() -> None:
    """Preserve the existing rotation's differentiable backward contract."""
    args = _inputs(17, 1, False)
    original = vp._rotate_mix_op(*args)
    reference = (original, original[:, :, :32].permute(1, 0, 2))
    actual = vp._rotate_mix_scalar_op(*args)

    def derivatives(outputs):
        loss = sum(v.square().sum() for v in outputs)
        first = torch.autograd.grad(loss, args[0], create_graph=True)[0]
        second = torch.autograd.grad(first.square().sum(), args[0])[0]
        return first, second

    expected_grads = derivatives(reference)
    actual_grads = derivatives(actual)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)


def test_float64_reference() -> None:
    args = _inputs(17, 2)
    reference_args = tuple(
        v.detach().double().requires_grad_(v.requires_grad)
        if isinstance(v, torch.Tensor) and v.is_floating_point()
        else v
        for v in args
    )
    # Wigner's off-degree entries are structural zeros, not independent
    # variables. Project the dense reference onto the same tangent space.
    block_mask = torch.zeros(16, 16, device="cuda", dtype=torch.float64)
    for degree in range(4):
        block_mask[degree**2 : (degree + 1) ** 2, degree**2 : (degree + 1) ** 2] = 1
    ref = vp._rotate_mix_reference(
        reference_args[0],
        reference_args[1],
        reference_args[4] * block_mask,
        reference_args[5],
        reference_args[6],
        3,
        2,
        2,
    )
    got, scalar = vp._rotate_mix_scalar_op(*args)
    torch.testing.assert_close(got.double(), ref, rtol=1e-4, atol=1e-5)
    g, gs = torch.randn_like(got), torch.randn_like(scalar)
    actual = torch.autograd.grad(
        (got, scalar), (args[0], args[4], args[5], args[6]), (g, gs)
    )
    expected = torch.autograd.grad(
        (ref, ref[:, :, :32].permute(1, 0, 2)),
        tuple(reference_args[i] for i in (0, 4, 5, 6)),
        (g.double(), gs.double()),
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=1e-4, atol=1e-5)


def test_schema_and_fake() -> None:
    torch.library.opcheck(
        vp._rotate_mix_scalar_op,
        _inputs(17, 1),
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )


@pytest.mark.parametrize("capture", [False, True])
def test_dynamic_compile(capture: bool) -> None:
    op = vp._rotate_mix_scalar_op if capture else vp._rotate_mix_op
    compiled = torch.compile(op, fullgraph=True, dynamic=True)
    for edges in (17, 33):
        args = _inputs(edges, 1)
        actual = compiled(*args)
        expected = op(*args)
        if not capture:
            actual, expected = (actual,), (expected,)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        g = tuple(torch.randn_like(t) for t in expected)
        leaves = tuple(args[i] for i in (0, 4, 5))
        a = torch.autograd.grad(actual, leaves, g)
        b = torch.autograd.grad(expected, leaves, g)
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training", True),
        ("triton_infer_level", 1),
        ("triton_infer_level", 3),
        ("lmax", 2),
        ("n_focus", 1),
        ("so2_focus_dim", 64),
        ("focus_compete", False),
    ],
)
def test_scope(monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    conv = SimpleNamespace(
        training=False,
        triton_infer_level=2,
        lmax=3,
        n_focus=2,
        so2_focus_dim=32,
        focus_compete=True,
    )
    x = torch.empty(1, 16, 64, device="cuda")
    monkeypatch.delenv("DP_SO2_SCALAR_CAPTURE_INFER", raising=False)
    assert not vp._save_competition_scalar(conv, x)
    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", "1")
    assert vp._save_competition_scalar(conv, x)
    assert not vp._save_competition_scalar(conv, x.cpu())
    assert not vp._save_competition_scalar(conv, x.double())
    setattr(conv, field, value)
    assert not vp._save_competition_scalar(conv, x)
