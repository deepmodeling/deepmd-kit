# SPDX-License-Identifier: LGPL-3.0-or-later
"""Owned residual workspace reuse, without mutating saved or caller tensors."""

from unittest.mock import (
    patch,
)

import pytest
import torch

from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as vp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not vp.SO2_VALUE_PATH_TRITON_AVAILABLE,
    reason="CUDA and the SO2 Triton value path are required",
)


def _inputs(
    edges: int, gates: int, lmax: int, focus: int, width: int, apply_alpha: bool
) -> tuple:
    """Make well-conditioned operands with nonzero residual and matrix terms."""
    torch.manual_seed(42)
    row = (3 * lmax + 1) * width
    m0, m1 = (lmax + 1) * width, 2 * lmax * width
    initial = torch.randn(focus, edges, row, device="cuda") * 0.1
    alpha = torch.rand(edges, focus, device="cuda") + 0.2
    weight0 = torch.randn(gates + 1, focus, m0, m0, device="cuda") * 0.02
    weight1 = torch.randn(gates + 1, focus, m1, m1, device="cuda") * 0.02
    gate = torch.randn(gates, focus, width, lmax * width, device="cuda") * 0.02
    output, saved, final = vp._mixing_stack_impl(
        initial, alpha, weight0, weight1, gate, lmax, width, apply_alpha
    )
    return (
        torch.randn_like(output),
        output,
        saved,
        final,
        alpha,
        weight0.transpose(2, 3).contiguous(),
        weight1.transpose(2, 3).contiguous(),
        gate,
        gate.transpose(2, 3).contiguous(),
    )


@pytest.mark.parametrize(
    "shape", [(2, 2, 32), (3, 2, 32), (3, 1, 64), (6, 2, 96), (6, 3, 128)]
)
@pytest.mark.parametrize("edges", [0, 37, 129])
@pytest.mark.parametrize("gates", [0, 1, 3])
@pytest.mark.parametrize("apply_alpha", [False, True])
def test_backward_matches_reference_without_mutation(
    shape: tuple[int, int, int], edges: int, gates: int, apply_alpha: bool
) -> None:
    """Cover empty edges, partial tiles, wide channels and residual recurrences."""
    lmax, focus, width = shape
    inputs = _inputs(edges, gates, lmax, focus, width, apply_alpha)
    snapshots = [value.clone() for value in inputs]
    actual = vp._mixing_stack_bwd_impl(*inputs, lmax, width, apply_alpha)
    assert actual[0].shape == (focus, edges, (3 * lmax + 1) * width)
    assert actual[1].shape == (edges, focus)
    if edges:
        expected = vp._mixing_stack_backward_reference(
            *inputs, None, None, lmax, width, apply_alpha
        )
        torch.testing.assert_close(actual[0], expected[0], atol=5e-5, rtol=1e-4)
        if apply_alpha:
            torch.testing.assert_close(actual[1], expected[1], atol=5e-5, rtol=1e-4)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize(
    "with_weights,keep", [(False, False), (True, False), (False, True), (True, True)]
)
def test_reuse_only_without_weights_or_saved_state(
    with_weights: bool, keep: bool
) -> None:
    """Check allocation identity at real GEMM launches, including training paths."""
    inputs = _inputs(37, 2, 3, 2, 32, True)
    original_wrap = vp.wrap_triton
    aliases = []

    class LaunchProbe:
        def __init__(self, wrapped) -> None:
            self.wrapped = wrapped

        def __getitem__(self, grid):
            launch = self.wrapped[grid]

            def record(*args, **kwargs):
                # Residual is arg1, independent GEMM input arg0, output arg5.
                aliases.append(
                    (
                        args[1].data_ptr() == args[5].data_ptr(),
                        args[0].data_ptr() == args[5].data_ptr(),
                    )
                )
                return launch(*args, **kwargs)

            return record

    def observed_wrap(kernel):
        wrapped = original_wrap(kernel)
        return LaunchProbe(wrapped) if kernel is vp._stack_gemm_bwd_kernel else wrapped

    with patch.object(vp, "wrap_triton", observed_wrap):
        actual = vp._stack_backward_traversal(
            *inputs,
            None,
            None,
            3,
            32,
            True,
            with_weights=with_weights,
            keep=keep,
        )
    assert len(aliases) == 3
    assert aliases[0] == (False, False)
    assert aliases[1:] == [(not with_weights and not keep, False)] * 2
    assert (actual[2] is not None) is with_weights
    assert (actual[3] is not None) is keep


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("apply_alpha", [False, True])
def test_reuse_matches_separate_workspace(
    dtype: torch.dtype, apply_alpha: bool
) -> None:
    """Reuse must not change arithmetic in any supported working precision."""
    inputs = tuple(value.to(dtype) for value in _inputs(129, 3, 3, 2, 32, apply_alpha))
    snapshots = [value.clone() for value in inputs]
    actual = vp._stack_backward_traversal(
        *inputs, None, None, 3, 32, apply_alpha, with_weights=False, keep=False
    )
    # Keeping per-layer state uses distinct buffers and identical GEMM arithmetic.
    expected = vp._stack_backward_traversal(
        *inputs, None, None, 3, 32, apply_alpha, with_weights=False, keep=True
    )
    count = 2 if apply_alpha else 1
    for result, reference in zip(actual[:count], expected[:count], strict=True):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)


@pytest.mark.parametrize(
    "trainable_weights",
    [(True, True, True), (True, False, False), (False, False, False)],
)
@pytest.mark.parametrize("lmax,width", [(2, 32), (6, 96)])
def test_force_loss_and_updates_with_frozen_weights(
    trainable_weights: tuple[bool, bool, bool], lmax: int, width: int
) -> None:
    """Compare first/second derivatives and two updates with the dense reference."""
    torch.manual_seed(19)
    edges, focus, gates = 7, 2, 2
    row, m0, m1 = (3 * lmax + 1) * width, (lmax + 1) * width, 2 * lmax * width
    values = (
        torch.randn(focus, edges, row, device="cuda") * 0.1,
        torch.rand(edges, focus, device="cuda") + 0.2,
        torch.randn(gates + 1, focus, m0, m0, device="cuda") * 0.02,
        torch.randn(gates + 1, focus, m1, m1, device="cuda") * 0.02,
        torch.randn(gates, focus, width, lmax * width, device="cuda") * 0.02,
    )
    requires = (True, True, *trainable_weights)
    actual_inputs = [
        value.clone().requires_grad_(enabled)
        for value, enabled in zip(values, requires, strict=True)
    ]
    reference_inputs = [
        value.clone().requires_grad_(enabled)
        for value, enabled in zip(values, requires, strict=True)
    ]
    for _ in range(2):
        gradients = []
        for operation, operands in (
            (vp._mixing_stack_op, actual_inputs),
            (vp._mixing_stack_reference, reference_inputs),
        ):
            output = operation(*operands, lmax, width, True)[0]
            energy = output.square().sum() / edges
            force = torch.autograd.grad(energy, operands[0], create_graph=True)[0]
            objective = energy + force.square().sum()
            leaves = [value for value in operands if value.requires_grad]
            gradients.append(torch.autograd.grad(objective, leaves))
        for actual, expected in zip(*gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        with torch.no_grad():
            for operands, updates in zip(
                (actual_inputs, reference_inputs), gradients, strict=True
            ):
                leaves = [value for value in operands if value.requires_grad]
                for value, gradient in zip(leaves, updates, strict=True):
                    value.add_(gradient, alpha=-0.01)
        for actual, expected in zip(actual_inputs, reference_inputs, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
