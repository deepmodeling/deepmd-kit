# SPDX-License-Identifier: LGPL-3.0-or-later
"""Selection and bounded scratch contracts, without executing GPU kernels."""

from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

import pytest
import torch
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
)

pytest.importorskip("triton")

from deepmd.pt_expt.kernels.triton.sezm import so2_edge_cohort as ecs
from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as sv


@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("level", [None, "0", "1", "2", "3"])
def test_existing_inference_levels(monkeypatch, backend, level):
    """Real constructors read the existing control before binding the stack."""
    for name in (
        "DP_TRITON_TRAIN",
        "DP_CUDA_INFER",
        "DP_CUDA_TRAIN",
        "DP_CUTE_INFER",
        "DP_CUTILE_INFER",
    ):
        monkeypatch.setenv(name, "0")
    if level is None:
        monkeypatch.delenv("DP_TRITON_INFER", raising=False)
    else:
        monkeypatch.setenv("DP_TRITON_INFER", level)
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
        .cpu()
        .eval()
    )
    assert conv.triton_infer_level == int(level or 0)
    path = conv._triton_value_path
    if int(level or 0) < 2:
        assert path is None
    else:
        assert isinstance(path, sv._TritonSO2ValuePath)
        if level == "2":
            assert path._stack_op is sv._mixing_stack_op
        elif sv.stack_fp16x3_configs(32, 3) is not None:
            assert path._stack_op is not sv._mixing_stack_op


class _LaunchRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def record(*args, **kwargs):
            self.calls.append((args, kwargs))

        return record


@pytest.mark.parametrize("changed", [1, 5, 6, 7])
@pytest.mark.parametrize("kind", ["precision", "stride"])
def test_unsupported_parameters_keep_native(monkeypatch, changed, kind):
    """FP32 outputs alone do not qualify every parameter dtype or layout."""

    def forbidden(*args):
        raise AssertionError("mixed-precision inference entered ECS")

    monkeypatch.setattr(ecs, "cohort_backward", forbidden)
    with FakeTensorMode():
        go = torch.empty(37, 2, 320, device="cuda")
        u = torch.empty(2, 37, 320, device="cuda")
        alpha = torch.empty(37, 2, device="cuda")
        z = torch.empty(2, 2, 37, 320, device="cuda")
        w0 = torch.empty(3, 2, 128, 128, device="cuda")
        w1 = torch.empty(3, 2, 192, 192, device="cuda")
        gw = torch.empty(2, 2, 32, 96, device="cuda")
        saved = [u, alpha, u, z, u, w0, w1, gw]
        if kind == "precision":
            saved[changed] = saved[changed].to(torch.bfloat16)
        else:
            saved[changed] = (
                saved[changed].transpose(-2, -1).contiguous().transpose(-2, -1)
            )
            assert not saved[changed].is_contiguous()
        ctx = SimpleNamespace(
            saved_tensors=saved,
            needs_input_grad=(True, True, False, False, False, False, False, False),
            lmax=3,
            focus_dim=32,
            apply_alpha=True,
        )
        with (
            patch.object(
                sv,
                "_mixing_stack_bwd_op",
                return_value=(torch.empty_like(u), torch.empty_like(alpha)),
            ) as native,
            torch.no_grad(),
        ):
            sv._mixing_stack_backward(ctx, go, None, None)
        native.assert_called_once()


@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("edges", [0, 37, 8193, 16385])
def test_scratch_is_cohort_bounded_and_reused(monkeypatch, inside, edges):
    """Check allocation sizes/identities at live launches using fake CUDA inputs.

    This tests the real Python traversal and eligibility checks, but substitutes
    launch recorders for kernels. Numerical GPU coverage lives in the companion
    test file. A full-graph scratch allocation or per-cohort reallocation fails.
    """
    gemm, point, recompute = (_LaunchRecorder() for _ in range(3))
    monkeypatch.setattr(sv, "_stack_gemm_bwd_kernel", gemm)
    monkeypatch.setattr(sv, "_stack_point_bwd_kernel", point)
    monkeypatch.setattr(sv, "_stack_recompute_kernel", recompute)
    monkeypatch.setattr(
        sv, "stack_fp32_configs", lambda *args: ((32, 64, 32, 4, 2),) * 3
    )
    monkeypatch.setattr(
        sv,
        "_point_backward_schedule",
        lambda *args, **kw: (inside, (16, 4, 2), (16, 4, 2)),
    )
    with FakeTensorMode():
        go = torch.empty(edges, 2, 320, device="cuda")
        z = torch.empty(2, 2, edges, 320, device="cuda")
        alpha = torch.empty(edges, 2, device="cuda")
        w0 = torch.empty(3, 2, 128, 128, device="cuda")
        w1 = torch.empty(3, 2, 192, 192, device="cuda")
        gw = torch.empty(2, 2, 32, 96, device="cuda")
        gwt = torch.empty(2, 2, 96, 32, device="cuda")
        allocated = []
        original = torch.Tensor.new_empty

        def observe(tensor, shape, *args, **kwargs):
            result = original(tensor, shape, *args, **kwargs)
            allocated.append(result)
            return result

        with (
            patch.object(torch.Tensor, "new_empty", observe),
            patch.object(torch, "empty_like", wraps=torch.empty_like) as like,
        ):
            output = ecs.cohort_input_backward_impl(
                go, z, alpha, w0, w1, gw, gwt, ecs.COHORT_EDGES
            )
        assert tuple(output.shape) == (2, edges, 320)
        if not edges:
            assert [tuple(t.shape) for t in allocated] == [(2, 0, 320)]
            assert like.call_count == 0
            assert not gemm.calls
            return
        cap = min(edges, ecs.COHORT_EDGES)
        expected = [(2, edges, 320)]
        if edges > cap:
            expected.append((2, cap, 320))
        expected.append((2, cap, 96))
        assert [tuple(t.shape) for t in allocated] == expected
        residual = allocated[1] if edges > cap else output
        sigmoid = allocated[-1]
        assert like.call_count == 1
        assert like.call_args.args[0] is residual
        chunks = (edges + cap - 1) // cap
        assert len(point.calls) == chunks * 2
        assert len(recompute.calls) == (0 if inside else chunks * 2)
        scratch_ids = {
            (id(args[0]), id(args[2]), id(args[4])) for args, _ in point.calls
        }
        assert len(scratch_ids) == 1
        assert next(iter(scratch_ids))[:2] == (id(residual), id(sigmoid))
        assert all(tuple(args[4].shape) == (2, cap, 320) for args, _ in point.calls)
        assert all(args[8] <= cap for args, _ in point.calls)
