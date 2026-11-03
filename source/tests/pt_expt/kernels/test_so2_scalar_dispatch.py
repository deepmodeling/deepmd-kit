# SPDX-License-Identifier: LGPL-3.0-or-later
"""Scalar capture must be selected and consumed, not merely numerically valid."""

from types import (
    SimpleNamespace,
)
from unittest.mock import (
    patch,
)

import pytest
import torch

from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as vp

GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not vp.SO2_VALUE_PATH_TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)


@pytest.fixture
def strict_environment(monkeypatch):
    """Keep the tested route independent of ambient acceleration flags."""
    for name, value in {
        "DP_TRITON_INFER": "2",
        "DP_CUDA_INFER": "0",
        "DP_TRITON_TRAIN": "0",
        "DP_CUDA_TRAIN": "0",
        "DP_CUTE_INFER": "0",
        "DP_CUTILE_INFER": "0",
    }.items():
        monkeypatch.setenv(name, value)
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = old


def test_export_target_scope(monkeypatch):
    """CPU tracing may opt in for CUDA export, but ordinary CPU stays off."""
    conv = SimpleNamespace(
        training=False,
        triton_infer_level=2,
        lmax=3,
        n_focus=2,
        so2_focus_dim=32,
        focus_compete=True,
    )
    x = torch.empty(7, 16, 64, device="cpu")
    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", "1")
    assert not vp._save_competition_scalar(conv, x)
    conv._force_scalar_capture_cuda_target = True
    assert vp._save_competition_scalar(conv, x)
    conv._force_scalar_capture_cuda_target = False
    assert not vp._save_competition_scalar(conv, x)
    conv._force_scalar_capture_cuda_target = True
    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", "0")
    assert not vp._save_competition_scalar(conv, x)


@pytest.mark.parametrize("edges", [0, 1, 17])
def test_cpu_scalar_layout_and_vjp(edges):
    """CPU reference and fake/CUDA layouts share the same output contract."""
    torch.manual_seed(42)
    x = torch.randn(7, 16, 64, device="cpu", requires_grad=True)
    src = torch.arange(edges, device="cpu") % 7
    order = src.argsort(stable=True)
    ptr = torch.cat((src.new_zeros(1), torch.bincount(src, minlength=7).cumsum(0)))
    w = torch.eye(16, device="cpu").repeat(edges, 1, 1).requires_grad_()
    kc = torch.randn(edges, 25, device="cpu", requires_grad=True)
    cb = torch.randn(64, device="cpu", requires_grad=True)
    args = (x, src, order, ptr, w, kc, cb, 3, 2, 1)
    u, scalar = vp._rotate_mix_scalar_op(*args)
    ref = vp._rotate_mix_op(*args)
    torch.testing.assert_close(u, ref, atol=0, rtol=0)
    torch.testing.assert_close(scalar, ref[:, :, :32].permute(1, 0, 2), atol=0, rtol=0)
    assert scalar.stride() == (64, edges * 64, 1)
    if edges:
        assert scalar.untyped_storage().data_ptr() != u.untyped_storage().data_ptr()
    gu, gs = torch.randn_like(u), torch.randn_like(scalar)
    actual = torch.autograd.grad((u, scalar), (x, w, kc, cb), (gu, gs))
    expected = torch.autograd.grad(
        (ref, ref[:, :, :32].permute(1, 0, 2)), (x, w, kc, cb), (gu, gs)
    )
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@GPU
@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("capture", [False, True])
def test_value_path_dispatch(monkeypatch, strict_environment, backend, capture):
    """Exercise the production callable and prove the competition data source."""
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
    path = vp.make_triton_value_path(conv)
    assert path is not None
    x = torch.randn(7, 16, 64, device="cuda") * 0.1
    radial = torch.randn(17, 4, 32, device="cuda") * 0.1
    cache = SimpleNamespace(
        src=torch.arange(17, device="cuda") % 7,
        D_full=torch.eye(16, device="cuda").repeat(17, 1, 1),
    )
    seen = {}
    original = vp._rotate_mix_scalar_op

    def capture_spy(*args):
        u, scalar = original(*args)
        # A distinct sentinel proves provenance: a wrong slice has identical
        # normal values and would otherwise pass an equality-only assertion.
        scalar = scalar + 0.125
        seen["scalar"] = scalar
        return u, scalar

    original_alpha = conv._focus_alpha

    def alpha_spy(gate_src):
        if capture:
            torch.testing.assert_close(gate_src, seen["scalar"], rtol=0, atol=0)
        return original_alpha(gate_src)

    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", str(int(capture)))
    with (
        patch.object(
            vp, "_rotate_mix_scalar_op", side_effect=capture_spy
        ) as scalar_call,
        patch.object(vp, "_rotate_mix_op", wraps=vp._rotate_mix_op) as default_call,
        patch.object(conv, "_focus_alpha", side_effect=alpha_spy) as alpha_call,
    ):
        path(x, cache, radial)
    assert scalar_call.call_count == int(capture)
    assert default_call.call_count == int(not capture)
    assert alpha_call.call_count == 1


def _model_params():
    """A small two-block model with the supported scalar-capture layout."""
    return {
        "type": "SeZM",
        "type_map": ["C"],
        "use_compile": False,
        "descriptor": {
            "type": "SeZM",
            "sel": [32],
            "rcut": 4.0,
            "channels": 32,
            "n_focus": 2,
            "focus_dim": 32,
            "n_radial": 4,
            "radial_mlp": [32],
            "use_env_seed": True,
            "l_schedule": [3, 3, 0],
            "mmax": 1,
            "so2_layers": 3,
            "n_atten_head": 1,
            "ffn_neurons": 32,
            "ffn_blocks": 1,
            "use_amp": False,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {"neuron": [16], "precision": "float32", "seed": 7},
    }


@GPU
def test_model_capture_on_off(monkeypatch, strict_environment):
    """A real frozen model preserves outputs and executes scalar capture."""
    from deepmd.pt.model.model import (
        get_model,
    )

    model = get_model(_model_params()).to("cuda").eval()
    model.requires_grad_(False)
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.2, 0.2, 0.1], [0.3, 1.1, 0.2], [0.1, 0.2, 1.3]]],
        device="cuda",
        dtype=torch.float64,
    )
    atype = torch.zeros(1, 4, device="cuda", dtype=torch.long)
    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", "0")
    reference = model(coord, atype, None)
    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", "1")
    with patch.object(
        vp, "_rotate_mix_scalar_op", wraps=vp._rotate_mix_scalar_op
    ) as spy:
        actual = model(coord, atype, None)
    assert spy.call_count > 0
    assert actual.keys() == reference.keys()
    for name in actual:
        if isinstance(actual[name], torch.Tensor):
            assert torch.isfinite(actual[name]).all()
            torch.testing.assert_close(
                actual[name], reference[name], rtol=1e-4, atol=1e-5
            )


@GPU
@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize(
    "capture,target", [(True, "cuda"), (False, "cuda"), (True, "cpu")]
)
def test_export_dispatch(
    monkeypatch, strict_environment, tmp_path, backend, capture, target
):
    """Both official exporters must select capture before CPU tracing ends.

    Stop at the export boundary, avoiding compiler/toolchain dependencies in
    this routing regression. Full AOTI package inference is benchmarked separately.
    """
    from deepmd.pt.entrypoints import (
        freeze_pt2,
    )
    from deepmd.pt.model.model import (
        get_model,
    )
    from deepmd.pt.train.wrapper import (
        ModelWrapper,
    )
    from deepmd.pt_expt.kernels import (
        autotune,
    )
    from deepmd.pt_expt.utils.serialization import (
        _trace_and_export_impl,
    )

    monkeypatch.setenv("DP_SO2_SCALAR_CAPTURE_INFER", str(int(capture)))
    monkeypatch.setattr(freeze_pt2, "_tune_triton_configs", lambda *args: None)
    monkeypatch.setattr(autotune, "run_autotune", lambda *args: None)
    params = _model_params()
    model = get_model(params).cpu().eval()
    seen = []

    class ExportBoundary(Exception):
        pass

    def inspect_graph(traced, *args, **kwargs):
        targets = [str(node.target) for node in traced.graph.nodes]
        count = sum("sezm_triton.so2_rotate_mix_scalar." in name for name in targets)
        seen.append(count)
        assert (count > 0) == (capture and target == "cuda")
        raise ExportBoundary

    monkeypatch.setattr(torch.export, "export", inspect_graph)
    with pytest.raises(ExportBoundary):
        if backend == "pt":
            checkpoint = tmp_path / "model.pt"
            wrapper = ModelWrapper(model, model_params=params)
            torch.save({"model": wrapper.state_dict()}, checkpoint)
            freeze_pt2.freeze_sezm_to_pt2(
                str(checkpoint),
                str(tmp_path / "model.pt2"),
                device=torch.device(target),
                with_comm=False,
            )
        else:
            _trace_and_export_impl(
                {"model": model.serialize(), "model_def_script": params},
                lower_kind="graph",
                do_atomic_virial=True,
                target_device=torch.device(target),
            )
    assert len(seen) == 1
