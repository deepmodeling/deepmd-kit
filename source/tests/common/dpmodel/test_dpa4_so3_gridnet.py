# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity / equivariance tests for the DPA4 ``SO3GridNet``.

``SO3GridNet`` is the capstone of the SO(3)-grid port: it packs
``n_frames = 2 * kmax + 1`` Wigner-D frames along the trailing channel axis,
exercising the general ``n_frames > 1`` ``_to_grid``/``_from_grid`` paths of
``BaseGridNet`` and, in ``mode='cross'``, the ``FrameExpand``/``FrameContract``
seam. Tests mirror the pt ``TestSO3GridNet`` in
``source/tests/pt/model/test_descriptor_sezm_grid_projection.py``.

pt imports live inside the test functions because ruff TID253 bans
module-level ``deepmd.pt`` imports under ``source/tests/common``; pt modules are
pinned to CPU (``.to("cpu")``) under the CUDA-default-device CI.
"""

import copy

import array_api_compat
import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import S2GridNet as DPS2GridNet
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import SO3GridNet as DPSO3GridNet


def _grid_op_param_names(op_type):
    return {
        "glu": (),
        "mlp": ("left_proj", "right_proj", "out_proj"),
        "branch": ("left_proj", "right_proj", "router", "out_proj"),
    }[op_type]


def _build_so3_nets(
    *,
    mode,
    op_type,
    layout,
    lmax=2,
    mmax=None,
    kmax=1,
    channels=4,
    n_focus=1,
    grid_branches=1,
    mlp_bias=False,
    lebedev_precision=None,
    residual_scale_init=None,
    precision="float64",
    seed=7,
):
    """Build a pt + dp ``SO3GridNet`` with identical (perturbed) weights."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import SO3GridNet as PTSO3GridNet

    pt_dtype = {"float64": torch.float64, "float32": torch.float32}[precision]
    common = {
        "lmax": lmax,
        "mmax": mmax,
        "kmax": kmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": mode,
        "op_type": op_type,
        "layout": layout,
        "lebedev_precision": lebedev_precision,
        "coefficient_layout": "packed",
        "grid_branches": grid_branches,
        "residual_scale_init": residual_scale_init,
        "mlp_bias": mlp_bias,
        "trainable": True,
        "seed": seed,
    }
    pt_net = PTSO3GridNet(dtype=pt_dtype, **common).to("cpu")
    rng = np.random.default_rng(2100)
    with torch.no_grad():
        for p in pt_net.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape))).to(p.dtype)

    dp_net = DPSO3GridNet(precision=precision, **common)

    state = {k: v.detach().cpu().numpy() for k, v in pt_net.state_dict().items()}
    expected = {"scalar_gate.weight"}
    if mlp_bias:
        expected.add("scalar_gate.bias")
    expected |= {f"grid_op.{n}.weight" for n in _grid_op_param_names(op_type)}
    if mode == "cross":
        expected |= {"frame_expand.weight", "frame_contract.weight"}
    if residual_scale_init is not None:
        expected.add("residual_scale")
    assert set(state) == expected, set(state)

    dp_net.scalar_gate.weight = state["scalar_gate.weight"]
    if mlp_bias:
        dp_net.scalar_gate.bias = state["scalar_gate.bias"]
    for name in _grid_op_param_names(op_type):
        getattr(dp_net.grid_op, name).weight = state[f"grid_op.{name}.weight"]
    if mode == "cross":
        dp_net.frame_expand.weight = state["frame_expand.weight"]
        dp_net.frame_contract.weight = state["frame_contract.weight"]
    if residual_scale_init is not None:
        dp_net.residual_scale = state["residual_scale"]
    return pt_net, dp_net


def _make_so3_inputs(*, dp_net, mode, layout, n_batch, rng):
    """Build (query, context) for the given mode/layout; context None for self."""
    # D axis is the per-frame coefficient count (frames packed in channels).
    coeff_dim = dp_net.projector.coeff_dim // dp_net.n_frames
    n_focus = dp_net.n_focus
    if mode == "self":
        channels = dp_net.query_channels
        query = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
        return query, None
    # cross: both query and context carry ``context_channels`` (== channels).
    channels = dp_net.context_channels
    if layout == "flat":
        query = rng.normal(size=(n_batch, coeff_dim, n_focus * channels))
        context = rng.normal(size=(n_batch, coeff_dim, n_focus * channels))
    else:  # ndfc
        query = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
        context = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
    return query, context


def _run(net, query, context, backend):
    """Run a net with the given backend; return numpy output."""
    if backend == "pt":
        import torch

        q = torch.from_numpy(query)
        c = None if context is None else torch.from_numpy(context)
        return net(q, c).detach().cpu().numpy()
    out = net.call(query, None if context is None else context)
    return np.asarray(out)


def _rotate_ndfc(x, d_matrix):
    """Rotate coefficient-layout tensors (N, D, F, C) by per-batch (N, D, D)."""
    return np.einsum("nij,njfc->nifc", d_matrix, x)


# === parity =========================================================


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
@pytest.mark.parametrize("kmax", [1, 2])  # frame band width (n_frames = 2*kmax+1)
def test_so3_self_parity(op_type, kmax) -> None:
    """mode='self' SO3GridNet matches pt at 1e-12 (n_frames>1 to/from-grid)."""
    lmax, n_focus, n_batch = 2, 2, 5
    pt_net, dp_net = _build_so3_nets(
        mode="self",
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
    )
    rng = np.random.default_rng(11)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode="self", layout="ndfc", n_batch=n_batch, rng=rng
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
@pytest.mark.parametrize("kmax", [1, 2])  # frame band width (n_frames = 2*kmax+1)
def test_so3_self_equivariance(op_type, kmax) -> None:
    """net(rot(x)) == rot(net(x)) for a shared SO(3) rotation (self mode)."""
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    lmax, n_focus, n_batch = 2, 1, 4
    _, dp_net = _build_so3_nets(
        mode="self",
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
        grid_branches=2,
    )
    rng = np.random.default_rng(33)
    query, _ = _make_so3_inputs(
        dp_net=dp_net, mode="self", layout="ndfc", n_batch=n_batch, rng=rng
    )
    quat = rng.normal(size=(n_batch, 4))
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    d_matrix = np.asarray(WignerDCalculator(lmax, precision="float64")(quat)[0])

    y_rot_in = _run(dp_net, _rotate_ndfc(query, d_matrix), None, "dp")
    y_then_rot = _rotate_ndfc(_run(dp_net, query, None, "dp"), d_matrix)
    np.testing.assert_allclose(y_rot_in, y_then_rot, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
@pytest.mark.parametrize("kmax", [1, 2])  # frame band width (n_frames = 2*kmax+1)
def test_so3_cross_parity(op_type, kmax) -> None:
    """mode='cross' SO3GridNet matches pt at 1e-12 (frame_expand/contract seam)."""
    lmax, n_focus, n_batch = 2, 2, 5
    pt_net, dp_net = _build_so3_nets(
        mode="cross",
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
    )
    rng = np.random.default_rng(22)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode="cross", layout="ndfc", n_batch=n_batch, rng=rng
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    assert dp_out.shape == query.shape
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
def test_so3_cross_equivariance(op_type) -> None:
    """net(rot(q), rot(c)) == rot(net(q, c)) for a shared SO(3) rotation."""
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    lmax, n_focus, n_batch, kmax = 2, 1, 4, 1
    _, dp_net = _build_so3_nets(
        mode="cross",
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
        grid_branches=2,
    )
    rng = np.random.default_rng(44)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode="cross", layout="ndfc", n_batch=n_batch, rng=rng
    )
    quat = rng.normal(size=(n_batch, 4))
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    d_matrix = np.asarray(WignerDCalculator(lmax, precision="float64")(quat)[0])

    y_rot_in = _run(
        dp_net, _rotate_ndfc(query, d_matrix), _rotate_ndfc(context, d_matrix), "dp"
    )
    y_then_rot = _rotate_ndfc(_run(dp_net, query, context, "dp"), d_matrix)
    np.testing.assert_allclose(y_rot_in, y_then_rot, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
def test_so3_cross_flat_parity(op_type) -> None:
    """mode='cross', layout='flat' SO3GridNet matches pt at 1e-12."""
    lmax, n_focus, n_batch, kmax = 2, 3, 5, 2
    pt_net, dp_net = _build_so3_nets(
        mode="cross",
        op_type=op_type,
        layout="flat",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
    )
    rng = np.random.default_rng(55)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode="cross", layout="flat", n_batch=n_batch, rng=rng
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    assert dp_out.shape == query.shape
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
@pytest.mark.parametrize("op_type", ["glu", "mlp"])  # grid operation
def test_so3_fp32_parity(mode, op_type) -> None:
    """fp32 weight-copied SO3GridNet matches pt at ~1e-4.

    The flagship ``examples/water/dpa4/input.json`` runs ``precision:
    float32``. The grid path reduces over many Lebedev quadrature points, so
    fp32 accumulation error is far above the 1-2 ulp budget; the right budget
    is the "computation-in-fp32" one (rtol/atol ~1e-4), not bit-parity.
    """
    lmax, n_focus, n_batch, kmax = 2, 2, 5, 2
    pt_net, dp_net = _build_so3_nets(
        mode=mode,
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
        precision="float32",
    )
    rng = np.random.default_rng(123)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode=mode, layout="ndfc", n_batch=n_batch, rng=rng
    )
    query = query.astype(np.float32)
    if context is not None:
        context = context.astype(np.float32)
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-4, atol=1e-4)


# === serialize ======================================================


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
def test_so3_serialize_roundtrip(mode, op_type) -> None:
    """Serialize -> deserialize -> forward identical; @version == 1."""
    lmax, n_focus, n_batch, kmax = 2, 2, 5, 2
    _, dp_net = _build_so3_nets(
        mode=mode,
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
        residual_scale_init=0.5,
    )
    rng = np.random.default_rng(66)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode=mode, layout="ndfc", n_batch=n_batch, rng=rng
    )
    dp_out = _run(dp_net, query, context, "dp")

    data = dp_net.serialize()
    assert data["@version"] == 1
    assert data["config"]["projector"]["@class"] == "SO3GridProjector"
    assert "residual_scale" in data["@variables"]
    if mode == "cross":
        assert "frame_expand.weight" in data["@variables"]
        assert "frame_contract.weight" in data["@variables"]
    else:
        assert "frame_expand.weight" not in data["@variables"]

    restored = DPSO3GridNet.deserialize(data)
    np.testing.assert_array_equal(restored.residual_scale, dp_net.residual_scale)
    np.testing.assert_allclose(
        _run(restored, query, context, "dp"), dp_out, rtol=1e-12, atol=1e-12
    )


# === torch namespace ================================================


def test_torch_namespace() -> None:
    """cross-mode SO3GridNet.call on torch.from_numpy input matches numpy.

    Guards the frame-axis to/from-grid reshape/permute pitfall: a numpy-only
    bug there (e.g. ``np.einsum`` on a tensor) would diverge here.
    """
    import torch

    lmax, n_focus, n_batch, kmax = 2, 2, 5, 2
    _, dp_net = _build_so3_nets(
        mode="cross",
        op_type="mlp",
        layout="ndfc",
        lmax=lmax,
        kmax=kmax,
        n_focus=n_focus,
        residual_scale_init=0.7,
    )
    rng = np.random.default_rng(77)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode="cross", layout="ndfc", n_batch=n_batch, rng=rng
    )
    np_out = np.asarray(dp_net.call(query, context))
    torch_out = dp_net.call(torch.from_numpy(query), torch.from_numpy(context))
    assert array_api_compat.is_torch_array(torch_out)
    np.testing.assert_allclose(
        np_out, torch_out.detach().cpu().numpy(), rtol=1e-12, atol=1e-12
    )


# === S2 regression (n_frames == 1 path untouched) ===================


def _build_s2_nets(*, mode, op_type, layout, lmax=2, channels=4, n_focus=1):
    """Build a pt + dp ``S2GridNet`` with identical (perturbed) weights."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": mode,
        "op_type": op_type,
        "layout": layout,
        "grid_resolution_list": None,
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "grid_branches": 1,
        "residual_scale_init": None,
        "mlp_bias": False,
        "trainable": True,
        "seed": 7,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common).to("cpu")
    rng = np.random.default_rng(2100)
    with torch.no_grad():
        for p in pt_net.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
    dp_net = DPS2GridNet(precision="float64", **common)
    state = {k: v.detach().cpu().numpy() for k, v in pt_net.state_dict().items()}
    dp_net.scalar_gate.weight = state["scalar_gate.weight"]
    for name in _grid_op_param_names(op_type):
        getattr(dp_net.grid_op, name).weight = state[f"grid_op.{name}.weight"]
    return pt_net, dp_net


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
def test_s2_regression(mode) -> None:
    """An existing S2GridNet (n_frames == 1) still matches pt at 1e-12."""
    lmax, n_focus, n_batch, channels = 2, 2, 5, 4
    pt_net, dp_net = _build_s2_nets(
        mode=mode, op_type="mlp", layout="ndfc", lmax=lmax, n_focus=n_focus
    )
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(88)
    if mode == "self":
        query = rng.normal(size=(n_batch, coeff_dim, n_focus, 2 * channels))
        context = None
    else:
        query = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
        context = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


def test_so3_cross_mixed_precision_runs() -> None:
    """fp32 inputs through an fp64 SO3GridNet cross net run cleanly.

    ``_FrameMixer`` casts its weights to the operand dtype, so operands are
    lifted to compute precision before frame expansion (matching pt's fp64
    FrameExpand); the mixed-precision path must run and stay close to the
    fp64-input result.
    """
    _pt, dp_net = _build_so3_nets(
        mode="cross", op_type="glu", layout="ndfc", precision="float64"
    )
    rng = np.random.default_rng(909)
    query, context = _make_so3_inputs(
        dp_net=dp_net, mode="cross", layout="ndfc", n_batch=3, rng=rng
    )
    out32 = np.asarray(
        dp_net.call(query.astype(np.float32), context.astype(np.float32))
    )
    out64 = np.asarray(dp_net.call(query, context))
    assert np.all(np.isfinite(out32))
    np.testing.assert_allclose(out32, out64, rtol=1e-4, atol=1e-4)


def test_so3_deserialize_rejects_bad_projector() -> None:
    """SO3GridNet.deserialize validates the nested projector @class/@version."""
    _pt, dp_net = _build_so3_nets(mode="self", op_type="glu", layout="ndfc")
    data = dp_net.serialize()
    bad_class = copy.deepcopy(data)
    bad_class["config"]["projector"]["@class"] = "S2GridProjector"
    with pytest.raises(ValueError, match="projector"):
        DPSO3GridNet.deserialize(bad_class)
    bad_ver = copy.deepcopy(data)
    bad_ver["config"]["projector"]["@version"] = 99
    with pytest.raises(ValueError, match="version"):
        DPSO3GridNet.deserialize(bad_ver)
    missing_ver = copy.deepcopy(data)
    del missing_ver["config"]["projector"]["@version"]
    with pytest.raises(ValueError, match="@version"):
        DPSO3GridNet.deserialize(missing_ver)
