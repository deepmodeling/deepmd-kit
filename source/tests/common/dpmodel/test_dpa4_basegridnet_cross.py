# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity / equivariance tests for the generalized DPA4 ``BaseGridNet``.

These cover the ``mode='cross'``, ``layout='flat'`` and ``residual_scale_init``
paths that ``BaseGridNet`` gained when it was generalized to mirror the current
pt ``deepmd.pt.model.descriptor.sezm_nn.grid_net.BaseGridNet``. All tests use
``S2GridNet`` (``n_frames == 1``); the ``n_frames > 1`` SO(3) frame contraction
in ``_to_grid``/``_from_grid`` is exercised structurally by the SO(3) port
(verified there). pt imports live inside the test functions because ruff TID253
bans module-level ``deepmd.pt`` imports under ``source/tests/common``; pt
modules are pinned to CPU (``.to("cpu")``) under the CUDA-default-device CI.
"""

import array_api_compat
import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    S2GridNet as DPS2GridNet,
)


def _grid_op_param_names(op_type):
    return {
        "glu": (),
        "mlp": ("left_proj", "right_proj", "out_proj"),
        "branch": ("left_proj", "right_proj", "router", "out_proj"),
    }[op_type]


def _build_nets(
    *,
    mode,
    op_type,
    layout,
    residual_scale_init=None,
    lmax=2,
    channels=4,
    n_focus=1,
    grid_branches=1,
    mlp_bias=False,
    grid_resolution_list=None,
    seed=7,
):
    """Build a pt + dp ``S2GridNet`` with identical (perturbed) weights."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        S2GridNet as PTS2GridNet,
    )

    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": mode,
        "op_type": op_type,
        "layout": layout,
        "grid_resolution_list": grid_resolution_list,
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "grid_branches": grid_branches,
        "residual_scale_init": residual_scale_init,
        "mlp_bias": mlp_bias,
        "trainable": True,
        "seed": seed,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common).to("cpu")
    rng = np.random.default_rng(2100)
    with torch.no_grad():
        for p in pt_net.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    dp_net = DPS2GridNet(precision="float64", **common)

    state = {k: v.detach().cpu().numpy() for k, v in pt_net.state_dict().items()}
    expected = {"scalar_gate.weight"}
    if mlp_bias:
        expected.add("scalar_gate.bias")
    expected |= {f"grid_op.{n}.weight" for n in _grid_op_param_names(op_type)}
    if residual_scale_init is not None:
        expected.add("residual_scale")
    assert set(state) == expected, set(state)

    dp_net.scalar_gate.weight = state["scalar_gate.weight"]
    if mlp_bias:
        dp_net.scalar_gate.bias = state["scalar_gate.bias"]
    for name in _grid_op_param_names(op_type):
        getattr(dp_net.grid_op, name).weight = state[f"grid_op.{name}.weight"]
    if residual_scale_init is not None:
        dp_net.residual_scale = state["residual_scale"]
    return pt_net, dp_net


def _coeff_dim(lmax):
    return (lmax + 1) ** 2


def _make_inputs(*, mode, layout, n_batch, lmax, n_focus, channels, rng):
    """Build (query, context) for the given mode/layout. context is None for self."""
    coeff_dim = _coeff_dim(lmax)
    if mode == "self":
        if layout == "nfdc":
            query = rng.normal(size=(n_batch, n_focus, coeff_dim, 2 * channels))
        else:  # ndfc
            query = rng.normal(size=(n_batch, coeff_dim, n_focus, 2 * channels))
        return query, None
    # cross
    if layout == "flat":
        query = rng.normal(size=(n_batch, coeff_dim, n_focus * channels))
        context = rng.normal(size=(n_batch, coeff_dim, n_focus * channels))
    elif layout == "nfdc":
        query = rng.normal(size=(n_batch, n_focus, coeff_dim, channels))
        context = rng.normal(size=(n_batch, n_focus, coeff_dim, channels))
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


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
def test_s2_self_regression(op_type) -> None:
    """mode='self' S2GridNet still matches pt at 1e-12 (guards the self path)."""
    lmax, n_focus, n_batch = 2, 2, 5
    pt_net, dp_net = _build_nets(
        mode="self", op_type=op_type, layout="ndfc", lmax=lmax, n_focus=n_focus
    )
    rng = np.random.default_rng(11)
    query, context = _make_inputs(
        mode="self",
        layout="ndfc",
        n_batch=n_batch,
        lmax=lmax,
        n_focus=n_focus,
        channels=dp_net.channels,
        rng=rng,
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
def test_s2_cross_parity(op_type) -> None:
    """mode='cross' S2GridNet matches pt at 1e-12 (separate query/context)."""
    lmax, n_focus, n_batch = 2, 2, 5
    pt_net, dp_net = _build_nets(
        mode="cross", op_type=op_type, layout="ndfc", lmax=lmax, n_focus=n_focus
    )
    rng = np.random.default_rng(22)
    query, context = _make_inputs(
        mode="cross",
        layout="ndfc",
        n_batch=n_batch,
        lmax=lmax,
        n_focus=n_focus,
        channels=dp_net.channels,
        rng=rng,
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    assert dp_out.shape == query.shape
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


def _rotate_ndfc(x, d_matrix):
    """Rotate coefficient-layout tensors (N, D, F, C) by per-batch (N, D, D)."""
    return np.einsum("nij,njfc->nifc", d_matrix, x)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operation
def test_s2_cross_equivariance(op_type) -> None:
    """net(rot(q), rot(c)) == rot(net(q, c)) for a shared SO(3) rotation.

    The default Lebedev grid has algebraic precision >= ``3 * lmax``, so the
    degree-``2 * lmax`` grid product integrates exactly and the grid net is
    equivariant to machine precision.
    """
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    lmax, n_focus, n_batch, channels = 2, 1, 4, 4
    _, dp_net = _build_nets(
        mode="cross",
        op_type=op_type,
        layout="ndfc",
        lmax=lmax,
        n_focus=n_focus,
        channels=channels,
    )
    rng = np.random.default_rng(33)
    query, context = _make_inputs(
        mode="cross",
        layout="ndfc",
        n_batch=n_batch,
        lmax=lmax,
        n_focus=n_focus,
        channels=channels,
        rng=rng,
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
def test_layout_flat_parity(op_type) -> None:
    """mode='cross', layout='flat' matches pt at 1e-12."""
    lmax, n_focus, n_batch = 2, 3, 5
    pt_net, dp_net = _build_nets(
        mode="cross", op_type=op_type, layout="flat", lmax=lmax, n_focus=n_focus
    )
    rng = np.random.default_rng(44)
    query, context = _make_inputs(
        mode="cross",
        layout="flat",
        n_batch=n_batch,
        lmax=lmax,
        n_focus=n_focus,
        channels=dp_net.channels,
        rng=rng,
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    assert dp_out.shape == query.shape
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("residual_scale_init", [None, 0.5])  # residual scale init
def test_residual_scale_parity(residual_scale_init) -> None:
    """residual_scale_init parity vs pt at 1e-12; residual_scale (de)serialized."""
    lmax, n_focus, n_batch = 2, 2, 5
    pt_net, dp_net = _build_nets(
        mode="cross",
        op_type="glu",
        layout="ndfc",
        lmax=lmax,
        n_focus=n_focus,
        residual_scale_init=residual_scale_init,
    )
    rng = np.random.default_rng(55)
    query, context = _make_inputs(
        mode="cross",
        layout="ndfc",
        n_batch=n_batch,
        lmax=lmax,
        n_focus=n_focus,
        channels=dp_net.channels,
        rng=rng,
    )
    dp_out = _run(dp_net, query, context, "dp")
    pt_out = _run(pt_net, query, context, "pt")
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)

    # serialize -> deserialize keeps residual_scale and the forward output
    data = dp_net.serialize()
    assert data["config"]["residual_scale_init"] == residual_scale_init
    if residual_scale_init is None:
        assert "residual_scale" not in data["@variables"]
    else:
        assert "residual_scale" in data["@variables"]
    restored = DPS2GridNet.deserialize(data)
    if residual_scale_init is None:
        assert restored.residual_scale is None
    else:
        np.testing.assert_array_equal(restored.residual_scale, dp_net.residual_scale)
    np.testing.assert_allclose(
        _run(restored, query, context, "dp"), dp_out, rtol=1e-12, atol=1e-12
    )


def test_torch_namespace() -> None:
    """cross-mode S2GridNet.call on torch.from_numpy input matches numpy result."""
    import torch

    lmax, n_focus, n_batch, channels = 2, 2, 5, 4
    _, dp_net = _build_nets(
        mode="cross",
        op_type="mlp",
        layout="ndfc",
        lmax=lmax,
        n_focus=n_focus,
        channels=channels,
        residual_scale_init=0.7,
    )
    rng = np.random.default_rng(66)
    query, context = _make_inputs(
        mode="cross",
        layout="ndfc",
        n_batch=n_batch,
        lmax=lmax,
        n_focus=n_focus,
        channels=channels,
        rng=rng,
    )
    np_out = np.asarray(dp_net.call(query, context))
    torch_out = dp_net.call(torch.from_numpy(query), torch.from_numpy(context))
    assert array_api_compat.is_torch_array(torch_out)
    np.testing.assert_allclose(
        np_out, torch_out.detach().cpu().numpy(), rtol=1e-12, atol=1e-12
    )
