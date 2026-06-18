# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity / equivariance tests for the DPA4 ``SO3GridNet`` (self + cross).

Covers the dpmodel port of the SO(3) Wigner-D grid net, including the
``mode='cross'`` frame machinery (``FrameExpand``/``FrameContract``,
``n_frames > 1``) and the ``layout='flat'`` frame-width path.  All pt imports
are kept inside the test functions (ruff TID253).

Test menu:

* ``test_so3_self_parity`` -- self-mode weight-copied fp64 parity (glu/mlp/branch, kmax 1/2).
* ``test_so3_self_equivariance`` -- rotate input, SO(3) equivariance.
* ``test_so3_cross_parity`` -- cross-mode weight-copied fp64 parity.
* ``test_so3_cross_equivariance`` -- rotate query & context, SO(3) equivariance.
* ``test_so3_cross_flat_parity`` -- ``layout='flat'`` frame-width parity.
* ``test_so3_serialize_roundtrip`` -- serialize/deserialize forward identical.
* ``test_torch_namespace`` -- ``.call`` on torch inputs matches numpy.
* ``test_s2_regression`` -- the existing S2GridNet self+cross still matches pt.
"""

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
)

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    S2GridNet,
    SO3GridNet,
)


def _rotate_ndfc(x: np.ndarray, d_matrix: np.ndarray) -> np.ndarray:
    """Rotate coefficient-layout tensors with shape ``(N, D, F, C)``."""
    return np.einsum("nij,njfc->nifc", d_matrix, x)


def _random_quaternion(n_batch: int, seed: int) -> np.ndarray:
    """Sample normalized quaternions in ``(w, x, y, z)`` order."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_batch, 4)).astype(np.float64)
    return q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True))


def _copy_so3gridnet(dp_net: SO3GridNet, pt_net) -> None:
    """Copy a pt ``SO3GridNet`` state-dict into the dpmodel net."""
    sd = pt_net.state_dict()

    def _np(key):
        return sd[key].detach().cpu().numpy().astype(np.float64)

    dp_net.scalar_gate.weight = _np("scalar_gate.weight")
    if "scalar_gate.bias" in sd:
        dp_net.scalar_gate.bias = _np("scalar_gate.bias").reshape(
            dp_net.scalar_gate.bias.shape
        )
    if dp_net.op_type == "mlp":
        for name in ("left_proj", "right_proj", "out_proj"):
            getattr(dp_net.grid_op, name).weight = _np(f"grid_op.{name}.weight")
    elif dp_net.op_type == "branch":
        for name in ("left_proj", "right_proj", "router", "out_proj"):
            getattr(dp_net.grid_op, name).weight = _np(f"grid_op.{name}.weight")
    if dp_net.frame_expand is not None:
        dp_net.frame_expand.weight = _np("frame_expand.weight")
    if dp_net.frame_contract is not None:
        dp_net.frame_contract.weight = _np("frame_contract.weight")
    if dp_net.residual_scale is not None:
        dp_net.residual_scale = _np("residual_scale").reshape(
            dp_net.residual_scale.shape
        )


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
@pytest.mark.parametrize("kmax", [1, 2])  # frame-index half-width
def test_so3_self_parity(op_type, kmax) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import SO3GridNet as PTSO3GridNet

    lmax, channels, n_focus = 2, 2, 1
    common = {
        "lmax": lmax,
        "kmax": kmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "self",
        "op_type": op_type,
        "layout": "ndfc",
        "coefficient_layout": "packed",
        "grid_branches": 2,
        "trainable": False,
        "seed": 17 + kmax,
    }
    pt_net = PTSO3GridNet(dtype=torch.float64, **common).to("cpu")
    dp_net = SO3GridNet(precision="float64", **common)
    _copy_so3gridnet(dp_net, pt_net)

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(2024 + kmax)
    x = rng.standard_normal(
        (n_batch, coeff_dim, n_focus, dp_net.query_channels)
    ).astype(np.float64)

    dp_out = np.asarray(dp_net.call(x))
    pt_out = pt_net(torch.from_numpy(x)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, dp_net.output_channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp"])  # grid op
@pytest.mark.parametrize("kmax", [1, 2])  # frame-index half-width
def test_so3_self_equivariance(op_type, kmax) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    lmax, channels, n_focus = 2, 2, 1
    dp_net = SO3GridNet(
        lmax=lmax,
        kmax=kmax,
        channels=channels,
        n_focus=n_focus,
        mode="self",
        op_type=op_type,
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_branches=2,
        trainable=False,
        seed=31 + kmax,
    )

    n_batch = 2
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(4096 + kmax)
    x = rng.standard_normal(
        (n_batch, coeff_dim, n_focus, dp_net.query_channels)
    ).astype(np.float64)

    quat = _random_quaternion(n_batch, seed=77 + kmax)
    d_matrix, _ = WignerDCalculator(lmax=lmax, precision="float64").call(quat)
    d_matrix = np.asarray(d_matrix)

    y_rotated_input = np.asarray(dp_net.call(_rotate_ndfc(x, d_matrix)))
    y_then_rotated = _rotate_ndfc(np.asarray(dp_net.call(x)), d_matrix)
    max_error = float(np.max(np.abs(y_rotated_input - y_then_rotated)))
    assert max_error <= 1e-10, f"equivariance error {max_error}"


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
@pytest.mark.parametrize("kmax", [1, 2])  # frame-index half-width
def test_so3_cross_parity(op_type, kmax) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import SO3GridNet as PTSO3GridNet

    lmax, channels, n_focus = 2, 2, 1
    common = {
        "lmax": lmax,
        "kmax": kmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "cross",
        "op_type": op_type,
        "layout": "ndfc",
        "coefficient_layout": "packed",
        "grid_branches": 2,
        "trainable": False,
        "seed": 41 + kmax,
    }
    pt_net = PTSO3GridNet(dtype=torch.float64, **common).to("cpu")
    dp_net = SO3GridNet(precision="float64", **common)
    _copy_so3gridnet(dp_net, pt_net)

    n_batch = 3
    coeff_dim = dp_net.projector.coeff_dim // dp_net.n_frames
    rng = np.random.default_rng(606 + kmax)
    q = rng.standard_normal(
        (n_batch, coeff_dim, n_focus, dp_net.context_channels)
    ).astype(np.float64)
    c = rng.standard_normal(
        (n_batch, coeff_dim, n_focus, dp_net.context_channels)
    ).astype(np.float64)

    dp_out = np.asarray(dp_net.call(q, c))
    pt_out = pt_net(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, dp_net.output_channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp"])  # grid op
@pytest.mark.parametrize("kmax", [1, 2])  # frame-index half-width
def test_so3_cross_equivariance(op_type, kmax) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    lmax, channels, n_focus = 2, 2, 1
    dp_net = SO3GridNet(
        lmax=lmax,
        kmax=kmax,
        channels=channels,
        n_focus=n_focus,
        mode="cross",
        op_type=op_type,
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_branches=2,
        trainable=False,
        seed=51 + kmax,
    )

    n_batch = 2
    coeff_dim = dp_net.projector.coeff_dim // dp_net.n_frames
    rng = np.random.default_rng(8192 + kmax)
    q = rng.standard_normal(
        (n_batch, coeff_dim, n_focus, dp_net.context_channels)
    ).astype(np.float64)
    c = rng.standard_normal(
        (n_batch, coeff_dim, n_focus, dp_net.context_channels)
    ).astype(np.float64)

    quat = _random_quaternion(n_batch, seed=99 + kmax)
    d_matrix, _ = WignerDCalculator(lmax=lmax, precision="float64").call(quat)
    d_matrix = np.asarray(d_matrix)

    y_rotated_input = np.asarray(
        dp_net.call(_rotate_ndfc(q, d_matrix), _rotate_ndfc(c, d_matrix))
    )
    y_then_rotated = _rotate_ndfc(np.asarray(dp_net.call(q, c)), d_matrix)
    max_error = float(np.max(np.abs(y_rotated_input - y_then_rotated)))
    assert max_error <= 1e-10, f"equivariance error {max_error}"


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
def test_so3_cross_flat_parity(op_type) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import SO3GridNet as PTSO3GridNet

    lmax, channels, n_focus, kmax = 2, 2, 2, 1
    common = {
        "lmax": lmax,
        "kmax": kmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "cross",
        "op_type": op_type,
        "layout": "flat",
        "coefficient_layout": "packed",
        "grid_branches": 2,
        "trainable": False,
        "seed": 67,
    }
    pt_net = PTSO3GridNet(dtype=torch.float64, **common).to("cpu")
    dp_net = SO3GridNet(precision="float64", **common)
    _copy_so3gridnet(dp_net, pt_net)

    n_batch = 3
    coeff_dim = dp_net.projector.coeff_dim // dp_net.n_frames
    rng = np.random.default_rng(909)
    # flat layout: (N, D, F * context_channels)
    q = rng.standard_normal(
        (n_batch, coeff_dim, n_focus * dp_net.context_channels)
    ).astype(np.float64)
    c = rng.standard_normal(
        (n_batch, coeff_dim, n_focus * dp_net.context_channels)
    ).astype(np.float64)

    dp_out = np.asarray(dp_net.call(q, c))
    pt_out = pt_net(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus * dp_net.output_channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
def test_so3_serialize_roundtrip(mode) -> None:
    lmax, channels, n_focus, kmax = 2, 2, 1, 2
    dp_net = SO3GridNet(
        lmax=lmax,
        kmax=kmax,
        channels=channels,
        n_focus=n_focus,
        mode=mode,
        op_type="branch",
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_branches=2,
        residual_scale_init=0.5,
        trainable=False,
        seed=73,
    )
    data = dp_net.serialize()
    assert data["@version"] == 1
    if mode == "cross":
        assert "frame_expand.weight" in data["@variables"]
        assert "frame_contract.weight" in data["@variables"]
    dp_net2 = SO3GridNet.deserialize(data)

    n_batch = 3
    if mode == "self":
        coeff_dim = (lmax + 1) ** 2
        rng = np.random.default_rng(135)
        x = rng.standard_normal(
            (n_batch, coeff_dim, n_focus, dp_net.query_channels)
        ).astype(np.float64)
        args = (x,)
    else:
        coeff_dim = dp_net.projector.coeff_dim // dp_net.n_frames
        rng = np.random.default_rng(246)
        q = rng.standard_normal(
            (n_batch, coeff_dim, n_focus, dp_net.context_channels)
        ).astype(np.float64)
        c = rng.standard_normal(
            (n_batch, coeff_dim, n_focus, dp_net.context_channels)
        ).astype(np.float64)
        args = (q, c)

    assert_allclose(
        np.asarray(dp_net2.call(*args)),
        np.asarray(dp_net.call(*args)),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
def test_torch_namespace(mode) -> None:
    import torch

    lmax, channels, n_focus, kmax = 2, 2, 1, 2
    dp_net = SO3GridNet(
        lmax=lmax,
        kmax=kmax,
        channels=channels,
        n_focus=n_focus,
        mode=mode,
        op_type="mlp",
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        trainable=False,
        seed=23,
    )

    n_batch = 3
    if mode == "self":
        coeff_dim = (lmax + 1) ** 2
        rng = np.random.default_rng(246)
        x = rng.standard_normal(
            (n_batch, coeff_dim, n_focus, dp_net.query_channels)
        ).astype(np.float64)
        args = (x,)
    else:
        coeff_dim = dp_net.projector.coeff_dim // dp_net.n_frames
        rng = np.random.default_rng(357)
        q = rng.standard_normal(
            (n_batch, coeff_dim, n_focus, dp_net.context_channels)
        ).astype(np.float64)
        c = rng.standard_normal(
            (n_batch, coeff_dim, n_focus, dp_net.context_channels)
        ).astype(np.float64)
        args = (q, c)

    np_out = np.asarray(dp_net.call(*args))
    torch_args = tuple(torch.from_numpy(a) for a in args)
    torch_out = dp_net.call(*torch_args).detach().cpu().numpy()
    assert_allclose(torch_out, np_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
def test_s2_regression(mode) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    lmax, channels, n_focus = 2, 2, 1
    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": mode,
        "op_type": "branch",
        "layout": "ndfc",
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "grid_branches": 2,
        "trainable": False,
        "seed": 19,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common).to("cpu")
    dp_net = S2GridNet(precision="float64", **common)
    sd = pt_net.state_dict()

    def _np(key):
        return sd[key].detach().cpu().numpy().astype(np.float64)

    dp_net.scalar_gate.weight = _np("scalar_gate.weight")
    for name in ("left_proj", "right_proj", "router", "out_proj"):
        getattr(dp_net.grid_op, name).weight = _np(f"grid_op.{name}.weight")

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(135)
    if mode == "self":
        x = rng.standard_normal((n_batch, coeff_dim, n_focus, 2 * channels)).astype(
            np.float64
        )
        args = (x,)
    else:
        q = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(
            np.float64
        )
        c = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(
            np.float64
        )
        args = (q, c)

    dp_out = np.asarray(dp_net.call(*args))
    torch_args = tuple(torch.from_numpy(a) for a in args)
    pt_out = pt_net(*torch_args).detach().cpu().numpy()
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)
