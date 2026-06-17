# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity / equivariance tests for the DPA4 ``BaseGridNet`` cross-mode path.

Covers the dpmodel port of ``mode='cross'`` (with ``layout='flat'`` and
``residual_scale_init``) using the S2 projector (``n_frames == 1``).  All pt
imports are kept inside the test functions (ruff TID253).

Test menu:

* ``test_s2_cross_parity`` -- weight-copied fp64 forward parity (glu/mlp/branch).
* ``test_s2_cross_equivariance`` -- rotate query & context, SO(3) equivariance.
* ``test_layout_flat_parity`` -- ``layout='flat'`` parity.
* ``test_residual_scale_parity`` -- ``residual_scale_init`` parity + serialize.
* ``test_self_mode_regression`` -- the existing self-mode path still matches pt.
* ``test_torch_namespace`` -- cross-mode ``.call`` on torch inputs matches numpy.
"""

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
)

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    S2GridNet,
)


def _rotate_ndfc(x: np.ndarray, d_matrix: np.ndarray) -> np.ndarray:
    """Rotate coefficient-layout tensors with shape ``(N, D, F, C)``."""
    return np.einsum("nij,njfc->nifc", d_matrix, x)


def _random_quaternion(n_batch: int, seed: int) -> np.ndarray:
    """Sample normalized quaternions in ``(w, x, y, z)`` order."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_batch, 4)).astype(np.float64)
    return q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True))


def _copy_s2gridnet(dp_net: S2GridNet, pt_net) -> None:
    """Copy a pt ``S2GridNet`` scalar gate + grid op (+ residual) into dpmodel."""
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
    if dp_net.residual_scale is not None:
        dp_net.residual_scale = _np("residual_scale").reshape(
            dp_net.residual_scale.shape
        )


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
@pytest.mark.parametrize("lmax,channels", [(2, 2), (3, 2)])  # lmax, channels
def test_s2_cross_parity(op_type, lmax, channels) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    n_focus = 1
    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "cross",
        "op_type": op_type,
        "layout": "ndfc",
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "grid_branches": 2,
        "trainable": False,
        "seed": 17 + lmax,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common)
    dp_net = S2GridNet(precision="float64", **common)
    _copy_s2gridnet(dp_net, pt_net)

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(2024 + lmax)
    q = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)

    dp_out = np.asarray(dp_net.call(q, c))
    pt_out = pt_net(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
@pytest.mark.parametrize("lmax,channels", [(2, 2), (3, 2)])  # lmax, channels
def test_s2_cross_equivariance(op_type, lmax, channels) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    n_focus = 1
    dp_net = S2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="cross",
        op_type=op_type,
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_method="lebedev",
        grid_branches=2,
        trainable=False,
        seed=31 + lmax,
    )

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(4096 + lmax)
    q = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)

    quat = _random_quaternion(n_batch, seed=77 + lmax)
    d_matrix, _ = WignerDCalculator(lmax=lmax, precision="float64").call(quat)
    d_matrix = np.asarray(d_matrix)

    y_rotated_input = np.asarray(
        dp_net.call(_rotate_ndfc(q, d_matrix), _rotate_ndfc(c, d_matrix))
    )
    y_then_rotated = _rotate_ndfc(np.asarray(dp_net.call(q, c)), d_matrix)
    max_error = float(np.max(np.abs(y_rotated_input - y_then_rotated)))
    assert max_error <= 1e-10, f"equivariance error {max_error}"


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
def test_layout_flat_parity(op_type) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    lmax, channels, n_focus = 2, 2, 2
    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "cross",
        "op_type": op_type,
        "layout": "flat",
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "grid_branches": 2,
        "trainable": False,
        "seed": 53,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common)
    dp_net = S2GridNet(precision="float64", **common)
    _copy_s2gridnet(dp_net, pt_net)

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(909)
    # flat layout: (N, D, F * C)
    q = rng.standard_normal((n_batch, coeff_dim, n_focus * channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, coeff_dim, n_focus * channels)).astype(np.float64)

    dp_out = np.asarray(dp_net.call(q, c))
    pt_out = pt_net(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus * channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "residual_scale_init", [None, 0.5]
)  # residual-scale initial value
def test_residual_scale_parity(residual_scale_init) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    lmax, channels, n_focus = 2, 2, 1
    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "cross",
        "op_type": "glu",
        "layout": "ndfc",
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "residual_scale_init": residual_scale_init,
        "trainable": False,
        "seed": 71,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common)
    dp_net = S2GridNet(precision="float64", **common)
    _copy_s2gridnet(dp_net, pt_net)

    if residual_scale_init is None:
        assert dp_net.residual_scale is None
    else:
        assert dp_net.residual_scale is not None
        assert dp_net.residual_scale.shape == (n_focus, channels)

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(606)
    q = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)

    dp_out = np.asarray(dp_net.call(q, c))
    pt_out = pt_net(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)

    # residual_scale must survive serialize/deserialize.
    data = dp_net.serialize()
    if residual_scale_init is None:
        assert "residual_scale" not in data["@variables"]
    else:
        assert "residual_scale" in data["@variables"]
    dp_net2 = S2GridNet.deserialize(data)
    if residual_scale_init is None:
        assert dp_net2.residual_scale is None
    else:
        assert dp_net2.residual_scale is not None
        assert_allclose(
            np.asarray(dp_net2.residual_scale),
            np.asarray(dp_net.residual_scale),
            atol=1e-12,
            rtol=1e-12,
        )
    assert_allclose(
        np.asarray(dp_net2.call(q, c)),
        dp_out,
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
def test_self_mode_regression(op_type) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    lmax, channels, n_focus = 2, 2, 1
    common = {
        "lmax": lmax,
        "channels": channels,
        "n_focus": n_focus,
        "mode": "self",
        "op_type": op_type,
        "layout": "ndfc",
        "coefficient_layout": "packed",
        "grid_method": "lebedev",
        "grid_branches": 2,
        "trainable": False,
        "seed": 19,
    }
    pt_net = PTS2GridNet(dtype=torch.float64, **common)
    dp_net = S2GridNet(precision="float64", **common)
    _copy_s2gridnet(dp_net, pt_net)

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(135)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, 2 * channels)).astype(
        np.float64
    )

    dp_out = np.asarray(dp_net.call(x))
    pt_out = pt_net(torch.from_numpy(x)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid op
def test_torch_namespace(op_type) -> None:
    import torch

    lmax, channels, n_focus = 2, 2, 1
    dp_net = S2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="cross",
        op_type=op_type,
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_method="lebedev",
        grid_branches=2,
        residual_scale_init=0.5,
        trainable=False,
        seed=23,
    )

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(246)
    q = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)

    np_out = np.asarray(dp_net.call(q, c))
    torch_out = (
        dp_net.call(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    )
    assert_allclose(torch_out, np_out, atol=1e-12, rtol=1e-12)
