# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the DPA4 ``GridMLP`` grid op and the ``op_type='mlp'`` path.

Compares the dpmodel port of ``GridMLP`` (and a full ``S2GridNet`` with
``op_type='mlp'``) against the reference pt implementations using weight-copied
fp64 parity, plus an SO(3) equivariance check for the mlp S2 grid net.  All pt
imports are kept inside the test functions (ruff TID253).
"""

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
)

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    GridMLP,
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


def _copy_grid_mlp(dp_obj: GridMLP, pt_obj) -> None:
    """Copy pt ``GridMLP`` projection weights into the dpmodel attributes."""
    sd = pt_obj.state_dict()
    for name in ("left_proj", "right_proj", "out_proj"):
        getattr(dp_obj, name).weight = (
            sd[f"{name}.weight"].detach().cpu().numpy().astype(np.float64)
        )


def _copy_s2gridnet_mlp(dp_net: S2GridNet, pt_net) -> None:
    """Copy a pt mlp ``S2GridNet`` scalar gate + grid op weights into dpmodel."""
    sd = pt_net.state_dict()
    dp_net.scalar_gate.weight = (
        sd["scalar_gate.weight"].detach().cpu().numpy().astype(np.float64)
    )
    for name in ("left_proj", "right_proj", "out_proj"):
        getattr(dp_net.grid_op, name).weight = (
            sd[f"grid_op.{name}.weight"].detach().cpu().numpy().astype(np.float64)
        )


@pytest.mark.parametrize("channels", [2, 4])  # channels per grid point
def test_grid_mlp_parity_self(channels) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        GridMLP as PTGridMLP,
    )

    pt_obj = PTGridMLP(
        channels=channels,
        mode="self",
        dtype=torch.float64,
        trainable=True,
        seed=0,
    )
    dp_obj = GridMLP(
        channels=channels,
        mode="self",
        precision="float64",
        trainable=True,
        seed=0,
    )
    _copy_grid_mlp(dp_obj, pt_obj)

    n_batch, n_grid, n_focus = 3, 5, 2
    rng = np.random.default_rng(123)
    q = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)

    dp_out = dp_obj.call(q, c)
    pt_out = pt_obj(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, n_grid, n_focus, channels)
    assert_allclose(np.asarray(dp_out), pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("channels", [2, 4])  # channels per grid point
def test_grid_mlp_parity_cross(channels) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        GridMLP as PTGridMLP,
    )

    pt_obj = PTGridMLP(
        channels=channels,
        mode="cross",
        dtype=torch.float64,
        trainable=True,
        seed=1,
    )
    dp_obj = GridMLP(
        channels=channels,
        mode="cross",
        precision="float64",
        trainable=True,
        seed=1,
    )
    _copy_grid_mlp(dp_obj, pt_obj)

    n_batch, n_grid, n_focus = 3, 5, 2
    rng = np.random.default_rng(321)
    q = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)

    dp_out = dp_obj.call(q, c)
    pt_out = pt_obj(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, n_grid, n_focus, channels)
    assert_allclose(np.asarray(dp_out), pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
def test_grid_mlp_serialize_roundtrip(mode) -> None:
    channels = 4
    obj = GridMLP(
        channels=channels,
        mode=mode,
        precision="float64",
        trainable=True,
        seed=5,
    )
    data = obj.serialize()
    assert data["@version"] == 1
    obj2 = GridMLP.deserialize(data)

    n_batch, n_grid, n_focus = 2, 4, 2
    rng = np.random.default_rng(11)
    q = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)
    assert_allclose(
        np.asarray(obj2.call(q, c)),
        np.asarray(obj.call(q, c)),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("mode", ["self", "cross"])  # pairing mode
def test_grid_mlp_torch_namespace(mode) -> None:
    import torch

    channels = 4
    obj = GridMLP(
        channels=channels,
        mode=mode,
        precision="float64",
        trainable=True,
        seed=9,
    )
    n_batch, n_grid, n_focus = 2, 4, 2
    rng = np.random.default_rng(13)
    q = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)
    c = rng.standard_normal((n_batch, n_grid, n_focus, channels)).astype(np.float64)
    np_out = np.asarray(obj.call(q, c))
    torch_out = (
        obj.call(torch.from_numpy(q), torch.from_numpy(c)).detach().cpu().numpy()
    )
    assert_allclose(torch_out, np_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("lmax,channels", [(2, 2), (3, 2)])  # lmax, channels
def test_s2gridnet_op_type_mlp_parity(lmax, channels) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        S2GridNet as PTS2GridNet,
    )

    n_focus = 1
    pt_net = PTS2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="self",
        op_type="mlp",
        dtype=torch.float64,
        layout="ndfc",
        coefficient_layout="packed",
        grid_method="lebedev",
        mlp_bias=False,
        trainable=False,
        seed=17 + lmax,
    )
    dp_net = S2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="self",
        op_type="mlp",
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_method="lebedev",
        mlp_bias=False,
        trainable=False,
        seed=17 + lmax,
    )
    _copy_s2gridnet_mlp(dp_net, pt_net)

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(2024 + lmax)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, 2 * channels)).astype(
        np.float64
    )

    dp_out = np.asarray(dp_net.call(x))
    pt_out = pt_net(torch.from_numpy(x)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, channels)
    assert_allclose(dp_out, pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("lmax,channels", [(2, 2), (3, 2)])  # lmax, channels
def test_s2gridnet_op_type_mlp_equivariance(lmax, channels) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
    )

    n_focus = 1
    dp_net = S2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="self",
        op_type="mlp",
        precision="float64",
        layout="ndfc",
        coefficient_layout="packed",
        grid_method="lebedev",
        mlp_bias=False,
        trainable=False,
        seed=31 + lmax,
    )

    n_batch = 3
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(4096 + lmax)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, 2 * channels)).astype(
        np.float64
    )

    quat = _random_quaternion(n_batch, seed=77 + lmax)
    d_matrix, _ = WignerDCalculator(lmax=lmax, precision="float64").call(quat)
    d_matrix = np.asarray(d_matrix)

    y_rotated_input = np.asarray(dp_net.call(_rotate_ndfc(x, d_matrix)))
    y_then_rotated = _rotate_ndfc(np.asarray(dp_net.call(x)), d_matrix)
    max_error = float(np.max(np.abs(y_rotated_input - y_then_rotated)))
    assert max_error <= 1e-10, f"equivariance error {max_error}"
