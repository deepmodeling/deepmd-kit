# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the DPA4 ``SO2Convolution`` cross-mode grid products.

These tests cover the ``node_wise_s2``/``node_wise_so3`` (edge-local) and
``message_node_s2``/``message_node_so3`` (post-aggregation) grid-product
branches wired into the dpmodel ``SO2Convolution``. Each test builds a pt and a
dpmodel ``SO2Convolution`` with the same config, copies the (perturbed) pt
weights into the dpmodel module via ``DP.deserialize(pt.serialize())`` (which
also exercises the serialize round-trip), runs both on the same random padded /
sparse edge data, and asserts forward parity at fp64 (~1e-12 on CPU).

pt imports live inside the test functions because ruff TID253 bans module-level
``deepmd.pt`` imports under ``source/tests/common``; pt modules are pinned to
CPU (``.to("cpu")``) under the CUDA-default-device CI.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Convolution as DPSO2Conv

# fp64 weight-copied parity is near-bit on CPU.
RTOL, ATOL = 1e-12, 1e-14

NLOC = 5
NNEI = 4


def _to_pt(x):
    import torch

    return torch.from_numpy(np.ascontiguousarray(x)).to("cpu")


def _assert_parity(a, t, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(a), t.detach().cpu().numpy(), rtol=rtol, atol=atol
    )


def _base_kwargs(**overrides):
    kwargs = {
        "lmax": 3,
        "mmax": 1,
        "kmax": 1,
        "channels": 4,
        "n_focus": 1,
        "focus_dim": 0,
        "focus_compete": True,
        "so2_norm": False,
        "mixing_layers": 2,
        "so2_attn_res": "none",
        "layer_scale": False,
        "n_atten_head": 1,
        "radial_so2_mode": "degree_channel",
        "radial_so2_rank": 1,
        "lebedev_quadrature": True,
        "activation_function": "silu",
        "mlp_bias": False,
        "eps": 1e-7,
    }
    kwargs.update(overrides)
    return kwargs


def _perturb(pt_mod, seed):
    import torch

    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for p in pt_mod.parameters():
            p += _to_pt(0.1 * rng.normal(size=tuple(p.shape))).to(p.dtype)


def _build_conv_pair(seed=17, perturb_seed=2060, dtype=None, **overrides):
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution as PTSO2Conv

    if dtype is None:
        dtype = torch.float64
    kwargs = _base_kwargs(**overrides)
    pt_mod = PTSO2Conv(**kwargs, dtype=dtype, seed=seed, trainable=True).to("cpu")
    # post_focus_mix is zero-initialized; perturb so the output is nonzero and
    # the (residual-scaled) grid-product contribution is observable.
    _perturb(pt_mod, perturb_seed)
    dp_mod = DPSO2Conv.deserialize(pt_mod.serialize())
    return pt_mod, dp_mod, kwargs


def _build_edge_data(
    rng, *, nloc, nnei, lmax, channels, masked="slots", np_dtype=np.float64
):
    """Build matching pt (sparse) and dp (padded) edge caches on CPU."""
    from deepmd.dpmodel.descriptor.dpa4_nn.edge_cache import (
        EdgeCache,
    )
    from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
        WignerDCalculator,
        build_edge_quaternion,
    )
    from deepmd.pt.model.descriptor.sezm_nn.edge_cache import (
        EdgeFeatureCache,
    )

    n_edge = nloc * nnei
    dim_full = (lmax + 1) ** 2
    src = np.array(
        [(i + 1 + k) % nloc for i in range(nloc) for k in range(nnei)],
        dtype=np.int64,
    )
    dst = np.repeat(np.arange(nloc, dtype=np.int64), nnei)
    mask = np.ones(n_edge, dtype=np.float64)
    if masked == "slots":
        mask[3] = 0.0
        mask[nnei + 1] = 0.0
        mask[-1] = 0.0
    elif masked != "none":
        raise ValueError(f"unknown masked mode {masked}")
    valid = mask > 0.5

    edge_vec = rng.normal(size=(n_edge, 3))
    edge_vec /= np.linalg.norm(edge_vec, axis=-1, keepdims=True)
    quat = build_edge_quaternion(edge_vec)
    D_full, Dt_full = WignerDCalculator(lmax, precision="float64").call(quat)
    D_full = np.asarray(D_full)
    Dt_full = np.asarray(Dt_full)
    edge_rbf = np.zeros((n_edge, 1))
    edge_env = rng.uniform(0.2, 1.0, size=(n_edge, 1))
    deg = ((edge_env[:, 0] ** 2) * mask).reshape(nloc, nnei).sum(axis=1)
    inv_sqrt_deg = (1.0 / np.sqrt(deg + 1.0)).reshape(nloc, 1, 1)
    radial = rng.normal(size=(n_edge, lmax + 1, channels))
    x = rng.normal(size=(nloc, dim_full, channels))

    # Cast all float caches/inputs to the requested precision; both pt and dp
    # then consume bit-identical inputs (the only divergence is the in-module
    # accumulation order/precision).
    edge_vec = edge_vec.astype(np_dtype)
    edge_rbf = edge_rbf.astype(np_dtype)
    edge_env = edge_env.astype(np_dtype)
    deg = deg.astype(np_dtype)
    inv_sqrt_deg = inv_sqrt_deg.astype(np_dtype)
    D_full = D_full.astype(np_dtype)
    Dt_full = Dt_full.astype(np_dtype)
    radial = radial.astype(np_dtype)
    x = x.astype(np_dtype)
    edge_type_feat_np = np.zeros((n_edge, channels), dtype=np_dtype)

    t = _to_pt
    pt_cache = EdgeFeatureCache(
        src=t(src[valid]),
        dst=t(dst[valid]),
        edge_type_feat=t(edge_type_feat_np[valid]),
        edge_vec=t(edge_vec[valid]),
        edge_rbf=t(edge_rbf[valid]),
        edge_env=t(edge_env[valid]),
        deg=t(deg),
        inv_sqrt_deg=t(inv_sqrt_deg),
        D_full=t(D_full[valid]),
        Dt_full=t(Dt_full[valid]),
        edge_src_gate=None,
    )
    dp_cache = EdgeCache(
        src=src,
        dst=dst,
        edge_type_feat=edge_type_feat_np,
        edge_vec=edge_vec,
        edge_rbf=edge_rbf,
        edge_env=edge_env,
        deg=deg,
        inv_sqrt_deg=inv_sqrt_deg,
        D_full=D_full,
        Dt_full=Dt_full,
        edge_src_gate=None,
        edge_mask=mask,
    )
    return pt_cache, dp_cache, radial, radial[valid], x


def _assert_conv_parity(
    pt_mod, dp_mod, kwargs, *, masked="slots", np_dtype=np.float64, rtol=RTOL, atol=ATOL
):
    rng = np.random.default_rng(2061)
    pt_cache, dp_cache, radial, radial_valid, x = _build_edge_data(
        rng,
        nloc=NLOC,
        nnei=NNEI,
        lmax=kwargs["lmax"],
        channels=kwargs["channels"],
        masked=masked,
        np_dtype=np_dtype,
    )
    out_dp = dp_mod.call(x, dp_cache, radial)
    out_pt = pt_mod(_to_pt(x), pt_cache, _to_pt(radial_valid))
    _assert_parity(out_dp, out_pt, rtol=rtol, atol=atol)


@pytest.mark.parametrize("masked", ["none", "slots"])  # padded-slot pattern
def test_so2_node_wise_s2_parity(masked) -> None:
    # edge-local S2 cross product between source and destination node features
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        node_wise_s2=True, lebedev_quadrature=True
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs, masked=masked)


@pytest.mark.parametrize("masked", ["none", "slots"])  # padded-slot pattern
@pytest.mark.parametrize("lmax,mmax", [(2, 2), (3, 1)])  # full + truncated SO3
def test_so2_node_wise_so3_parity(masked, lmax, mmax) -> None:
    # edge-local SO(3) Wigner-D cross product; (3, 1) is the example truncation
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        node_wise_so3=True, lmax=lmax, mmax=mmax, lebedev_quadrature=False
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs, masked=masked)


@pytest.mark.parametrize("masked", ["none", "slots"])  # padded-slot pattern
def test_so2_message_node_s2_parity(masked) -> None:
    # post-aggregation packed-layout S2 cross product (message vs node)
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        message_node_s2=True, lebedev_quadrature=True
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs, masked=masked)


@pytest.mark.parametrize("masked", ["none", "slots"])  # padded-slot pattern
@pytest.mark.parametrize("lmax,mmax", [(2, 2), (3, 1)])  # full + truncated SO3
def test_so2_message_node_so3_parity(masked, lmax, mmax) -> None:
    # post-aggregation SO(3) cross product; (3, 1) mirrors the flagship example
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        message_node_so3=True, lmax=lmax, mmax=mmax, lebedev_quadrature=False
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs, masked=masked)


def test_so2_both_so3_parity() -> None:
    # node_wise_so3 + message_node_so3 together, example-config-like
    # (lmax=3, mmax=1, n_focus=2, mixing_layers=3, degree_channel radial).
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        node_wise_so3=True,
        message_node_so3=True,
        lmax=3,
        mmax=1,
        n_focus=2,
        mixing_layers=3,
        lebedev_quadrature=False,
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs)


def test_so2_grid_mlp_cross() -> None:
    # op_type='mlp' grid product (message_node_grid_mlp=True selects the MLP op)
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        message_node_so3=True,
        message_node_grid_mlp=True,
        lmax=3,
        mmax=1,
        lebedev_quadrature=False,
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs)


def test_so2_grid_branch_cross() -> None:
    # op_type='branch' grid product (mirrors the example's grid_branch=[1,...]).
    pt_mod, dp_mod, kwargs = _build_conv_pair(
        message_node_so3=True,
        message_node_grid_branch=2,
        lmax=3,
        mmax=1,
        lebedev_quadrature=False,
    )
    _assert_conv_parity(pt_mod, dp_mod, kwargs)


def test_so2_message_node_so3_fp32_parity() -> None:
    # fp32 parity for the example-config-shaped (lmax=3, mmax=1) message_node_so3
    # cross product. The flagship examples/water/dpa4/input.json runs
    # precision: float32; the grid path reduces over many Lebedev points, so the
    # right budget is the "computation-in-fp32" one (~1e-4), not fp64 bit-parity.
    import torch

    pt_mod, dp_mod, kwargs = _build_conv_pair(
        message_node_so3=True,
        lmax=3,
        mmax=1,
        lebedev_quadrature=False,
        dtype=torch.float32,
    )
    _assert_conv_parity(
        pt_mod, dp_mod, kwargs, np_dtype=np.float32, rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("masked", ["none", "slots"])  # padded-slot pattern
def test_so2_no_grid_regression(masked) -> None:
    # all grid flags False: the base SO2Convolution path must still match pt.
    pt_mod, dp_mod, kwargs = _build_conv_pair()
    assert dp_mod.node_wise_grid_product is None
    assert dp_mod.message_node_grid_product is None
    _assert_conv_parity(pt_mod, dp_mod, kwargs, masked=masked)
