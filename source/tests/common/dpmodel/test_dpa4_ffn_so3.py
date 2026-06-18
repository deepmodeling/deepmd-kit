# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the DPA4 ``EquivariantFFN`` SO3-grid (``ffn_so3_grid``) path.

These tests cover the newly un-guarded ``ffn_so3_grid=True`` branch of the
dpmodel ``EquivariantFFN`` which wires ``SO3GridNet(mode='self')`` in place of
the ``S2GridNet`` used by the ``s2_activation`` path. They mirror the current pt
``EquivariantFFN`` in ``deepmd.pt.model.descriptor.sezm_nn.ffn``.

Note: ``EquivariantFFN`` exposes neither ``mmax`` nor ``n_focus`` — the SO3 grid
always uses ``mmax = lmax`` (projector default) and ``n_focus = 1`` internally
(matching pt). The truncated ``mmax < lmax`` path is exercised at the
``SO3GridNet`` level in ``test_dpa4_so3_gridnet.py``.

pt imports live inside the test functions (ruff TID253 bans module-level
``deepmd.pt`` imports under ``source/tests/common``); pt modules are pinned to
CPU (``.to("cpu")``) under the CUDA-default-device CI.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.ffn import EquivariantFFN as DPFFN


def _build_ffn_pair(*, lmax, channels, hidden_channels, ffn_config, seed=7):
    """Build a pt + dp ``EquivariantFFN`` sharing identical (perturbed) weights.

    Returns ``(pt_ffn, dp_ffn)``. The weight copy goes pt -> dp via
    ``DPFFN.deserialize(pt_ffn.serialize())`` (both share state_dict key names).
    Weights are perturbed first because ``so3_linear_2`` is zero-initialised,
    which would otherwise make the FFN output identically zero.
    """
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.ffn import EquivariantFFN as PTFFN

    pt_ffn = PTFFN(
        lmax=lmax,
        channels=channels,
        hidden_channels=hidden_channels,
        dtype=torch.float64,
        trainable=True,
        seed=seed,
        **ffn_config,
    ).to("cpu")
    rng = np.random.default_rng(2100)
    with torch.no_grad():
        for p in pt_ffn.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    dp_ffn = DPFFN.deserialize(pt_ffn.serialize())
    return pt_ffn, dp_ffn


def _run_pt(ffn, x):
    import torch

    return ffn(torch.from_numpy(x)).detach().cpu().numpy()


def _run_dp(ffn, x):
    return np.asarray(ffn.call(x))


# === SO3 grid parity ================================================


@pytest.mark.parametrize(
    "grid_mlp,grid_branch",  # grid op: (False,0)->glu, (True,0)->mlp, (False,1)->branch
    [(False, 0), (True, 0), (False, 1)],
)
def test_ffn_so3_grid_parity(grid_mlp, grid_branch) -> None:
    """ffn_so3_grid=True dp FFN matches pt at 1e-12 (lmax=3, kmax=1, mmax=lmax)."""
    lmax, channels, hidden_channels, kmax = 3, 8, 8, 1
    pt_ffn, dp_ffn = _build_ffn_pair(
        lmax=lmax,
        channels=channels,
        hidden_channels=hidden_channels,
        ffn_config={
            "kmax": kmax,
            "ffn_so3_grid": True,
            "grid_mlp": grid_mlp,
            "grid_branch": grid_branch,
        },
    )
    assert dp_ffn.ffn_so3_grid
    assert dp_ffn.grid_n_frames == 2 * kmax + 1
    rng = np.random.default_rng(11)
    # (N, D, F, C): D=(lmax+1)^2, F=n_focus=1, C=channels
    x = rng.normal(size=(5, (lmax + 1) ** 2, 1, channels))
    dp_out = _run_dp(dp_ffn, x)
    pt_out = _run_pt(pt_ffn, x)
    assert dp_out.shape == x.shape
    # output must be non-trivial (so3_linear_2 perturbed away from zero-init)
    assert np.max(np.abs(dp_out)) > 1e-6
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


def test_ffn_so3_grid_constructs() -> None:
    """The dp FFN with ffn_so3_grid=True constructs and runs (no NotImplementedError)."""
    from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import SO3GridNet as DPSO3GridNet

    lmax, channels, hidden_channels, kmax = 3, 8, 8, 1
    ffn = DPFFN(
        lmax=lmax,
        channels=channels,
        hidden_channels=hidden_channels,
        kmax=kmax,
        ffn_so3_grid=True,
        grid_mlp=False,
        grid_branch=0,
        precision="float64",
        trainable=True,
        seed=3,
    )
    assert isinstance(ffn.act, DPSO3GridNet)
    assert ffn.grid_n_frames == 2 * kmax + 1
    # linear1 output channels mirror pt: 2 * grid_n_frames * hidden_channels
    assert ffn.so3_linear_1.out_channels == 2 * ffn.grid_n_frames * hidden_channels
    assert ffn.so3_linear_2.in_channels == ffn.grid_n_frames * hidden_channels
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, (lmax + 1) ** 2, 1, channels))
    out = _run_dp(ffn, x)
    assert out.shape == x.shape


# === S2 regression (ffn_so3_grid=False, n_frames==1 path untouched) =


@pytest.mark.parametrize("grid_mlp", [False, True])  # glu vs mlp grid op
def test_ffn_s2_regression(grid_mlp) -> None:
    """The s2_activation FFN path still matches pt at 1e-12 (not broken)."""
    lmax, channels, hidden_channels = 2, 4, 4
    pt_ffn, dp_ffn = _build_ffn_pair(
        lmax=lmax,
        channels=channels,
        hidden_channels=hidden_channels,
        ffn_config={
            "s2_activation": True,
            "ffn_so3_grid": False,
            "grid_mlp": grid_mlp,
            "lebedev_quadrature": True,
        },
    )
    assert not dp_ffn.ffn_so3_grid
    assert dp_ffn.grid_n_frames == 1
    rng = np.random.default_rng(22)
    x = rng.normal(size=(5, (lmax + 1) ** 2, 1, channels))
    dp_out = _run_dp(dp_ffn, x)
    pt_out = _run_pt(pt_ffn, x)
    assert np.max(np.abs(dp_out)) > 1e-6
    np.testing.assert_allclose(dp_out, pt_out, rtol=1e-12, atol=1e-12)


# === serialize roundtrip ============================================


@pytest.mark.parametrize(
    "grid_mlp,grid_branch",  # grid op: (False,0)->glu, (True,0)->mlp, (False,1)->branch
    [(False, 0), (True, 0), (False, 1)],
)
def test_ffn_so3_serialize_roundtrip(grid_mlp, grid_branch) -> None:
    """ffn_so3_grid FFN serialize -> deserialize -> forward identical."""
    lmax, channels, hidden_channels, kmax = 3, 8, 8, 1
    _, dp_ffn = _build_ffn_pair(
        lmax=lmax,
        channels=channels,
        hidden_channels=hidden_channels,
        ffn_config={
            "kmax": kmax,
            "ffn_so3_grid": True,
            "grid_mlp": grid_mlp,
            "grid_branch": grid_branch,
        },
    )
    data = dp_ffn.serialize()
    assert data["@class"] == "EquivariantFFN"
    assert data["config"]["ffn_so3_grid"] is True
    restored = DPFFN.deserialize(data)
    rng = np.random.default_rng(33)
    x = rng.normal(size=(5, (lmax + 1) ** 2, 1, channels))
    np.testing.assert_allclose(
        _run_dp(restored, x), _run_dp(dp_ffn, x), rtol=1e-12, atol=1e-12
    )
