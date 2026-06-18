# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the frame-aware DPA4 grid helper ``_project_frames``.

These mirror the current pt ``deepmd.pt.model.descriptor.sezm_nn.grid_net``
(refactored by PR #5552 to operate on coefficients). pt imports live inside
the test functions because ruff TID253 bans module-level ``deepmd.pt`` imports
under ``source/tests/common``.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import GridProduct as DPGridProduct
from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    _project_frames,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so3 import ChannelLinear as DPChannelLinear


@pytest.mark.parametrize("n_frames", [1, 2, 3])  # number of Wigner-D frames
def test_project_frames_parity(n_frames) -> None:
    """The dpmodel ``_project_frames`` matches pt with a weight-copied ChannelLinear."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        _project_frames as pt_project_frames,
    )
    from deepmd.pt.model.descriptor.sezm_nn.so3 import ChannelLinear as PTChannelLinear

    c_in, c_out = 4, 6
    # pin to CPU so torch.from_numpy fp64 inputs and the module agree under the
    # CUDA-default-device CI configuration
    pt_proj = PTChannelLinear(
        in_channels=c_in,
        out_channels=c_out,
        dtype=torch.float64,
        bias=False,
        trainable=True,
        seed=11,
    ).to("cpu")
    rng = np.random.default_rng(2026)
    with torch.no_grad():
        for p in pt_proj.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))
    state = {k: v.detach().cpu().numpy() for k, v in pt_proj.state_dict().items()}
    assert set(state) == {"weight"}

    dp_proj = DPChannelLinear(
        in_channels=c_in,
        out_channels=c_out,
        precision="float64",
        bias=False,
        trainable=True,
        seed=11,
    )
    dp_proj.weight = state["weight"]

    n_batch, coeff_dim, n_focus = 5, 9, 2
    coeff = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * c_in))

    dp_out = _project_frames(coeff, dp_proj, n_frames)
    pt_out = pt_project_frames(torch.from_numpy(coeff), pt_proj, n_frames)
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, n_frames * c_out)
    np.testing.assert_allclose(
        np.asarray(dp_out),
        pt_out.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_project_frames_torch_namespace() -> None:
    """``_project_frames`` on torch input matches the numpy-input result.

    Array-API pitfall guard: the helper must work with any array namespace.
    """
    import torch

    c_in, c_out, n_frames = 4, 5, 2
    dp_proj = DPChannelLinear(
        in_channels=c_in,
        out_channels=c_out,
        precision="float64",
        bias=False,
        trainable=True,
        seed=21,
    )
    rng = np.random.default_rng(99)
    coeff = rng.normal(size=(3, 9, 2, n_frames * c_in))

    np_out = _project_frames(coeff, dp_proj, n_frames)
    torch_out = _project_frames(torch.from_numpy(coeff), dp_proj, n_frames)
    np.testing.assert_allclose(
        np.asarray(np_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_grid_product_parity() -> None:
    """The dpmodel ``GridProduct`` matches pt over a real S2 projector's grid fns."""
    import torch

    from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import S2GridNet as DPS2GridNet
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import GridProduct as PTGridProduct
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2GridNet

    lmax, channels, n_focus = 2, 4, 1
    # op_type='glu' makes grid_op a GridProduct; we reuse the nets only for
    # their (parameter-free, deterministic) _to_grid/_from_grid projectors.
    pt_net = PTS2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="self",
        op_type="glu",
        dtype=torch.float64,
        layout="ndfc",
        grid_method="lebedev",
        trainable=True,
        seed=7,
    ).to("cpu")
    dp_net = DPS2GridNet(
        lmax=lmax,
        channels=channels,
        n_focus=n_focus,
        mode="self",
        op_type="glu",
        precision="float64",
        layout="ndfc",
        grid_method="lebedev",
        trainable=True,
        seed=7,
    )

    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(314)
    left = rng.normal(size=(5, coeff_dim, n_focus, channels))
    right = rng.normal(size=(5, coeff_dim, n_focus, channels))
    scalar = rng.normal(size=(5, n_focus, 2 * channels))

    dp_out = DPGridProduct().call(
        left,
        right,
        scalar,
        to_grid=dp_net._to_grid,
        from_grid=dp_net._from_grid,
    )
    pt_out = PTGridProduct()(
        torch.from_numpy(left),
        torch.from_numpy(right),
        torch.from_numpy(scalar),
        to_grid=pt_net._to_grid,
        from_grid=pt_net._from_grid,
    )
    assert dp_out.shape == (5, coeff_dim, n_focus, channels)
    np.testing.assert_allclose(
        np.asarray(dp_out),
        pt_out.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
