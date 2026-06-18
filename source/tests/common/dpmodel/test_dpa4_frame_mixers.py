# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the DPA4 per-degree frame channel mixers.

Compares the dpmodel ports of ``FrameContract``, ``FrameExpand`` and
``_build_frame_degree_index`` against the reference pt implementations using
weight-copied fp64 parity.  All pt imports are kept inside the test functions
(ruff TID253).
"""

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
)

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    FrameContract,
    FrameExpand,
    _build_frame_degree_index,
)

# (lmax, channels, kmax) combinations; n_frames = 2 * kmax + 1
PARAMS = [
    (2, 4, 1),
    (3, 2, 2),
]


def _copy_weight(dp_obj, pt_obj) -> None:
    """Copy the pt weight tensor into the dpmodel numpy attribute."""
    dp_obj.weight = pt_obj.weight.detach().cpu().numpy().astype(np.float64)


@pytest.mark.parametrize("lmax,channels,kmax", PARAMS)  # lmax, channels, kmax
def test_frame_contract_parity(lmax, channels, kmax) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        FrameContract as PTFrameContract,
    )

    n_frames = 2 * kmax + 1
    pt_obj = PTFrameContract(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        dtype=torch.float64,
        trainable=True,
        seed=0,
    ).to("cpu")
    dp_obj = FrameContract(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=0,
    )
    _copy_weight(dp_obj, pt_obj)

    n_batch, n_focus = 3, 2
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(123)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, n_frames * channels)).astype(
        np.float64
    )

    dp_out = dp_obj.call(x)
    pt_out = pt_obj(torch.from_numpy(x)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, channels)
    assert_allclose(np.asarray(dp_out), pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("lmax,channels,kmax", PARAMS)  # lmax, channels, kmax
def test_frame_expand_parity(lmax, channels, kmax) -> None:
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import FrameExpand as PTFrameExpand

    n_frames = 2 * kmax + 1
    pt_obj = PTFrameExpand(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        dtype=torch.float64,
        trainable=True,
        seed=0,
    ).to("cpu")
    dp_obj = FrameExpand(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=0,
    )
    _copy_weight(dp_obj, pt_obj)

    n_batch, n_focus = 3, 2
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(321)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)

    dp_out = dp_obj.call(x)
    pt_out = pt_obj(torch.from_numpy(x)).detach().cpu().numpy()
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, n_frames * channels)
    assert_allclose(np.asarray(dp_out), pt_out, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("lmax,channels,kmax", PARAMS)  # lmax, channels, kmax
def test_expand_then_contract_shapes(lmax, channels, kmax) -> None:
    n_frames = 2 * kmax + 1
    expand = FrameExpand(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=0,
    )
    contract = FrameContract(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=1,
    )
    n_batch, n_focus = 3, 2
    coeff_dim = (lmax + 1) ** 2
    rng = np.random.default_rng(7)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, channels)).astype(np.float64)
    expanded = expand.call(x)
    assert expanded.shape == (n_batch, coeff_dim, n_focus, n_frames * channels)
    contracted = contract.call(expanded)
    assert contracted.shape == (n_batch, coeff_dim, n_focus, channels)


@pytest.mark.parametrize("cls", [FrameContract, FrameExpand])  # mixer class
def test_serialize_roundtrip(cls) -> None:
    lmax, channels, kmax = 2, 4, 1
    n_frames = 2 * kmax + 1
    obj = cls(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=5,
    )
    data = obj.serialize()
    assert data["@version"] == 1
    obj2 = cls.deserialize(data)
    assert_allclose(obj2.weight, obj.weight, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(obj2.degree_index, obj.degree_index)

    n_batch, n_focus = 2, 2
    coeff_dim = (lmax + 1) ** 2
    in_ch = channels if cls is FrameExpand else n_frames * channels
    rng = np.random.default_rng(11)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, in_ch)).astype(np.float64)
    assert_allclose(
        np.asarray(obj2.call(x)),
        np.asarray(obj.call(x)),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize("lmax", [1, 2, 3])  # max angular momentum
def test_degree_index(lmax) -> None:
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        _build_frame_degree_index as pt_build_frame_degree_index,
    )

    dp_idx = _build_frame_degree_index(
        lmax=lmax, mmax=lmax, coefficient_layout="packed"
    )
    pt_idx = (
        pt_build_frame_degree_index(lmax=lmax, mmax=lmax, coefficient_layout="packed")
        .detach()
        .cpu()
        .numpy()
    )
    assert dp_idx.shape == ((lmax + 1) ** 2,)
    np.testing.assert_array_equal(dp_idx, pt_idx)
    # each (l, m) row maps to degree l: row d has degree dp_idx[d]
    expected = np.concatenate(
        [np.full(2 * l + 1, l, dtype=np.int64) for l in range(lmax + 1)]
    )
    np.testing.assert_array_equal(dp_idx, expected)


@pytest.mark.parametrize("cls", [FrameContract, FrameExpand])  # mixer class
def test_torch_namespace_smoke(cls) -> None:
    import torch

    lmax, channels, kmax = 2, 4, 1
    n_frames = 2 * kmax + 1
    obj = cls(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=9,
    )
    n_batch, n_focus = 2, 2
    coeff_dim = (lmax + 1) ** 2
    in_ch = channels if cls is FrameExpand else n_frames * channels
    rng = np.random.default_rng(13)
    x = rng.standard_normal((n_batch, coeff_dim, n_focus, in_ch)).astype(np.float64)
    np_out = np.asarray(obj.call(x))
    torch_out = obj.call(torch.from_numpy(x)).detach().cpu().numpy()
    assert_allclose(torch_out, np_out, atol=1e-12, rtol=1e-12)
