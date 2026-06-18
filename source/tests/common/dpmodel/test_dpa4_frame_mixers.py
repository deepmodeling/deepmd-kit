# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the DPA4 SO3-grid per-degree frame mixers.

These mirror the current pt
``deepmd.pt.model.descriptor.sezm_nn.grid_net`` ``FrameContract`` /
``FrameExpand`` (and the ``_build_frame_degree_index`` helper). The pt mixers
realise a per-degree ``einsum("ndfi,dio->ndfo", coeff, weight[degree_index])``;
the dpmodel port realises the same map as a broadcast batched ``xp.matmul``.

pt imports live inside the test functions because ruff TID253 bans
module-level ``deepmd.pt`` imports under ``source/tests/common``. pt modules
are pinned to CPU so ``torch.from_numpy`` fp64 inputs and the module agree
under the CUDA-default-device CI configuration.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    FrameContract as DPFrameContract,
    FrameExpand as DPFrameExpand,
    _build_frame_degree_index,
)

# (lmax, channels, kmax); n_frames K = 2 * kmax + 1
_CASES = [(2, 4, 1), (3, 2, 2)]


def _copy_weight(pt_mod, dp_mod) -> None:
    """Copy the pt mixer ``weight`` state-dict entry into the dpmodel mixer."""
    state = {k: v.detach().cpu().numpy() for k, v in pt_mod.state_dict().items()}
    assert set(state) == {"weight"}, state.keys()
    dp_mod.weight = state["weight"]


@pytest.mark.parametrize("lmax,channels,kmax", _CASES)  # degree, channels, kmax
def test_frame_contract_parity(lmax, channels, kmax) -> None:
    """The dpmodel ``FrameContract`` matches pt with weight-copied fp64 weights."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        FrameContract as PTFrameContract,
    )

    n_frames = 2 * kmax + 1
    coeff_dim = (lmax + 1) ** 2
    n_batch, n_focus = 5, 2
    rng = np.random.default_rng(2026)

    pt_mod = PTFrameContract(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        dtype=torch.float64,
        trainable=True,
        seed=7,
    ).to("cpu")
    dp_mod = DPFrameContract(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=7,
    )
    _copy_weight(pt_mod, dp_mod)

    coeff = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))
    dp_out = dp_mod.call(coeff)
    pt_out = pt_mod(torch.from_numpy(coeff))
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, channels)
    np.testing.assert_allclose(
        np.asarray(dp_out), pt_out.detach().cpu().numpy(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("lmax,channels,kmax", _CASES)  # degree, channels, kmax
def test_frame_expand_parity(lmax, channels, kmax) -> None:
    """The dpmodel ``FrameExpand`` matches pt with weight-copied fp64 weights."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        FrameExpand as PTFrameExpand,
    )

    n_frames = 2 * kmax + 1
    coeff_dim = (lmax + 1) ** 2
    n_batch, n_focus = 5, 2
    rng = np.random.default_rng(2027)

    pt_mod = PTFrameExpand(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        dtype=torch.float64,
        trainable=True,
        seed=11,
    ).to("cpu")
    dp_mod = DPFrameExpand(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=11,
    )
    _copy_weight(pt_mod, dp_mod)

    coeff = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
    dp_out = dp_mod.call(coeff)
    pt_out = pt_mod(torch.from_numpy(coeff))
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, n_frames * channels)
    np.testing.assert_allclose(
        np.asarray(dp_out), pt_out.detach().cpu().numpy(), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("lmax,channels,kmax", _CASES)  # degree, channels, kmax
def test_expand_then_contract_shapes(lmax, channels, kmax) -> None:
    """Shape round-trip ``(N,D,F,C) -> expand -> (N,D,F,K*C) -> contract -> (N,D,F,C)``."""
    n_frames = 2 * kmax + 1
    coeff_dim = (lmax + 1) ** 2
    n_batch, n_focus = 3, 2
    rng = np.random.default_rng(404)

    expand = DPFrameExpand(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=1,
    )
    contract = DPFrameContract(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=2,
    )
    coeff = rng.normal(size=(n_batch, coeff_dim, n_focus, channels))
    expanded = expand.call(coeff)
    assert expanded.shape == (n_batch, coeff_dim, n_focus, n_frames * channels)
    contracted = contract.call(expanded)
    assert contracted.shape == (n_batch, coeff_dim, n_focus, channels)


@pytest.mark.parametrize("cls", [DPFrameContract, DPFrameExpand])  # mixer class
def test_serialize_roundtrip(cls) -> None:
    """Serialize -> deserialize -> forward is identical; @version == 1."""
    lmax, channels, n_frames = 2, 4, 3
    coeff_dim = (lmax + 1) ** 2
    n_batch, n_focus = 3, 2
    rng = np.random.default_rng(505)

    mod = cls(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=5,
    )
    # perturb to non-default weights
    mod.weight = mod.weight + 0.1 * rng.normal(size=mod.weight.shape)

    data = mod.serialize()
    assert data["@version"] == 1
    assert data["config"]["n_frames"] == n_frames
    assert set(data["@variables"]) == {"weight"}
    restored = cls.deserialize(data)
    np.testing.assert_array_equal(restored.weight, mod.weight)

    in_ch = mod.weight.shape[1]
    coeff = rng.normal(size=(n_batch, coeff_dim, n_focus, in_ch))
    out0 = mod.call(coeff)
    out1 = restored.call(coeff)
    np.testing.assert_allclose(
        np.asarray(out0), np.asarray(out1), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("lmax,mmax", [(2, 2), (3, 3), (3, 1)])  # degree, order
@pytest.mark.parametrize("layout", ["packed", "m_major"])  # coefficient layout
def test_degree_index(lmax, mmax, layout) -> None:
    """``_build_frame_degree_index`` maps each (l, m) row to its degree l.

    Compared against the pt helper output.
    """
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        _build_frame_degree_index as pt_build,
    )

    dp_idx = _build_frame_degree_index(lmax=lmax, mmax=mmax, coefficient_layout=layout)
    pt_idx = pt_build(lmax=lmax, mmax=mmax, coefficient_layout=layout)
    np.testing.assert_array_equal(np.asarray(dp_idx), pt_idx.detach().cpu().numpy())
    # explicit (l, m) check for the packed, untruncated case
    if layout == "packed" and mmax == lmax:
        expected = np.repeat(np.arange(lmax + 1), [2 * l + 1 for l in range(lmax + 1)])
        np.testing.assert_array_equal(np.asarray(dp_idx), expected)


@pytest.mark.parametrize("cls", [DPFrameContract, DPFrameExpand])  # mixer class
def test_torch_namespace(cls) -> None:
    """Mixer ``call`` on torch input matches the numpy-input result.

    Array-API pitfall guard (no ``np.einsum`` on tensors).
    """
    import torch

    lmax, channels, n_frames = 2, 4, 3
    coeff_dim = (lmax + 1) ** 2
    n_batch, n_focus = 3, 2
    rng = np.random.default_rng(606)

    mod = cls(
        lmax=lmax,
        mmax=lmax,
        coefficient_layout="packed",
        n_frames=n_frames,
        channels=channels,
        precision="float64",
        trainable=True,
        seed=9,
    )
    in_ch = mod.weight.shape[1]
    coeff = rng.normal(size=(n_batch, coeff_dim, n_focus, in_ch))
    np_out = mod.call(coeff)
    torch_out = mod.call(torch.from_numpy(coeff))
    np.testing.assert_allclose(
        np.asarray(np_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
