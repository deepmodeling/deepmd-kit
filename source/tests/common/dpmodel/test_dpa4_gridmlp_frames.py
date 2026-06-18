# SPDX-License-Identifier: LGPL-3.0-or-later
"""Parity tests for the frame-aware DPA4 ``GridMLP``.

These mirror the current pt ``deepmd.pt.model.descriptor.sezm_nn.grid_net``
``GridMLP``, which packs operands as ``(N, D, F, n_frames * C)`` and projects
each Wigner-D frame independently. pt imports live inside the test functions
because ruff TID253 bans module-level ``deepmd.pt`` imports under
``source/tests/common``.

The ``to_grid``/``from_grid`` callables are supplied as namespace-agnostic
closures that reproduce the pt ``BaseGridNet`` frame-aware projector einsums
(``"gdk,ndfkc->ngfc"`` / ``"dkg,ngfc->ndfkc"``) with random matrices. The same
matrices are fed to both backends, so the closures only need to be identical,
not orthonormal. ``test_gridmlp_s2_regression`` additionally checks the
``n_frames == 1`` path against a real S2 projector's grid functions.
"""

import array_api_compat
import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
    GridMLP as DPGridMLP,
)


def _make_grid_fns(to_mat, from_mat, n_frames):
    """Build namespace-agnostic frame-aware ``to_grid``/``from_grid`` closures.

    Parameters
    ----------
    to_mat : np.ndarray
        Shape ``(G, D, n_frames)``; reproduces ``projector.to_grid_mat``.
    from_mat : np.ndarray
        Shape ``(D, n_frames, G)``; reproduces ``projector.from_grid_mat``.
    n_frames : int
        Number of Wigner-D frames.
    """

    def to_grid(coeff):
        # einsum "gdk,ndfkc->ngfc": sum over d (coeff dim) and k (frame).
        xp = array_api_compat.array_namespace(coeff)
        dev = array_api_compat.device(coeff)
        mat = xp.asarray(to_mat, device=dev)
        if mat.dtype != coeff.dtype:
            mat = xp.astype(mat, coeff.dtype)
        n_batch, coeff_dim, n_focus, _ = coeff.shape
        # (N, D, F, n_frames, C)
        cv = xp.reshape(coeff, (n_batch, coeff_dim, n_focus, n_frames, -1))
        # (N, G, D, F, n_frames, C)
        prod = mat[None, :, :, None, :, None] * cv[:, None, :, :, :, :]
        prod = xp.sum(prod, axis=4)  # contract frame -> (N, G, D, F, C)
        return xp.sum(prod, axis=2)  # contract coeff dim -> (N, G, F, C)

    def from_grid(grid):
        # einsum "dkg,ngfc->ndfkc": sum over g (grid point).
        xp = array_api_compat.array_namespace(grid)
        dev = array_api_compat.device(grid)
        mat = xp.asarray(from_mat, device=dev)
        if mat.dtype != grid.dtype:
            mat = xp.astype(mat, grid.dtype)
        n_batch, _, n_focus, n_channels = grid.shape
        coeff_dim = from_mat.shape[0]
        # (N, F, G, C) so the grid axis lands at position 4 of the 6D product
        grid_p = xp.permute_dims(grid, (0, 2, 1, 3))
        # (N, D, F, n_frames, G, C)
        prod = mat[None, :, None, :, :, None] * grid_p[:, None, :, None, :, :]
        prod = xp.sum(prod, axis=4)  # contract grid -> (N, D, F, n_frames, C)
        return xp.reshape(prod, (n_batch, coeff_dim, n_focus, n_frames * n_channels))

    return to_grid, from_grid


def _copy_pt_to_dp(pt_mlp, dp_mlp):
    """Copy pt ``GridMLP`` state-dict weights into the dpmodel ``GridMLP``."""
    state = {k: v.detach().cpu().numpy() for k, v in pt_mlp.state_dict().items()}
    assert set(state) == {
        "left_proj.weight",
        "right_proj.weight",
        "out_proj.weight",
    }
    dp_mlp.left_proj.weight = state["left_proj.weight"]
    dp_mlp.right_proj.weight = state["right_proj.weight"]
    dp_mlp.out_proj.weight = state["out_proj.weight"]


@pytest.mark.parametrize("n_frames", [1, 2, 3])  # number of Wigner-D frames
@pytest.mark.parametrize("mode", ["self", "cross"])  # operand pairing mode
def test_gridmlp_parity(n_frames, mode) -> None:
    """The dpmodel ``GridMLP`` matches pt over identical frame-aware grid fns."""
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        GridMLP as PTGridMLP,
    )

    channels, n_batch, coeff_dim, n_focus, grid_size = 4, 5, 9, 2, 7
    rng = np.random.default_rng(2026)
    to_mat = rng.normal(size=(grid_size, coeff_dim, n_frames))
    from_mat = rng.normal(size=(coeff_dim, n_frames, grid_size))
    np_to_grid, np_from_grid = _make_grid_fns(to_mat, from_mat, n_frames)

    # pin to CPU so torch.from_numpy fp64 inputs and the module agree under the
    # CUDA-default-device CI configuration
    pt_mlp = PTGridMLP(
        channels=channels,
        mode=mode,
        n_frames=n_frames,
        dtype=torch.float64,
        trainable=True,
        seed=7,
    ).to("cpu")
    with torch.no_grad():
        for p in pt_mlp.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    dp_mlp = DPGridMLP(
        channels=channels,
        mode=mode,
        n_frames=n_frames,
        precision="float64",
        trainable=True,
        seed=7,
    )
    _copy_pt_to_dp(pt_mlp, dp_mlp)

    left = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))
    right = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))

    dp_out = dp_mlp.call(left, right, to_grid=np_to_grid, from_grid=np_from_grid)
    pt_out = pt_mlp(
        torch.from_numpy(left),
        torch.from_numpy(right),
        None,
        to_grid=np_to_grid,
        from_grid=np_from_grid,
    )
    assert dp_out.shape == (n_batch, coeff_dim, n_focus, n_frames * channels)
    np.testing.assert_allclose(
        np.asarray(dp_out),
        pt_out.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ["self", "cross"])  # operand pairing mode
def test_gridmlp_s2_regression(mode) -> None:
    """``n_frames == 1`` ``GridMLP`` matches pt over a real S2 projector.

    Guards the S2 path: the frame-aware reshape with ``n_frames == 1`` is an
    identity, so the output stays byte-identical to the previous S2-only
    specialization (which this generalization replaces).
    """
    import torch

    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        GridMLP as PTGridMLP,
        S2GridNet as PTS2GridNet,
    )

    from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import (
        S2GridNet as DPS2GridNet,
    )

    lmax, channels, n_focus = 2, 4, 1
    # op_type='glu' makes grid_op a GridProduct; we reuse the nets only for
    # their (parameter-free, deterministic) _to_grid/_from_grid S2 projectors.
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
    rng = np.random.default_rng(99)

    pt_mlp = PTGridMLP(
        channels=channels,
        mode=mode,
        n_frames=1,
        dtype=torch.float64,
        trainable=True,
        seed=13,
    ).to("cpu")
    with torch.no_grad():
        for p in pt_mlp.parameters():
            p += torch.from_numpy(0.1 * rng.normal(size=tuple(p.shape)))

    dp_mlp = DPGridMLP(
        channels=channels,
        mode=mode,
        n_frames=1,
        precision="float64",
        trainable=True,
        seed=13,
    )
    _copy_pt_to_dp(pt_mlp, dp_mlp)

    left = rng.normal(size=(5, coeff_dim, n_focus, channels))
    right = rng.normal(size=(5, coeff_dim, n_focus, channels))

    dp_out = dp_mlp.call(
        left, right, to_grid=dp_net._to_grid, from_grid=dp_net._from_grid
    )
    pt_out = pt_mlp(
        torch.from_numpy(left),
        torch.from_numpy(right),
        None,
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


@pytest.mark.parametrize("mode", ["self", "cross"])  # operand pairing mode
def test_gridmlp_serialize_roundtrip(mode) -> None:
    """Serialize -> deserialize -> forward is identical; n_frames in config."""
    channels, n_frames, n_batch, coeff_dim, n_focus, grid_size = 4, 2, 3, 9, 2, 7
    rng = np.random.default_rng(404)
    to_mat = rng.normal(size=(grid_size, coeff_dim, n_frames))
    from_mat = rng.normal(size=(coeff_dim, n_frames, grid_size))
    to_grid, from_grid = _make_grid_fns(to_mat, from_mat, n_frames)

    mlp = DPGridMLP(
        channels=channels,
        mode=mode,
        n_frames=n_frames,
        precision="float64",
        trainable=True,
        seed=5,
    )
    # perturb to non-default weights
    mlp.left_proj.weight = mlp.left_proj.weight + 0.1 * rng.normal(
        size=mlp.left_proj.weight.shape
    )
    mlp.right_proj.weight = mlp.right_proj.weight + 0.1 * rng.normal(
        size=mlp.right_proj.weight.shape
    )
    mlp.out_proj.weight = mlp.out_proj.weight + 0.1 * rng.normal(
        size=mlp.out_proj.weight.shape
    )

    data = mlp.serialize()
    assert data["@version"] == 1
    assert data["config"]["n_frames"] == n_frames
    restored = DPGridMLP.deserialize(data)

    left = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))
    right = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))
    out0 = mlp.call(left, right, to_grid=to_grid, from_grid=from_grid)
    out1 = restored.call(left, right, to_grid=to_grid, from_grid=from_grid)
    np.testing.assert_allclose(
        np.asarray(out0), np.asarray(out1), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("mode", ["self", "cross"])  # operand pairing mode
def test_gridmlp_torch_namespace(mode) -> None:
    """``GridMLP.call`` on torch input matches the numpy-input result.

    Array-API pitfall guard: the dpmodel forward must work with any namespace.
    """
    import torch

    channels, n_frames, n_batch, coeff_dim, n_focus, grid_size = 4, 2, 3, 9, 2, 7
    rng = np.random.default_rng(77)
    to_mat = rng.normal(size=(grid_size, coeff_dim, n_frames))
    from_mat = rng.normal(size=(coeff_dim, n_frames, grid_size))
    to_grid, from_grid = _make_grid_fns(to_mat, from_mat, n_frames)

    mlp = DPGridMLP(
        channels=channels,
        mode=mode,
        n_frames=n_frames,
        precision="float64",
        trainable=True,
        seed=9,
    )
    left = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))
    right = rng.normal(size=(n_batch, coeff_dim, n_focus, n_frames * channels))

    np_out = mlp.call(left, right, to_grid=to_grid, from_grid=from_grid)
    torch_out = mlp.call(
        torch.from_numpy(left),
        torch.from_numpy(right),
        to_grid=to_grid,
        from_grid=from_grid,
    )
    np.testing.assert_allclose(
        np.asarray(np_out),
        torch_out.detach().cpu().numpy(),
        rtol=1e-12,
        atol=1e-12,
    )
