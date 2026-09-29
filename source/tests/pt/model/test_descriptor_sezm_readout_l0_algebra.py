# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CPU algebra checks for the Neo degree-zero readout boundary."""

from __future__ import (
    annotations,
)

import torch

COEFF_DIM = 16
N_FRAMES = 3
GRID_SIZE = 152
HIDDEN_CHANNELS = 192


def _readout_reference(left, right, to_grid, from_grid):
    left_grid = torch.einsum("gj,njh->ngh", to_grid, left)
    right_grid = torch.einsum("gj,njh->ngh", to_grid, right)
    return torch.einsum("g,ngh->nh", from_grid[0], left_grid * right_grid)


def _gram_reference(left, right, gram):
    transformed_right = torch.einsum("ij,njh->nih", gram, right)
    return torch.sum(left * transformed_right, dim=1)


def _output_ffn():
    from deepmd.pt.model.descriptor.sezm_nn.ffn import (
        EquivariantFFN,
    )

    return (
        EquivariantFFN(
            lmax=3,
            channels=32,
            hidden_channels=96,
            kmax=1,
            grid_mlp=True,
            grid_branch=0,
            dtype=torch.float32,
            s2_activation=False,
            ffn_so3_grid=True,
            activation_function="silu",
            glu_activation=True,
            mlp_bias=False,
            trainable=False,
            seed=17,
        )
        .to("cpu")
        .eval()
    )


def test_final_slice_has_exactly_zero_l_greater_than_zero_output_cotangent():
    module = _output_ffn()
    value = torch.randn(
        2,
        COEFF_DIM,
        1,
        32,
        device="cpu",
        requires_grad=True,
    )
    ffn_out = module(value)
    seed = torch.randn(2, 32, device="cpu")

    cotangent = torch.autograd.grad(
        (value + ffn_out)[:, 0, 0, :],
        ffn_out,
        seed,
    )[0]

    assert torch.equal(cotangent[:, 0, 0, :], seed)
    assert torch.count_nonzero(cotangent[:, 1:, :, :]) == 0


def test_degree_zero_nonzero_frame_projector_rows_are_structural_zero():
    projector = _output_ffn().act.projector
    from_grid = projector.from_grid_mat.reshape(
        COEFF_DIM,
        N_FRAMES,
        GRID_SIZE,
    )

    assert projector.frame_set == [0, -1, 1]
    assert torch.count_nonzero(from_grid[0, 0]) > 0
    assert torch.count_nonzero(from_grid[0, 1]) == 0
    assert torch.count_nonzero(from_grid[0, 2]) == 0


def test_scalar_readout_matches_row_zero_of_generic_grid_product():
    generator = torch.Generator().manual_seed(20260704)
    left = 0.1 * torch.randn(
        3,
        COEFF_DIM * N_FRAMES,
        HIDDEN_CHANNELS,
        device="cpu",
        generator=generator,
    )
    right = 0.1 * torch.randn(
        left.shape,
        device="cpu",
        generator=generator,
    )
    projector = _output_ffn().act.projector
    to_grid = projector.to_grid_mat
    from_grid = projector.from_grid_mat

    left_grid = torch.einsum("gj,njh->ngh", to_grid, left)
    right_grid = torch.einsum("gj,njh->ngh", to_grid, right)
    generic = torch.einsum(
        "jg,ngh->njh",
        from_grid,
        left_grid * right_grid,
    )

    actual = _readout_reference(left, right, to_grid, from_grid)

    assert torch.allclose(actual, generic[:, 0, :], atol=5.0e-5, rtol=5.0e-5)


def test_dense_gram_matches_both_projected_input_adjoints():
    from deepmd.pt_expt.kernels.cute.sezm.output_grid.readout_l0 import (
        build_readout_l0_gram,
    )

    generator = torch.Generator().manual_seed(20260718)
    left = 0.1 * torch.randn(
        2,
        COEFF_DIM * N_FRAMES,
        HIDDEN_CHANNELS,
        device="cpu",
        generator=generator,
    )
    right = 0.1 * torch.randn(left.shape, device="cpu", generator=generator)
    dq0 = 0.1 * torch.randn(
        left.shape[0],
        HIDDEN_CHANNELS,
        device="cpu",
        generator=generator,
    )
    projector = _output_ffn().act.projector
    gram = build_readout_l0_gram(
        projector.to_grid_mat,
        projector.from_grid_mat,
    )

    left_ref = left.detach().clone().requires_grad_(True)
    right_ref = right.detach().clone().requires_grad_(True)
    q0 = _readout_reference(
        left_ref,
        right_ref,
        projector.to_grid_mat,
        projector.from_grid_mat,
    )
    expected_left, expected_right = torch.autograd.grad(
        q0,
        (left_ref, right_ref),
        dq0,
    )
    actual_left = dq0[:, None, :] * torch.einsum(
        "ij,njh->nih",
        gram,
        right,
    )
    actual_right = dq0[:, None, :] * torch.einsum(
        "ji,njh->nih",
        gram,
        left,
    )

    assert gram.shape == (COEFF_DIM * N_FRAMES, COEFF_DIM * N_FRAMES)
    assert gram.dtype == torch.float32
    assert gram.is_contiguous()
    assert not gram.requires_grad
    torch.testing.assert_close(actual_left, expected_left, atol=5.0e-5, rtol=5.0e-5)
    torch.testing.assert_close(
        actual_right,
        expected_right,
        atol=5.0e-5,
        rtol=5.0e-5,
    )


def test_dense_gram_forward_matches_row_zero_grid_projection():
    from deepmd.pt_expt.kernels.cute.sezm.output_grid.readout_l0 import (
        build_readout_l0_gram,
    )

    generator = torch.Generator().manual_seed(20260720)
    left = 0.1 * torch.randn(
        3,
        COEFF_DIM * N_FRAMES,
        HIDDEN_CHANNELS,
        device="cpu",
        generator=generator,
    )
    right = 0.1 * torch.randn(left.shape, device="cpu", generator=generator)
    projector = _output_ffn().act.projector
    gram = build_readout_l0_gram(
        projector.to_grid_mat,
        projector.from_grid_mat,
    )

    expected = _readout_reference(
        left,
        right,
        projector.to_grid_mat,
        projector.from_grid_mat,
    )
    actual = _gram_reference(left, right, gram)

    torch.testing.assert_close(actual, expected, atol=5.0e-5, rtol=5.0e-5)


def test_dense_gram_forward_is_channelwise_without_cross_channel_mixing():
    generator = torch.Generator().manual_seed(20260721)
    left = torch.randn(
        2,
        COEFF_DIM * N_FRAMES,
        HIDDEN_CHANNELS,
        device="cpu",
        generator=generator,
    )
    right = torch.randn(left.shape, device="cpu", generator=generator)
    gram = torch.randn(
        COEFF_DIM * N_FRAMES,
        COEFF_DIM * N_FRAMES,
        device="cpu",
        generator=generator,
    )
    changed_channel = 37

    baseline = _gram_reference(left, right, gram)
    changed_right = right.clone()
    changed_right[:, :, changed_channel].mul_(1.5)
    changed = _gram_reference(left, changed_right, gram)

    unaffected = torch.ones(HIDDEN_CHANNELS, dtype=torch.bool, device="cpu")
    unaffected[changed_channel] = False
    torch.testing.assert_close(changed[:, unaffected], baseline[:, unaffected])
    torch.testing.assert_close(
        changed[:, changed_channel],
        1.5 * baseline[:, changed_channel],
    )
