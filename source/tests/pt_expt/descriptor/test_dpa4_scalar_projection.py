# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for the optimized DPA4 scalar projection paths."""

from collections.abc import (
    Iterable,
)

import torch

from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
    SO3Linear,
)
from deepmd.pt_expt.common import (
    try_convert_module,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
    _promote_trainable_tree,
)


def _make_descriptor() -> DescrptDPA4:
    return DescrptDPA4(
        ntypes=2,
        sel=8,
        rcut=4.0,
        channels=4,
        n_radial=4,
        lmax=2,
        mmax=1,
        kmax=1,
        n_blocks=1,
        l_schedule=[2],
        n_atten_head=1,
        grid_mlp=[True, False, False],
        grid_branch=[0, 0, 0],
        node_wise_so3=True,
        so3_readout="mlp",
        readout_layers=2,
        random_gamma=False,
        precision="float64",
        use_amp=False,
        seed=7,
    ).to("cpu")


def _assert_output_and_gradient_parity(
    full_output: torch.Tensor,
    scalar_output: torch.Tensor,
    full_inputs: tuple[torch.Tensor, ...],
    scalar_inputs: tuple[torch.Tensor, ...],
    parameters: Iterable[torch.nn.Parameter],
    probe: torch.Tensor,
) -> None:
    """Compare outputs and first derivatives of two shared-parameter graphs."""
    parameters = tuple(parameters)
    assert parameters
    torch.testing.assert_close(full_output, scalar_output, atol=1e-12, rtol=1e-12)

    full_grads = torch.autograd.grad(
        torch.sum(full_output * probe),
        (*full_inputs, *parameters),
        allow_unused=True,
    )
    scalar_grads = torch.autograd.grad(
        torch.sum(scalar_output * probe),
        (*scalar_inputs, *parameters),
        allow_unused=True,
    )
    assert all(gradient is not None for gradient in full_grads[: len(full_inputs)])
    assert any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in full_grads[len(full_inputs) :]
    )
    for full_grad, scalar_grad in zip(full_grads, scalar_grads, strict=True):
        assert (full_grad is None) == (scalar_grad is None)
        if full_grad is not None:
            torch.testing.assert_close(
                full_grad,
                scalar_grad,
                atol=1e-12,
                rtol=1e-12,
            )


def test_so3_linear_multi_focus_scalar_projection_matches_full() -> None:
    """Scalar projection preserves independent focus batches and gradients."""
    layer = try_convert_module(
        SO3Linear(
            lmax=2,
            in_channels=3,
            out_channels=4,
            n_focus=2,
            mlp_bias=True,
            precision="float64",
            seed=8430,
        )
    )
    assert layer is not None
    layer = _promote_trainable_tree(layer).to("cpu")

    torch.manual_seed(8430)
    full_input = torch.randn(
        3,
        (layer.lmax + 1) ** 2,
        layer.n_focus,
        layer.in_channels,
        dtype=torch.float64,
        requires_grad=True,
    )
    scalar_input = full_input.detach().clone().requires_grad_(True)
    probe = torch.randn(
        3,
        1,
        layer.n_focus,
        layer.out_channels,
        dtype=torch.float64,
    )

    full = layer(full_input)[:, 0:1, :, :]
    scalar = layer.call_scalar(scalar_input)
    _assert_output_and_gradient_parity(
        full,
        scalar,
        (full_input,),
        (scalar_input,),
        layer.parameters(),
        probe,
    )


def test_self_grid_mlp_scalar_readout_matches_full_projection() -> None:
    """Direct Haar contraction matches the full self-grid output and gradients."""
    descriptor = _make_descriptor()
    net = descriptor.output_ffn.act
    assert net.mode == "self"
    assert net.op_type == "mlp"
    assert net.layout == "ndfc"

    torch.manual_seed(8450)
    coeff_dim = net.projector.coeff_dim // net.n_frames
    full_input = torch.randn(
        2,
        coeff_dim,
        net.n_focus,
        net.query_channels,
        dtype=torch.float64,
        requires_grad=True,
    )
    scalar_input = full_input.detach().clone().requires_grad_(True)
    probe = torch.randn(
        2,
        1,
        net.n_focus,
        net.output_channels,
        dtype=torch.float64,
    )

    full = net(full_input)[:, 0:1, :, :]
    scalar = net.call_scalar(scalar_input)
    _assert_output_and_gradient_parity(
        full,
        scalar,
        (full_input,),
        (scalar_input,),
        net.parameters(),
        probe,
    )


def test_cross_grid_mlp_scalar_readout_matches_full_projection() -> None:
    """The scalar cross path contracts only the degree-zero frame weights."""
    descriptor = _make_descriptor()
    net = descriptor.blocks[0].so2_conv.node_wise_grid_product
    assert net.mode == "cross"
    assert net.op_type == "mlp"
    assert net.layout == "flat"

    torch.manual_seed(8460)
    coeff_dim = net.projector.coeff_dim // net.n_frames
    shape = (2, coeff_dim, net.n_focus * net.context_channels)
    full_query = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    full_context = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    scalar_query = full_query.detach().clone().requires_grad_(True)
    scalar_context = full_context.detach().clone().requires_grad_(True)
    probe = torch.randn(
        2,
        1,
        net.n_focus * net.output_channels,
        dtype=torch.float64,
    )

    full = net(full_query, full_context)[:, 0:1, :]
    scalar = net.call_scalar(scalar_query, scalar_context)
    _assert_output_and_gradient_parity(
        full,
        scalar,
        (full_query, full_context),
        (scalar_query, scalar_context),
        net.parameters(),
        probe,
    )


def test_self_grid_mlp_paired_projection_matches_separate_projections() -> None:
    """The shared transform preserves full-grid outputs and first derivatives."""
    descriptor = _make_descriptor()
    net = descriptor.output_ffn.act
    assert net._combine_grid_projection
    net.train()

    torch.manual_seed(8440)
    coeff_dim = net.projector.coeff_dim // net.n_frames
    paired_input = torch.randn(
        2,
        coeff_dim,
        net.n_focus,
        net.query_channels,
        dtype=torch.float64,
        requires_grad=True,
    )
    separate_input = paired_input.detach().clone().requires_grad_(True)
    probe = torch.randn(
        2,
        coeff_dim,
        net.n_focus,
        net.output_channels,
        dtype=torch.float64,
    )

    paired_output = net(paired_input)
    net._combine_grid_projection = False
    separate_output = net(separate_input)
    _assert_output_and_gradient_parity(
        paired_output,
        separate_output,
        (paired_input,),
        (separate_input,),
        net.parameters(),
        probe,
    )


def test_descriptor_readout_scalar_path_matches_full_projection() -> None:
    """The descriptor readout preserves full-stack outputs and gradients."""
    descriptor = _make_descriptor()
    parameters = tuple(
        parameter
        for name, parameter in descriptor.named_parameters()
        if name.startswith(("readout_pre_layers.", "output_ffn."))
    )
    generator = torch.Generator(device="cpu").manual_seed(8470)
    with torch.no_grad():
        for parameter in parameters:
            parameter.add_(
                0.05
                * torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
            )

    full_input = torch.randn(
        3,
        descriptor.node_readout_dim,
        1,
        descriptor.channels,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    scalar_input = full_input.detach().clone().requires_grad_(True)

    full_hidden = full_input
    for layer in descriptor.readout_pre_layers:
        full_hidden = full_hidden + layer(full_hidden)
    full = (full_hidden + descriptor.output_ffn(full_hidden))[:, 0:1, :, :]
    scalar = descriptor._apply_readout(scalar_input, scalar_input.shape[0])
    probe = torch.randn(
        full.shape,
        dtype=full.dtype,
        device=full.device,
        generator=generator,
    )
    _assert_output_and_gradient_parity(
        full,
        scalar,
        (full_input,),
        (scalar_input,),
        parameters,
        probe,
    )
