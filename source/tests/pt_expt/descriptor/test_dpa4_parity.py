# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inversion symmetry of the DPA4 building blocks with natural parity."""

from __future__ import (
    annotations,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as DPSO2Linear
from deepmd.pt_expt.descriptor.dpa4 import (
    _promote_trainable_tree,
)
from deepmd.pt_expt.descriptor.dpa4_nn.grid_net import (
    S2GridNet,
    SO3GridNet,
)
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import (
    SO2Linear,
)


def _second_order(
    outputs: tuple[torch.Tensor, ...], leaves: tuple[torch.Tensor, ...]
) -> list[tuple[torch.Tensor, ...]]:
    """First and second derivatives of ``sum(output**2)`` for each output."""
    result = []
    for output in outputs:
        first = torch.autograd.grad(output.square().sum(), leaves, create_graph=True)
        second = torch.autograd.grad(sum(item.square().sum() for item in first), leaves)
        result.append((*first, *second))
    return result


@pytest.mark.parametrize("parity", [False, True])  # natural parity or not
@pytest.mark.parametrize("mmax", [1, 3])  # highest SO(2) order
def test_so2_local_mirror(parity: bool, mmax: int) -> None:
    """Natural-parity local maps commute with the reflection through the edge."""
    layer = SO2Linear(
        lmax=3,
        mmax=mmax,
        parity=parity,
        in_channels=4,
        out_channels=5,
        n_focus=2,
        precision="float64",
        mlp_bias=True,
        seed=31,
        trainable=True,
    ).cpu()
    mirror = [1] * 4
    for m in range(1, mmax + 1):
        # Each order stores its negative-m rows before its positive-m rows.
        mirror.extend([-((-1) ** m)] * (4 - m))
        mirror.extend([(-1) ** m] * (4 - m))
    sign = np.asarray(mirror, dtype=np.float64)[None, None, :, None]
    x = np.random.default_rng(37).normal(size=(2, 7, layer.reduced_dim, 4))
    reflected = layer(torch.from_numpy(x * sign))
    expected = layer(torch.from_numpy(x)) * torch.from_numpy(sign)
    if parity:
        torch.testing.assert_close(reflected, expected, atol=1e-13, rtol=1e-13)
    else:
        assert (reflected - expected).abs().max() > 1e-5
    portable = DPSO2Linear.deserialize(layer.serialize())
    np.testing.assert_allclose(
        portable.call(x * sign), reflected.detach().numpy(), atol=1e-13, rtol=1e-13
    )


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 1), (3, 3)])  # order coverage
@pytest.mark.parametrize("edges", [0, 5])  # empty and populated edge axis
def test_parity_linear_matches_explicit_zero_blocks(
    lmax: int, mmax: int, edges: int
) -> None:
    """Shared order weights match the dense map through second derivatives."""
    module = SO2Linear(
        lmax=lmax,
        mmax=mmax,
        in_channels=2,
        out_channels=3,
        n_focus=2,
        parity=True,
        mlp_bias=True,
        seed=132,
        trainable=True,
        precision="float64",
    ).cpu()
    generator = torch.Generator().manual_seed(133)
    value = torch.randn(
        2,
        edges,
        module.reduced_dim,
        2,
        generator=generator,
        dtype=torch.float64,
        requires_grad=True,
    )
    weight = module._build_so2_weight()
    dense = torch.einsum(
        "fei,ifo->feo", value.reshape(2, edges, module.reduced_dim * 2), weight
    ).reshape(2, edges, module.reduced_dim, 3)
    bias = module.bias0.reshape(2, 1, 1, 3)
    expected = torch.cat([dense[:, :, :1] + bias, dense[:, :, 1:]], dim=2)
    actual = module(value)
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=2e-14)
    leaves = (value, *module.parameters())
    want, got = _second_order((expected, actual), leaves)
    for left, right in zip(got, want, strict=True):
        torch.testing.assert_close(left, right, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operator
@pytest.mark.parametrize("mode", ["self", "cross"])  # one or two operands
@pytest.mark.parametrize(
    "grid,kmax", [("so3", 0), ("so3", 1), ("so3", 2), ("s2", 0)]
)  # grid family and SO(3) projection order
def test_grid_inversion_projection_matches_group_average(
    op_type: str, mode: str, grid: str, kmax: int
) -> None:
    """The parity product matches the group average through second derivatives."""
    GridNet = S2GridNet if grid == "s2" else SO3GridNet
    projection = {"grid_resolution_list": [7, 8]} if grid == "s2" else {"kmax": kmax}
    kwargs = dict(
        lmax=2,
        mmax=1,
        channels=2,
        n_focus=2,
        mode=mode,
        op_type=op_type,
        layout="ndfc",
        coefficient_layout="m_major",
        grid_branches=2,
        mlp_bias=True,
        residual_scale_init=0.7,
        trainable=True,
        seed=162,
        precision="float64",
        **projection,
    )
    unrestricted = _promote_trainable_tree(GridNet(**kwargs, parity=False).cpu())
    projected = _promote_trainable_tree(GridNet(**kwargs, parity=True).cpu())
    projected.load_state_dict(unrestricted.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(163)
    inputs = [
        torch.randn(
            2,
            7,
            2,
            unrestricted.query_channels,
            dtype=torch.float64,
            generator=generator,
        )
    ]
    if mode == "cross":
        inputs.append(
            torch.randn(
                2,
                7,
                2,
                unrestricted.context_channels,
                dtype=torch.float64,
                generator=generator,
            )
        )
    degree = torch.tensor([0, 1, 2, 1, 2, 1, 2])
    sign = (1 - 2 * (degree % 2))[None, :, None, None]
    results = []
    for net, average in ((unrestricted, True), (projected, False)):
        values = [value.clone().requires_grad_() for value in inputs]
        output = net(*values)
        if average:
            output = 0.5 * (output + sign * net(*(value * sign for value in values)))
        derivatives = torch.autograd.grad(
            output.square().sum(), values, create_graph=True
        )
        loss = output.sum() + sum(value.square().sum() for value in derivatives)
        gradients = torch.autograd.grad(
            loss, tuple(net.parameters()), allow_unused=True
        )
        results.append(((output,), derivatives, gradients))
    for expected, actual in zip(*results, strict=True):
        for left, right in zip(expected, actual, strict=True):
            if left is None:
                assert right is None
            else:
                torch.testing.assert_close(left, right, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_grid_inversion_projection_uses_loaded_quadrature(device: str) -> None:
    """Checkpoint quadratures govern parity products across precision changes."""
    kwargs = {
        "lmax": 2,
        "kmax": 1,
        "channels": 2,
        "n_focus": 1,
        "mode": "cross",
        "op_type": "glu",
        "layout": "ndfc",
        "trainable": False,
        "seed": 165,
    }
    source = SO3GridNet(**kwargs, precision="float32").cpu()
    reference = SO3GridNet(**kwargs, precision="float64").to(device)
    projected = SO3GridNet(**kwargs, precision="float64", parity=True).to(device)
    for net in (reference, projected):
        net.load_state_dict(source.state_dict(), strict=True)
    generator = torch.Generator(device=device).manual_seed(166)
    query, context = (
        torch.randn(2, 9, 1, 2, dtype=torch.float64, device=device, generator=generator)
        for _ in range(2)
    )
    degree = torch.tensor([0, 1, 1, 1, 2, 2, 2, 2, 2], device=device)
    sign = (1 - 2 * (degree % 2))[None, :, None, None]
    expected = 0.5 * (
        reference(query, context) + sign * reference(sign * query, sign * context)
    )
    torch.testing.assert_close(
        projected(query, context), expected, atol=2e-12, rtol=2e-12
    )


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operator
def test_inversion_projection_retains_one_quadrature(op_type: str) -> None:
    """Quadratic parity products keep the original grid and empty batches."""
    module = SO3GridNet(
        lmax=2,
        kmax=1,
        channels=2,
        n_focus=2,
        mode="self",
        op_type=op_type,
        grid_branches=2,
        layout="ndfc",
        parity=True,
        trainable=True,
        precision="float64",
    ).cpu()
    assert module._grid_size == module.projector.grid_size
    query = torch.empty(0, 9, 2, module.query_channels, dtype=torch.float64)
    assert module(query).shape == (0, 9, 2, 6)


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])  # grid operator
@pytest.mark.parametrize("mode", ["self", "cross"])  # one or two operands
def test_parity_scalar_on_nonclosed_sphere_grid(op_type: str, mode: str) -> None:
    """Scalar-only synthesis keeps the even part on nonclosed sphere grids."""
    kwargs = {
        "lmax": 2,
        "channels": 2,
        "n_focus": 2,
        "mode": mode,
        "op_type": op_type,
        "grid_branches": 2,
        "grid_method": "e3nn",
        "grid_resolution_list": [7, 8],
        "layout": "ndfc",
        "trainable": True,
        "seed": 141,
        "precision": "float64",
    }
    reference = S2GridNet(**kwargs).cpu()
    projected = S2GridNet(**kwargs, parity=True).cpu()
    projected.load_state_dict(reference.state_dict())
    generator = torch.Generator().manual_seed(142)
    query = torch.randn(
        2,
        9,
        2,
        projected.query_channels,
        dtype=torch.float64,
        generator=generator,
        requires_grad=True,
    )
    context = (
        torch.randn(
            2,
            9,
            2,
            projected.context_channels,
            dtype=torch.float64,
            generator=generator,
            requires_grad=True,
        )
        if mode == "cross"
        else None
    )
    sign = torch.tensor([1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=torch.float64)[
        None, :, None, None
    ]
    reflected_context = context * sign if context is not None else None
    expected = 0.5 * (
        reference.call_scalar(query, context)
        + reference.call_scalar(query * sign, reflected_context)
    )
    actual = projected.call_scalar(query, context)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        actual, projected(query, context)[:, :1], atol=2e-12, rtol=2e-12
    )
    leaves = (query, context) if context is not None else (query,)
    want, got = _second_order((expected, actual), leaves)
    for left, right in zip(got, want, strict=True):
        torch.testing.assert_close(left, right, atol=2e-11, rtol=2e-11)
