# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Focused contracts for Neo's C=96 block-FFN grid fusion."""

from __future__ import (
    annotations,
)

import pytest
import torch

from deepmd.pt.model.descriptor.sezm import (
    DescrptSeZM,
)
from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
    GridBranch,
    GridMLP,
    S2GridNet,
)


def test_single_branch_bypasses_router_and_accepts_fused_middle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    branch = (
        GridBranch(
            channels=2,
            n_branches=1,
            n_frames=3,
            dtype=torch.float32,
            trainable=False,
            seed=7,
        )
        .to("cpu")
        .eval()
    )
    monkeypatch.setattr(
        branch.router,
        "forward",
        lambda value: pytest.fail("single-branch router must not run"),
    )
    calls = 0

    def fused_middle(
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return left * right

    left = torch.randn(2, 4, 1, 6, device="cpu")
    out = branch(
        left,
        torch.randn_like(left),
        torch.randn(2, 1, 4, device="cpu"),
        to_grid=lambda value: value,
        from_grid=lambda value: value,
        grid_product=fused_middle,
    )

    assert calls == 1
    assert out.shape == left.shape


def test_single_branch_keeps_pytorch_middle_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    branch = (
        GridBranch(
            channels=2,
            n_branches=1,
            n_frames=1,
            dtype=torch.float32,
            trainable=False,
            seed=11,
        )
        .to("cpu")
        .eval()
    )
    monkeypatch.setattr(
        branch.router,
        "forward",
        lambda value: pytest.fail("single-branch router must not run"),
    )
    calls = {"to_grid": 0, "from_grid": 0}

    def to_grid(value: torch.Tensor) -> torch.Tensor:
        calls["to_grid"] += 1
        return value

    def from_grid(value: torch.Tensor) -> torch.Tensor:
        calls["from_grid"] += 1
        return value

    left = torch.randn(2, 4, 1, 2, device="cpu")
    out = branch(
        left,
        torch.randn_like(left),
        torch.randn(2, 1, 4, device="cpu"),
        to_grid=to_grid,
        from_grid=from_grid,
    )

    assert calls == {"to_grid": 2, "from_grid": 1}
    assert out.shape == left.shape


def test_single_branch_opt_in_matches_routed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = (
        GridBranch(
            channels=2,
            n_branches=1,
            n_frames=1,
            dtype=torch.float32,
            trainable=False,
            seed=17,
        )
        .to("cpu")
        .eval()
    )
    left = torch.randn(2, 4, 1, 2, device="cpu")
    right = torch.randn_like(left)
    scalar_pair = torch.randn(2, 1, 4, device="cpu")
    original_forward = branch.router.forward
    router_calls = 0

    def tracked_router(value: torch.Tensor) -> torch.Tensor:
        nonlocal router_calls
        router_calls += 1
        return original_forward(value)

    monkeypatch.setattr(branch.router, "forward", tracked_router)
    monkeypatch.delenv("DP_NEO_CUTE_INFER", raising=False)
    routed = branch(
        left,
        right,
        scalar_pair,
        to_grid=lambda value: value,
        from_grid=lambda value: value,
    )
    monkeypatch.setenv("DP_NEO_CUTE_INFER", "1")
    shortcut = branch(
        left,
        right,
        scalar_pair,
        to_grid=lambda value: value,
        from_grid=lambda value: value,
    )

    assert router_calls == 1
    torch.testing.assert_close(shortcut, routed)


def test_multi_branch_preserves_softmax_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = GridBranch(
        channels=2,
        n_branches=2,
        n_frames=1,
        dtype=torch.float32,
        trainable=False,
        seed=13,
    ).to("cpu")
    original_softmax = torch.softmax
    softmax_calls = 0

    def tracked_softmax(
        value: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        nonlocal softmax_calls
        softmax_calls += 1
        return original_softmax(value, dim=dim)

    monkeypatch.setattr(torch, "softmax", tracked_softmax)
    left = torch.randn(2, 4, 1, 2, device="cpu")
    out = branch(
        left,
        torch.randn_like(left),
        torch.randn(2, 1, 4, device="cpu"),
        to_grid=lambda value: value,
        from_grid=lambda value: value,
        grid_product=lambda left, right: pytest.fail(
            "multi-branch routing must keep the generic path"
        ),
    )

    assert softmax_calls == 1
    assert out.shape == left.shape
    assert torch.isfinite(out).all()


def test_single_branch_training_preserves_router_zero_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = (
        GridBranch(
            channels=2,
            n_branches=1,
            n_frames=1,
            dtype=torch.float32,
            trainable=True,
            seed=17,
        )
        .to("cpu")
        .train()
    )
    original_softmax = torch.softmax
    softmax_calls = 0

    def tracked_softmax(
        value: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        nonlocal softmax_calls
        softmax_calls += 1
        return original_softmax(value, dim=dim)

    monkeypatch.setattr(torch, "softmax", tracked_softmax)
    left = torch.randn(2, 4, 1, 2, device="cpu", requires_grad=True)
    right = torch.randn_like(left)
    out = branch(
        left,
        right,
        torch.randn(2, 1, 4, device="cpu"),
        to_grid=lambda value: value,
        from_grid=lambda value: value,
        grid_product=lambda left, right: pytest.fail(
            "training must keep the generic router path"
        ),
    )
    out.sum().backward()

    assert softmax_calls == 1
    assert branch.router.weight.grad is not None
    assert torch.count_nonzero(branch.router.weight.grad) == 0


def test_single_branch_eval_with_trainable_parameters_preserves_router_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = (
        GridBranch(
            channels=2,
            n_branches=1,
            n_frames=1,
            dtype=torch.float32,
            trainable=True,
            seed=19,
        )
        .to("cpu")
        .eval()
    )
    left = torch.randn(2, 4, 1, 2, device="cpu", requires_grad=True)
    out = branch(
        left,
        torch.randn_like(left),
        torch.randn(2, 1, 4, device="cpu"),
        to_grid=lambda value: value,
        from_grid=lambda value: value,
        grid_product=lambda left, right: pytest.fail(
            "trainable eval must preserve the generic router path"
        ),
    )
    out.sum().backward()

    assert branch.router.weight.grad is not None
    assert torch.count_nonzero(branch.router.weight.grad) == 0


@pytest.mark.parametrize("training", (False, True))
def test_grid_net_with_trainable_parameters_does_not_offer_fused_product(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
) -> None:
    net = S2GridNet(
        lmax=1,
        mmax=1,
        channels=2,
        n_focus=1,
        mode="self",
        op_type="mlp",
        dtype=torch.float32,
        layout="ndfc",
        coefficient_layout="packed",
        grid_method="e3nn",
        trainable=True,
        seed=23,
    ).to("cpu")
    net.train(training)
    monkeypatch.setattr(
        net,
        "_grid_product",
        lambda left, right: pytest.fail(
            "trainable grid net must not offer its first-order-only fused product"
        ),
    )

    query = torch.randn(2, 4, 1, 4, device="cpu", requires_grad=True)
    net(query).sum().backward()
    assert query.grad is not None


def test_frozen_eval_grid_net_offers_fused_product(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    net = (
        S2GridNet(
            lmax=1,
            mmax=1,
            channels=2,
            n_focus=1,
            mode="self",
            op_type="mlp",
            dtype=torch.float32,
            layout="ndfc",
            coefficient_layout="packed",
            grid_method="e3nn",
            trainable=False,
            seed=29,
        )
        .to("cpu")
        .eval()
    )
    calls = 0

    def tracked_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return net._from_grid(net._to_grid(left) * net._to_grid(right))

    monkeypatch.setattr(net, "_grid_product", tracked_product)
    output = net(torch.randn(2, 4, 1, 4, device="cpu"))

    assert calls == 1
    assert output.shape == (2, 4, 1, 2)


def test_exact_neo_has_two_c96_block_products_and_c192_readout() -> None:
    descriptor = DescrptSeZM(
        ntypes=2,
        sel=4,
        channels=32,
        lmax=3,
        mmax=1,
        n_blocks=2,
        so2_layers=3,
        n_focus=2,
        message_node_so3=True,
        ffn_neurons=0,
        ffn_so3_grid=True,
        grid_branch=[0, 0, 1],
        ffn_blocks=1,
        so3_readout="mlp",
        use_amp=False,
        precision="float32",
        trainable=False,
        seed=42,
    )

    assert len(descriptor.blocks) == 2
    for block in descriptor.blocks:
        assert len(block.ffns) == 1
        grid_op = block.ffns[0].act.grid_op
        assert isinstance(grid_op, GridBranch)
        assert grid_op.n_branches == 1
        assert grid_op.channels == 96

    readout_grid_op = descriptor.output_ffn.act.grid_op
    assert isinstance(readout_grid_op, GridMLP)
    assert readout_grid_op.hidden_channels == 192
