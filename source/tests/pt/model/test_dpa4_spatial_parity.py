# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spatial parity of DPA4 angular operators and adapters."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("lmax", [1, 2, 3])
def test_so2_linear_preserves_bond_plane_reflection(lmax: int) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as DPLinear
    from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Linear as PTLinear
    from deepmd.pt_expt.descriptor.dpa4_nn.so2 import SO2Linear as ExptLinear

    linear = PTLinear(
        lmax=lmax,
        in_channels=2,
        out_channels=3,
        n_focus=2,
        parity=True,
        dtype=torch.float64,
        seed=7,
        trainable=True,
    ).to("cpu")
    dp_linear = DPLinear.deserialize(linear.serialize())
    rng = np.random.default_rng(13)
    x = rng.normal(size=(2, 4, linear.reduced_dim, 2))
    reflection = np.ones(linear.reduced_dim)
    reflection[linear.neg_indices.cpu().numpy()] = -1.0
    reference = linear(torch.from_numpy(x))
    reflected = linear(torch.from_numpy(x * reflection[None, None, :, None]))
    torch.testing.assert_close(
        reflected,
        reference * torch.from_numpy(reflection[None, None, :, None]),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dp_linear.call(x), reference.detach().numpy(), atol=1e-12, rtol=1e-12
    )
    exported = ExptLinear.deserialize(linear.serialize()).to("cpu")
    exported_output = exported(torch.from_numpy(x))
    torch.testing.assert_close(exported_output, reference, atol=1e-12, rtol=1e-12)
    exported_output.square().sum().backward()
    assert all(
        weight.requires_grad and weight.grad is not None for weight in exported.weight_m
    )
    assert all(
        weight.shape[1] == linear.n_focus * (lmax - m) * linear.out_channels
        for m, weight in enumerate(linear.weight_m)
    )


@pytest.mark.parametrize("lmax", [1, 2])
@pytest.mark.parametrize("n_layers", [0, 2])
def test_cartesian_edge_preserves_inversion(lmax: int, n_layers: int) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.cartesian import (
        EdgeCartesianTensorProduct as DPEdge,
    )
    from deepmd.pt.model.descriptor.sezm_nn.cartesian import (
        EdgeCartesianTensorProduct as PTEdge,
    )

    config = {
        "lmax": lmax,
        "focus_dim": 2,
        "n_focus": 1,
        "n_layers": n_layers,
        "activation_function": "silu",
        "mlp_bias": False,
        "eps": 1e-7,
        "seed": 7,
        "trainable": True,
        "parity": True,
    }
    dp_edge = DPEdge(**config, precision="float64")
    edge = PTEdge(**config, dtype=torch.float64).to("cpu")
    dp_edge._load_variables(
        {key: value.detach().numpy() for key, value in edge.state_dict().items()}
    )
    rng = np.random.default_rng(19)
    x = rng.normal(size=(4, (lmax + 1) ** 2, 2))
    direction = rng.normal(size=(4, 3))
    radial = rng.normal(size=(4, lmax + 1, 2))
    signs = np.concatenate(
        [np.full(2 * degree + 1, (-1.0) ** degree) for degree in range(lmax + 1)]
    )
    reference = edge(
        torch.from_numpy(x), torch.from_numpy(direction), torch.from_numpy(radial)
    )
    reflected = edge(
        torch.from_numpy(x * signs[None, :, None]),
        torch.from_numpy(-direction),
        torch.from_numpy(radial),
    )
    torch.testing.assert_close(
        reflected,
        reference * torch.from_numpy(signs[None, :, None]),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        dp_edge.call(x, direction, radial),
        reference.detach().numpy(),
        atol=1e-12,
        rtol=1e-12,
    )
    assert "right_mult_proj" not in edge.state_dict()


@pytest.mark.parametrize("op_type", ["glu", "mlp", "branch"])
@pytest.mark.parametrize("mode", ["self", "cross"])
@pytest.mark.parametrize("sphere", ["s2", "s2_e3nn", "so3"])
def test_grid_nonlinearity_preserves_inversion(
    sphere: str, mode: str, op_type: str
) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import S2GridNet as DPS2
    from deepmd.dpmodel.descriptor.dpa4_nn.grid_net import SO3GridNet as DPSO3
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import S2GridNet as PTS2
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import SO3GridNet as PTSO3

    config = {
        "lmax": 2,
        "channels": 2,
        "n_focus": 1,
        "mode": mode,
        "op_type": op_type,
        "layout": "ndfc",
        "parity": True,
        "mlp_bias": True,
        "trainable": True,
        "seed": 21,
    }
    if sphere == "s2":
        config["grid_method"] = "lebedev"
    grid = (PTSO3 if sphere == "so3" else PTS2)(**config, dtype=torch.float64).to("cpu")
    dp_grid = (DPSO3 if sphere == "so3" else DPS2)(**config, precision="float64")
    dp_grid._load_variables(
        {key: value.detach().numpy() for key, value in grid.state_dict().items()}
    )
    rng = np.random.default_rng(23)
    channels = 2 * grid.n_frames * grid.channels if mode == "self" else grid.channels
    x = torch.from_numpy(rng.normal(size=(2, 9, 1, channels)))
    context = (
        torch.from_numpy(rng.normal(size=(2, 9, 1, channels)))
        if mode == "cross"
        else None
    )
    signs = torch.tensor(
        [1.0] + [-1.0] * 3 + [1.0] * 5,
        dtype=torch.float64,
        device="cpu",
    )[None, :, None, None]
    if sphere == "so3" and mode == "self":
        assert grid(x.float()).dtype == torch.float32
        assert dp_grid.call(x.float().numpy()).dtype == np.float32
    output = grid(x, context)
    reflected = grid(x * signs, None if context is None else context * signs)
    torch.testing.assert_close(reflected, output * signs, atol=1e-12, rtol=1e-12)
    # The e3nn projector and the independent NumPy construction retain their
    # tabulation tolerance; parity within either implementation is stricter.
    np.testing.assert_allclose(
        dp_grid.call(x.numpy(), None if context is None else context.numpy()),
        output.detach().numpy(),
        atol=1e-6 if sphere == "s2_e3nn" else 1e-12,
        rtol=1e-7 if sphere == "s2_e3nn" else 1e-12,
    )
    if sphere != "so3":
        assert not grid._project_inversion
    scalar = grid.forward_scalar(x, context)
    torch.testing.assert_close(scalar, output[:, :1], atol=1e-12, rtol=1e-12)


def test_e3nn_grid_requires_even_polar_resolution() -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
        S2GridProjector as DPProjector,
    )
    from deepmd.pt.model.descriptor.sezm_nn.projection import (
        S2GridProjector as PTProjector,
    )

    for projector, kwargs in (
        (DPProjector, {"precision": "float64"}),
        (PTProjector, {"dtype": torch.float64}),
    ):
        with pytest.raises(ValueError, match="even polar resolution"):
            projector(lmax=2, mmax=2, grid_resolution_list=[8, 7], **kwargs)
        assert projector(
            lmax=2, mmax=2, grid_resolution_list=[7, 8], **kwargs
        ).grid_resolution_list == [7, 8]


def test_plain_projector_load_preserves_archived_constants() -> None:
    """Ordinary checkpoints restore their stored geometric projection matrices."""
    from deepmd.dpmodel.descriptor.dpa4_nn.projection import (
        S2GridProjector,
    )
    from deepmd.pt_expt.common import (
        torch_module,
    )

    Projector = torch_module(S2GridProjector)
    source = Projector(lmax=2, precision="float64", parity=False)
    state = source.state_dict()
    assert "to_grid_mat" in state and "from_grid_mat" in state
    # A checkpoint owns these buffers even when a dependency constructs
    # slightly different quadrature constants in a later environment.
    state["to_grid_mat"] = state["to_grid_mat"] * 1.00001
    restored = Projector(lmax=2, precision="float64", parity=False)
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        restored.to_grid_mat, state["to_grid_mat"], atol=0, rtol=0
    )


@pytest.mark.parametrize("parity", [False, True])
def test_lora_so2_symmetry_serialization_and_merge(parity: bool) -> None:
    from deepmd.dpmodel.descriptor.dpa4_nn.lora import LoRASO2 as DPLoRA
    from deepmd.dpmodel.descriptor.dpa4_nn.so2 import SO2Linear as DPLinear
    from deepmd.pt.model.descriptor.sezm_nn.lora import LoRASO2 as PTLoRA
    from deepmd.pt.model.descriptor.sezm_nn.so2 import (
        SO2Linear,
    )

    base = SO2Linear(
        lmax=2,
        mmax=2,
        parity=parity,
        in_channels=3,
        out_channels=4,
        n_focus=2,
        dtype=torch.float64,
        seed=5,
        trainable=True,
    ).to("cpu")
    adapter = PTLoRA(base, rank=2).to("cpu")
    generator = torch.Generator(device="cpu").manual_seed(29)
    with torch.no_grad():
        adapter.B_m0.copy_(
            torch.randn(
                adapter.B_m0.shape,
                generator=generator,
                dtype=torch.float64,
                device="cpu",
            )
        )
        for weight in adapter.B_m:
            weight.copy_(
                torch.randn(
                    weight.shape,
                    generator=generator,
                    dtype=torch.float64,
                    device="cpu",
                )
            )
    x = torch.randn(
        2,
        5,
        base.reduced_dim,
        3,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    output = adapter(x)
    assert (output - base(x)).abs().max() > 1e-6
    if parity:
        signs = torch.ones(base.reduced_dim, dtype=x.dtype, device=x.device)
        signs[base.neg_indices] = -1.0
        torch.testing.assert_close(
            adapter(x * signs[None, None, :, None]),
            output * signs[None, None, :, None],
            atol=1e-12,
            rtol=1e-12,
        )
    # The portable record stores the adapter arrays explicitly.
    dp_adapter = DPLoRA(**base.serialize()["config"], lora_rank=2)
    payload = dp_adapter.serialize()
    payload["@variables"].update(
        {
            name: value.detach().numpy()
            for name, value in adapter.state_dict().items()
            if name != "lora_scaling"
        }
    )
    dp_adapter = DPLoRA.deserialize(payload)
    dp_restored = DPLoRA.deserialize(dp_adapter.serialize())
    merged = adapter.merge_into_base().to("cpu")
    restored = SO2Linear.deserialize(merged.serialize()).to("cpu")
    for module in (merged, restored):
        assert module.parity == parity
        torch.testing.assert_close(module(x), output, atol=1e-12, rtol=1e-12)
    for module in (
        dp_adapter,
        dp_restored,
        dp_adapter.merge_into_base(),
        DPLinear.deserialize(merged.serialize()),
    ):
        assert module.parity == parity
        np.testing.assert_allclose(
            module.call(x.numpy()), output.detach().numpy(), atol=1e-12, rtol=1e-12
        )
