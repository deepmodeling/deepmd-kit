# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dipole and polarizability models on the graph-only DPA4C descriptor.

DPA4C has no dense lower, so a non-energy model built on it must take the
graph route by default, and its degree-one moments must reach the tensorial
fitting as ``gr``.
"""

import numpy as np
import pytest
import torch

from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.model.graph_lower import (
    model_requires_graph_lower,
    model_uses_graph_lower,
)
from deepmd.pt_expt.utils import (
    env,
)

ROTATION = torch.tensor(
    [
        [-2.0 / 3.0, 2.0 / 15.0, 11.0 / 15.0],
        [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
        [1.0 / 3.0, 14.0 / 15.0, 2.0 / 15.0],
    ],
    dtype=torch.float64,
)


def _model(fitting_type: str, **fitting: object) -> torch.nn.Module:
    return get_model(
        {
            "type_map": ["O", "H"],
            "atom_exclude_types": [1],
            "descriptor": {
                "type": "dpa4c",
                "rcut": 3.0,
                "channels": 16,
                "lmax": 2,
                "n_radial": 4,
                "precision": "float64",
                "seed": 17,
                "output_rot_mat": True,
            },
            "fitting_net": {
                "type": fitting_type,
                "neuron": [16, 16],
                "precision": "float64",
                "seed": 19,
                **fitting,
            },
        }
    ).to(env.DEVICE)


def _frame(nframes: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small periodic water-like frames: three O and six H."""
    rng = np.random.default_rng(7)
    natoms = 9
    cell = 4.0 * np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    coord = rng.uniform(0.0, 4.0, (nframes, natoms, 3))
    atype = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    return (
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(np.tile(atype, (nframes, 1)), device=env.DEVICE),
        torch.tensor(
            np.tile(cell.reshape(1, 9), (nframes, 1)),
            dtype=torch.float64,
            device=env.DEVICE,
        ),
    )


def _rotate(
    coord: torch.Tensor, box: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = ROTATION.to(env.DEVICE)
    rotated_box = (box.reshape(-1, 3, 3) @ rotation.T).reshape(-1, 9)
    return coord @ rotation.T, rotated_box


def test_graph_only_routing_for_tensor_models() -> None:
    model = _model("dipole")
    assert model.get_descriptor().get_dim_emb() == 4
    assert model_requires_graph_lower(model)
    # The compiled graph trace stays energy-only.
    assert not model_uses_graph_lower(model)
    assert model._resolve_graph_method(None) is not None
    assert model._resolve_graph_method("legacy") is None


def test_dipole_forward_is_equivariant() -> None:
    model = _model("dipole")
    coord, atype, box = _frame()
    ret = model(coord, atype, box=box)
    assert ret["dipole"].shape == (2, 9, 3)
    assert ret["global_dipole"].shape == (2, 3)
    # Excluded hydrogen carries no atomic dipole.
    assert torch.all(ret["dipole"][:, 3:] == 0)
    assert ret["dipole"][:, :3].abs().sum() > 0

    rotated_coord, rotated_box = _rotate(coord, box)
    rotated = model(rotated_coord, atype, box=rotated_box)
    rotation = ROTATION.to(env.DEVICE)
    torch.testing.assert_close(
        rotated["dipole"], ret["dipole"] @ rotation.T, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        rotated["global_dipole"],
        ret["global_dipole"] @ rotation.T,
        atol=1e-10,
        rtol=1e-10,
    )


@pytest.mark.parametrize("fit_diag", [False, True])
def test_polar_forward_is_equivariant(fit_diag: bool) -> None:
    model = _model("polar", fit_diag=fit_diag)
    coord, atype, box = _frame()
    ret = model(coord, atype, box=box)
    assert ret["polar"].reshape(2, 9, 9).shape == (2, 9, 9)
    assert ret["global_polar"].numel() == 2 * 9

    rotated_coord, rotated_box = _rotate(coord, box)
    rotated = model(rotated_coord, atype, box=rotated_box)
    rotation = ROTATION.to(env.DEVICE)
    polar = ret["global_polar"].reshape(2, 3, 3)
    torch.testing.assert_close(
        rotated["global_polar"].reshape(2, 3, 3),
        rotation @ polar @ rotation.T,
        atol=1e-10,
        rtol=1e-10,
    )
    # The polarizability is symmetric by construction.
    torch.testing.assert_close(polar, polar.transpose(-1, -2))


def test_dipole_parameters_receive_gradients() -> None:
    model = _model("dipole")
    coord, atype, box = _frame()
    model(coord, atype, box=box)["global_dipole"].square().sum().backward()
    descriptor = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if "descriptor" in name and parameter.requires_grad
    ]
    assert descriptor
    assert any(grad is not None and grad.abs().sum() > 0 for grad in descriptor)
