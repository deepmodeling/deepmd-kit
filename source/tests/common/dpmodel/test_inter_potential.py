# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ``InterPotential`` (analytical ZBL bridging term) unit tests.

Ports the pt reference values from
``source/tests/pt/model/test_sezm_model.py::TestInterPotential``: the exact
universal-ZBL formula is reproduced in-test and the half-split per-edge
scatter must sum back to the full analytic pair energy.
"""

import math

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.inter_potential import (
    InterPotential,
)

_A_BOHR = 0.5291772109
_KE = 14.3996
_A_COEFF = (0.18175, 0.50986, 0.28022, 0.028171)
_B_COEFF = (3.1998, 0.94229, 0.4029, 0.20162)


def _analytic_zbl(r: float, zi: float, zj: float) -> float:
    a = 0.88534 * _A_BOHR / (zi**0.23 + zj**0.23)
    x = r / a
    phi = sum(
        a_k * math.exp(-b_k * x) for a_k, b_k in zip(_A_COEFF, _B_COEFF, strict=True)
    )
    return _KE * zi * zj / r * phi


def _two_atom_inputs(r: float):
    """Two atoms at distance r with BOTH directed edges (symmetric list)."""
    edge_vec = np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=np.float64)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)  # src, dst
    edge_mask = np.array([True, True])
    return edge_vec, edge_index, edge_mask


@pytest.mark.parametrize(
    ("type_map", "atypes", "zi", "zj"),
    [
        (["O"], [0, 0], 8.0, 8.0),  # O-O pair
        (["O", "H"], [0, 1], 8.0, 1.0),  # O-H pair
    ],
)
def test_zbl_known_value(type_map, atypes, zi, zj):
    r = 0.8
    pot = InterPotential(type_map=type_map)
    edge_vec, edge_index, edge_mask = _two_atom_inputs(r)
    out = pot.call(
        edge_vec,
        edge_index,
        np.asarray(atypes, dtype=np.int64),
        edge_mask,
        n_node=2,
    )
    assert out.shape == (1, 2, 1)
    # Half per directed edge -> the total is the full analytic pair energy.
    np.testing.assert_allclose(
        float(np.sum(out)), _analytic_zbl(r, zi, zj), rtol=0, atol=1e-5
    )


def test_virtual_types_masked():
    # real_type_count=1: type 1 is a virtual/placeholder type; its edges
    # contribute zero, and only real-real edges survive.
    pot = InterPotential(type_map=["O"])
    r = 0.9
    edge_vec = np.array(
        [[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]], dtype=np.float64
    )
    edge_index = np.array([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=np.int64)
    edge_mask = np.ones(4, dtype=bool)
    atypes = np.array([0, 0, 1], dtype=np.int64)  # atom 2 is virtual
    out = pot.call(edge_vec, edge_index, atypes, edge_mask, 3, real_type_count=1)
    np.testing.assert_allclose(
        float(np.sum(out)), _analytic_zbl(r, 8.0, 8.0), atol=1e-5
    )
    assert float(out[0, 2, 0]) == 0.0


def test_edge_mask_zeroes_edges():
    pot = InterPotential(type_map=["O"])
    edge_vec, edge_index, _ = _two_atom_inputs(0.8)
    out = pot.call(
        edge_vec,
        edge_index,
        np.zeros(2, dtype=np.int64),
        np.array([False, False]),
        n_node=2,
    )
    np.testing.assert_array_equal(np.asarray(out), 0.0)


def test_unknown_element_raises():
    with pytest.raises(ValueError, match="Unknown element symbol"):
        InterPotential(type_map=["O", "Xx"])


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown InterPotential mode"):
        InterPotential(type_map=["O"], mode="lj")


def test_torch_namespace_smoke_and_gradient():
    """Torch inputs match numpy at 1e-12 and edge_vec gradients exist."""
    import torch

    pot = InterPotential(type_map=["O", "H"])
    edge_vec_np, edge_index, edge_mask = _two_atom_inputs(0.8)
    atypes = np.array([0, 1], dtype=np.int64)
    ref = np.asarray(pot.call(edge_vec_np, edge_index, atypes, edge_mask, 2))

    ev = torch.tensor(edge_vec_np, dtype=torch.float64, requires_grad=True)
    out = pot.call(
        ev,
        torch.tensor(edge_index),
        torch.tensor(atypes),
        torch.tensor(edge_mask),
        2,
    )
    np.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-12)
    grad = torch.autograd.grad(out.sum(), ev)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().max().item() > 0.0
