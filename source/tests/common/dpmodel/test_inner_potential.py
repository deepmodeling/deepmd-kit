# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ``InnerPotential`` (analytical bridging term) unit tests.

Ports the pt reference values from
``source/tests/pt/model/test_sezm_model.py::TestInnerPotential``: the exact
universal-ZBL formula is reproduced in-test and the half-split per-edge
scatter must sum back to the full analytic pair energy.
"""

import math

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.inner_potential import (
    ELEMENT_TO_Z,
    InnerPotential,
    nlh_provenance,
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
    pot = InnerPotential(type_map=type_map)
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


def test_edge_mask_zeroes_edges():
    pot = InnerPotential(type_map=["O"])
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
        InnerPotential(type_map=["O", "Xx"])


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown InnerPotential mode"):
        InnerPotential(type_map=["O"], mode="lj")


def test_torch_namespace_smoke_and_gradient():
    """Torch inputs match numpy at 1e-12 and edge_vec gradients exist."""
    import torch

    pot = InnerPotential(type_map=["O", "H"])
    edge_vec_np, edge_index, edge_mask = _two_atom_inputs(0.8)
    atypes = np.array([0, 1], dtype=np.int64)
    ref = np.asarray(pot.call(edge_vec_np, edge_index, atypes, edge_mask, 2))

    # The device is explicit: the pt test package, which other common tests
    # import, points the default device at a nonexistent one on purpose.
    ev = torch.tensor(
        edge_vec_np, dtype=torch.float64, device="cpu", requires_grad=True
    )
    out = pot.call(
        ev,
        torch.tensor(edge_index, device="cpu"),
        torch.tensor(atypes, device="cpu"),
        torch.tensor(edge_mask, device="cpu"),
        2,
    )
    np.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-12)
    grad = torch.autograd.grad(out.sum(), ev)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# NLH mode: the bundled coefficient table and the conditions it guarantees
# ---------------------------------------------------------------------------
_ALL_ELEMENTS = list(ELEMENT_TO_Z)


def _nlh_series(pot: InnerPotential, ia: int, ib: int) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes in eV Å and decay rates in Å⁻¹ of one ordered type pair."""
    stride = pot.ntypes + 1
    row = np.asarray(pot.series_table)[ia * stride + ib]
    return row[:4], row[4:]


def test_nlh_mode_is_accepted_and_normalized():
    pot = InnerPotential(type_map=["O", "H"], mode="nlh")
    assert pot.mode == "NLH"


def test_nlh_matches_its_table():
    """The scattered energy is the series the table holds for that pair."""
    r = 0.8
    pot = InnerPotential(type_map=["O", "H"], mode="nlh")
    edge_vec, edge_index, edge_mask = _two_atom_inputs(r)
    out = pot.call(edge_vec, edge_index, np.array([0, 1], dtype=np.int64), edge_mask, 2)
    amp, rate = _nlh_series(pot, 0, 1)
    expected = float(np.sum(amp * np.exp(-rate * r)) / r)
    np.testing.assert_allclose(float(np.sum(out)), expected, rtol=1e-12)


def test_nlh_differs_from_zbl():
    """The two modes are distinct potentials, not the same table twice."""
    r = 0.8
    edge_vec, edge_index, edge_mask = _two_atom_inputs(r)
    atypes = np.zeros(2, dtype=np.int64)
    values = [
        float(
            np.sum(
                InnerPotential(type_map=["O"], mode=mode).call(
                    edge_vec, edge_index, atypes, edge_mask, 2
                )
            )
        )
        for mode in ("zbl", "nlh")
    ]
    assert values[1] > 0.0
    assert abs(values[1] / values[0] - 1.0) > 0.05


def test_nlh_screening_reaches_the_coulomb_limit():
    """Every amplitude is non-negative and the amplitudes sum to k_e Z_a Z_b."""
    pot = InnerPotential(type_map=_ALL_ELEMENTS, mode="nlh")
    charge = np.asarray([ELEMENT_TO_Z[symbol] for symbol in _ALL_ELEMENTS], dtype=float)
    ntypes = len(_ALL_ELEMENTS)
    table = np.asarray(pot.series_table).reshape(ntypes + 1, ntypes + 1, 8)
    amp = table[:ntypes, :ntypes, :4]
    assert amp.min() >= 0.0
    reference = _KE * charge[:, None] * charge[None, :]
    np.testing.assert_allclose(amp.sum(-1), reference, rtol=1e-12)


def test_nlh_is_positive_and_strictly_decreasing():
    """Non-negative amplitudes make V positive and falling at every radius."""
    pot = InnerPotential(type_map=_ALL_ELEMENTS, mode="nlh")
    ntypes = len(_ALL_ELEMENTS)
    table = np.asarray(pot.series_table).reshape(ntypes + 1, ntypes + 1, 8)
    radii = np.concatenate([np.geomspace(1e-3, 2.0, 200), np.linspace(2.02, 20.0, 200)])
    amp = table[:ntypes, :ntypes, None, :4]
    rate = table[:ntypes, :ntypes, None, 4:]
    v = (amp * np.exp(-rate * radii[:, None])).sum(-1) / radii
    assert v.min() > 0.0
    assert np.diff(v, axis=-1).max() < 0.0


def test_nlh_tail_bound_holds_for_every_pair():
    """The bound the fit imposes: 1 meV at 5 Å and 0.1 meV at 6 Å."""
    pot = InnerPotential(type_map=_ALL_ELEMENTS, mode="nlh")
    ntypes = len(_ALL_ELEMENTS)
    table = np.asarray(pot.series_table).reshape(ntypes + 1, ntypes + 1, 8)
    amp, rate = table[:ntypes, :ntypes, :4], table[:ntypes, :ntypes, 4:]
    for radius, bound in ((5.0, 1e-3), (6.0, 1e-4)):
        v = (amp * np.exp(-rate * radius)).sum(-1) / radius
        assert v.max() <= bound * (1.0 + 1e-6)


def test_nlh_table_is_symmetric_and_padded():
    """Swapping the two types leaves the row unchanged; padding rows vanish."""
    pot = InnerPotential(type_map=["H", "O", "U"], mode="nlh")
    table = np.asarray(pot.series_table).reshape(4, 4, 8)
    np.testing.assert_array_equal(table, np.swapaxes(table, 0, 1))
    np.testing.assert_array_equal(table[3], 0.0)
    np.testing.assert_array_equal(table[:, 3], 0.0)


def test_nlh_change_type_map_rebuilds_from_symbols():
    """Reordering, adding and dropping elements all rebuild the right rows."""
    pot = InnerPotential(type_map=["O", "H"], mode="nlh")
    original = np.asarray(_nlh_series(pot, 0, 1)[0]).copy()
    pot.change_type_map(["H", "O", "Fe"])
    assert pot.type_map == ["H", "O", "Fe"]
    assert pot.ntypes == 3
    assert np.asarray(pot.series_table).shape == (16, 8)
    np.testing.assert_allclose(_nlh_series(pot, 1, 0)[0], original, rtol=1e-12)
    fresh = InnerPotential(type_map=["H", "O", "Fe"], mode="nlh")
    np.testing.assert_allclose(
        np.asarray(pot.series_table), np.asarray(fresh.series_table), rtol=1e-12
    )


def test_pair_table_is_the_float32_view_of_the_series_table():
    """The fused kernels and this route read the same constants."""
    for mode in ("zbl", "nlh"):
        pot = InnerPotential(type_map=["O", "H"], mode=mode)
        assert np.asarray(pot.pair_table).dtype == np.float32
        np.testing.assert_allclose(
            np.asarray(pot.pair_table).astype(np.float64),
            np.asarray(pot.series_table),
            rtol=1e-6,
        )


def test_nlh_provenance_records_the_source_and_the_modification():
    """The attribution travels inside the data file."""
    meta = nlh_provenance()
    assert "10.5281/zenodo.14172633" in meta["source_data"]["doi"]
    assert "CC BY 4.0" in meta["source_data"]["licence"]
    assert "10.1103/PhysRevA.111.032818" in meta["source_paper"]["doi"]
    assert "not the coefficients published" in meta["modification"]
    assert "MP2" in meta["rows"]["Z1 = 5, Z2 = 10"]
    assert "ZBL" in meta["rows"]["Z1 > 92 or Z2 > 92"]
