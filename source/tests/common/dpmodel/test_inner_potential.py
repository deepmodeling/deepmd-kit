# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ``InnerPotential`` (analytical bridging term) unit tests.

The universal-ZBL formula and its truncation are reproduced in-test, and the
half-split per-edge scatter must sum back to the full analytic pair energy.
"""

import math

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.inner_potential import (
    ELEMENT_TO_Z,
    InnerPotential,
    InnerPotentialAtomicModel,
    nlh_provenance,
)
from deepmd.utils.element_radii import (
    covalent_radii_from_type_map,
)

_A_BOHR = 0.5291772109
_KE = 14.3996
_A_COEFF = (0.18175, 0.50986, 0.28022, 0.028171)
_B_COEFF = (3.1998, 0.94229, 0.4029, 0.20162)


def _zbl_series(zi: float, zj: float) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes in eV Å and decay rates in Å⁻¹ of the universal ZBL series."""
    a = 0.88534 * _A_BOHR / (zi**0.23 + zj**0.23)
    return _KE * zi * zj * np.asarray(_A_COEFF), np.asarray(_B_COEFF) / a


def _series_derivative(amp: np.ndarray, rate: np.ndarray, r: float, n: int) -> float:
    r"""The n-th radial derivative of :math:`\sum_k A_k e^{-c_k r}/r` in closed form."""
    factor = sum(
        math.factorial(n) / math.factorial(n - k) * rate ** (n - k) / r ** (k + 1)
        for k in range(n + 1)
    )
    return float((-1) ** n * np.sum(amp * np.exp(-rate * r) * factor))


def _truncated(amp: np.ndarray, rate: np.ndarray, r: float, end: float) -> float:
    """The series minus its cubic Taylor polynomial at ``end``; zero beyond."""
    if r >= end:
        return 0.0
    delta = r - end
    polynomial = sum(
        _series_derivative(amp, rate, end, k) * delta**k / math.factorial(k)
        for k in range(4)
    )
    return _series_derivative(amp, rate, r, 0) - polynomial


def _end_radius(pot: InnerPotential, ia: int, ib: int) -> float:
    """Separation at which the truncated term of one type pair ends, in Å."""
    radius = covalent_radii_from_type_map(pot.type_map)
    return pot.truncation * float(radius[ia] + radius[ib])


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
    end = _end_radius(pot, atypes[0], atypes[1])
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
    expected = _truncated(*_zbl_series(zi, zj), r, end)
    assert expected > 0.0
    np.testing.assert_allclose(float(np.sum(out)), expected, rtol=1e-12)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("mode", ["zbl", "nlh"])
def test_energy_and_force_at_the_cutoff(mode: str, precision: str) -> None:
    """The leading energy and force at the cutoff are quartic and cubic."""
    import torch

    pot = InnerPotential(["O"], mode=mode, precision=precision)
    assert pot.pair_table.dtype == np.dtype(precision)
    end = _end_radius(pot, 0, 0)
    evaluation_end = float(np.asarray(end, dtype=precision))
    amplitude, rate = _pair_series(pot, 0, 0)
    fourth = _series_derivative(amplitude.astype(float), rate.astype(float), end, 4)
    for fraction in (1.0 - 2.0**-15, 1.0 - 2.0**-28, 1.0, 1.01):
        r = evaluation_end * fraction
        edges, index, mask = _two_atom_inputs(r)
        vec = torch.tensor(edges, dtype=torch.float64, device="cpu", requires_grad=True)
        energy = pot.call(
            vec,
            torch.tensor(index, device="cpu"),
            torch.zeros(2, dtype=torch.long, device="cpu"),
            torch.tensor(mask, device="cpu"),
            2,
        ).sum()
        grad = torch.autograd.grad(energy, vec)[0]
        gap = max(evaluation_end - float(np.asarray(r, dtype=precision)), 0.0)
        assert energy.item() == pytest.approx(fourth * gap**4 / 24, rel=5e-4, abs=0)
        assert -2 * grad[0, 0].item() == pytest.approx(
            fourth * gap**3 / 6, rel=5e-4, abs=0
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


def _pair_series(
    pot: InnerPotential, ia: int, ib: int
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes in eV Å and decay rates in Å⁻¹ of one ordered type pair."""
    stride = pot.ntypes + 1
    row = np.asarray(pot.pair_table)[ia * stride + ib, :8]
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
    amp, rate = _pair_series(pot, 0, 1)
    expected = _truncated(amp, rate, r, _end_radius(pot, 0, 1))
    np.testing.assert_allclose(float(np.sum(out)), expected, rtol=1e-12)


def test_nlh_differs_from_zbl():
    """The two modes are distinct potentials, not the same table twice."""
    r = 0.8
    values = [
        _series_derivative(*_pair_series(InnerPotential(["O"], mode=mode), 0, 0), r, 0)
        for mode in ("zbl", "nlh")
    ]
    assert values[1] > 0.0
    assert abs(values[1] / values[0] - 1.0) > 0.05


def test_nlh_screening_reaches_the_coulomb_limit():
    """Every amplitude is non-negative and the amplitudes sum to k_e Z_a Z_b."""
    pot = InnerPotential(type_map=_ALL_ELEMENTS, mode="nlh")
    charge = np.asarray([ELEMENT_TO_Z[symbol] for symbol in _ALL_ELEMENTS], dtype=float)
    ntypes = len(_ALL_ELEMENTS)
    table = np.asarray(pot.pair_table)[:, :8].reshape(ntypes + 1, ntypes + 1, 8)
    amp = table[:ntypes, :ntypes, :4]
    assert amp.min() >= 0.0
    reference = _KE * charge[:, None] * charge[None, :]
    np.testing.assert_allclose(amp.sum(-1), reference, rtol=1e-12)


def test_nlh_is_positive_and_strictly_decreasing():
    """Non-negative amplitudes make V positive and falling at every radius."""
    pot = InnerPotential(type_map=_ALL_ELEMENTS, mode="nlh")
    ntypes = len(_ALL_ELEMENTS)
    table = np.asarray(pot.pair_table)[:, :8].reshape(ntypes + 1, ntypes + 1, 8)
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
    table = np.asarray(pot.pair_table)[:, :8].reshape(ntypes + 1, ntypes + 1, 8)
    amp, rate = table[:ntypes, :ntypes, :4], table[:ntypes, :ntypes, 4:]
    for radius, bound in ((5.0, 1e-3), (6.0, 1e-4)):
        v = (amp * np.exp(-rate * radius)).sum(-1) / radius
        assert v.max() <= bound * (1.0 + 1e-6)


def test_nlh_table_is_symmetric_and_padded():
    """Swapping the two types leaves the row unchanged; padding rows vanish."""
    pot = InnerPotential(type_map=["H", "O", "U"], mode="nlh")
    table = np.asarray(pot.pair_table)[:, :8].reshape(4, 4, 8)
    np.testing.assert_array_equal(table, np.swapaxes(table, 0, 1))
    np.testing.assert_array_equal(table[3], 0.0)
    np.testing.assert_array_equal(table[:, 3], 0.0)


def test_nlh_change_type_map_rebuilds_from_symbols():
    """Reordering, adding and dropping elements all rebuild the right rows."""
    pot = InnerPotential(type_map=["O", "H"], mode="nlh")
    original = np.asarray(_pair_series(pot, 0, 1)[0]).copy()
    pot.change_type_map(["H", "O", "Fe"])
    assert pot.type_map == ["H", "O", "Fe"]
    assert pot.ntypes == 3
    assert np.asarray(pot.pair_table).shape == (16, 16)
    np.testing.assert_allclose(_pair_series(pot, 1, 0)[0], original, rtol=1e-12)
    fresh = InnerPotential(type_map=["H", "O", "Fe"], mode="nlh")
    np.testing.assert_allclose(
        np.asarray(pot.pair_table), np.asarray(fresh.pair_table), rtol=1e-12
    )


def test_pair_table_has_zero_padding_rows() -> None:
    pot = InnerPotential(["O", "H"])
    table = pot.pair_table.reshape(3, 3, -1)
    np.testing.assert_array_equal(table[-1], 0.0)
    np.testing.assert_array_equal(table[:, -1], 0.0)


def test_nlh_provenance_records_the_source_and_the_modification():
    """The attribution travels inside the data file."""
    meta = nlh_provenance()
    assert "10.5281/zenodo.14172633" in meta["source_data"]["doi"]
    assert "CC BY 4.0" in meta["source_data"]["licence"]
    assert "10.1103/PhysRevA.111.032818" in meta["source_paper"]["doi"]
    assert "not the coefficients published" in meta["modification"]
    assert "MP2" in meta["rows"]["Z1 = 5, Z2 = 10"]
    assert "ZBL" in meta["rows"]["Z1 > 92 or Z2 > 92"]


# ---------------------------------------------------------------------------
# Truncation: the cubic Taylor remainder of the series at its end radius
# ---------------------------------------------------------------------------
def _pair_derivatives(pot: InnerPotential, r: float, atypes: list[int]) -> list[float]:
    """Return the pair energy and its first three radial derivatives."""
    import torch

    x = torch.tensor(r, dtype=torch.float64, device="cpu", requires_grad=True)
    zero = x * 0.0
    edge_vec = torch.stack(
        [torch.stack([x, zero, zero]), torch.stack([-x, zero, zero])]
    )
    out = pot.call(
        edge_vec,
        torch.tensor([[0, 1], [1, 0]], device="cpu"),
        torch.tensor(atypes, dtype=torch.int64, device="cpu"),
        torch.tensor([True, True], device="cpu"),
        2,
    ).sum()
    values = [out]
    for _ in range(3):
        values.append(torch.autograd.grad(values[-1], x, create_graph=True)[0])
    return [float(v.detach()) for v in values]


def _series_derivatives(pot: InnerPotential, r: float, ia: int, ib: int) -> list[float]:
    """Return the untruncated series of one type pair and its first three derivatives."""
    amp, rate = _pair_series(pot, ia, ib)
    return [_series_derivative(amp, rate, r, n) for n in range(4)]


def test_truncated_term_is_the_taylor_remainder():
    """Below the end the term is V minus its cubic Taylor polynomial; zero beyond."""
    pot = InnerPotential(type_map=["C", "O"], mode="zbl", truncation=1.2)
    end = _end_radius(pot, 0, 1)
    amp, rate = _pair_series(pot, 0, 1)
    # The remainder cancels the series to many digits near the end, so the
    # tolerance is relative to the series value there.
    atol = 1e-12 * _series_derivative(amp, rate, end, 0)
    for fraction in (0.2, 0.5, 0.9, 0.999):
        r = fraction * end
        np.testing.assert_allclose(
            _pair_derivatives(pot, r, [0, 1])[0],
            _truncated(amp, rate, r, end),
            rtol=1e-10,
            atol=atol,
        )
    for fraction in (1.0, 1.01, 1.5):
        assert _pair_derivatives(pot, fraction * end, [0, 1])[0] == 0.0


@pytest.mark.parametrize("mode", ["zbl", "nlh"])
def test_truncated_term_is_c3_at_its_end(mode):
    """Value and the first three derivatives vanish as the end is approached."""
    pot = InnerPotential(type_map=["Fe", "O"], mode=mode)
    end = _end_radius(pot, 0, 1)
    inside = _pair_derivatives(pot, end * (1.0 - 1e-7), [0, 1])
    scale = _series_derivatives(pot, end, 0, 1)
    for order, value in enumerate(inside):
        assert abs(value) <= 1e-5 * abs(scale[order]) + 1e-12, (order, value)


@pytest.mark.parametrize("mode", ["zbl", "nlh"])
def test_truncated_term_is_positive_decreasing_and_convex(mode):
    """Every pair of a broad type map keeps the signs Taylor's remainder implies."""
    type_map = ["H", "Li", "C", "O", "F", "Na", "Si", "Fe", "Cu", "Cs", "W", "U"]
    pot = InnerPotential(type_map=type_map, mode=mode)
    for a in range(len(type_map)):
        for b in range(a, len(type_map)):
            end = _end_radius(pot, a, b)
            for fraction in (0.1, 0.3, 0.6, 0.9, 0.99):
                value, first, second, _ = _pair_derivatives(pot, fraction * end, [a, b])
                assert value > 0.0 and first < 0.0 and second > 0.0, (
                    type_map[a],
                    type_map[b],
                    fraction,
                )


def test_padded_zero_length_edge_has_a_finite_hessian():
    """A masked edge of zero length leaves every second derivative finite."""
    import torch

    pot = InnerPotential(type_map=["O", "H"])
    edge_vec_np, edge_index, edge_mask = _two_atom_inputs(0.8)
    edge_vec_np = np.concatenate([edge_vec_np, np.zeros((1, 3))])
    edge_index = np.concatenate([edge_index, np.array([[0], [0]])], axis=1)
    edge_mask = np.concatenate([edge_mask, [False]])
    ev = torch.tensor(
        edge_vec_np, dtype=torch.float64, device="cpu", requires_grad=True
    )
    out = pot.call(
        ev,
        torch.tensor(edge_index, device="cpu"),
        torch.tensor(np.array([0, 1], dtype=np.int64), device="cpu"),
        torch.tensor(edge_mask, device="cpu"),
        2,
    )
    grad = torch.autograd.grad(out.sum(), ev, create_graph=True)[0]
    hessian_row = torch.autograd.grad(grad.sum(), ev)[0]
    assert torch.isfinite(grad).all()
    assert torch.isfinite(hessian_row).all()
    assert hessian_row[2].abs().max().item() == 0.0


@pytest.mark.parametrize("distance", [0.0, 1e-9])
def test_contact_curvature_in_single_precision(distance: float) -> None:
    """At nuclear contact the curvature is set by the regularized Coulomb term."""
    import torch

    pot = InnerPotential(["U"], precision="float32")
    x = torch.tensor(
        [distance, 0.0, 0.0], device="cpu", dtype=torch.float32, requires_grad=True
    )
    _, index, mask = _two_atom_inputs(distance)
    energy = pot.call(
        torch.stack((x, -x)),
        torch.tensor(index, device="cpu"),
        torch.zeros(2, dtype=torch.long, device="cpu"),
        torch.tensor(mask, device="cpu"),
        2,
    ).sum()
    gradient = torch.autograd.grad(energy, x, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient[0], x)[0][0]
    eps = 1e-10
    expected = (
        _KE
        * 92**2
        * sum(_A_COEFF)
        * (2 * distance**2 - eps**2)
        / (distance**2 + eps**2) ** 2.5
    )
    assert curvature.item() == pytest.approx(expected, rel=3e-6)


@pytest.mark.parametrize("truncation", [0.0, -1.0, float("nan"), float("inf")])
def test_truncation_must_be_finite_and_positive(truncation):
    with pytest.raises(ValueError, match="finite and positive"):
        InnerPotential(type_map=["O"], truncation=truncation)


def test_change_type_map_rebuilds_the_truncation():
    pot = InnerPotential(type_map=["O", "H"], mode="nlh", truncation=1.2)
    pot.change_type_map(["H", "O", "Fe"])
    fresh = InnerPotential(type_map=["H", "O", "Fe"], mode="nlh", truncation=1.2)
    np.testing.assert_array_equal(pot.pair_table, fresh.pair_table)


def test_rejected_type_map_change_leaves_the_model_unchanged():
    """A map whose largest pair would end beyond the cutoff is refused before any change."""
    model = InnerPotentialAtomicModel(
        ["O", "H"], mode="zbl", rcut=4.0, sel=8, truncation=1.2
    )
    table = np.asarray(model.potential.pair_table).copy()
    with pytest.raises(ValueError, match="cutoff"):
        model.change_type_map(["O", "H", "Cs"])
    assert model.get_type_map() == ["O", "H"]
    assert model.potential.type_map == ["O", "H"]
    np.testing.assert_array_equal(np.asarray(model.potential.pair_table), table)
