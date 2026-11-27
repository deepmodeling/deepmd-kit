# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent regression tests for the ASE calculator adapter."""

from unittest.mock import (
    Mock,
    patch,
)

import numpy as np
import pytest
from ase import (
    Atoms,
)
from ase.constraints import (
    FixAtoms,
)
from ase.optimize import (
    BFGS,
)

from deepmd.calculator import (
    DP,
    SpinFilter,
)


def test_stress_symmetrizes_flat_virial() -> None:
    """Convert the model virial to ASE's symmetric Voigt stress convention.

    Real DeePMD models normally produce an already-symmetric total virial, so
    the existing integration tests could not detect that transposing its flat
    nine-component representation was a no-op.  An intentionally asymmetric
    mock output makes each off-diagonal average independently observable.
    """
    model = Mock()
    model.has_spin = False
    model.get_type_map.return_value = ["H"]
    model.get_ntypes.return_value = 1
    virial = np.arange(1.0, 10.0).reshape(1, 9)
    model.eval.return_value = (
        np.array([[0.0]]),
        np.zeros((1, 1, 3)),
        virial,
    )

    with patch("deepmd.calculator.DeepPot", return_value=model):
        calculator = DP("unused-model")

    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=np.eye(3) * 2.0,
        pbc=True,
        calculator=calculator,
    )

    np.testing.assert_allclose(
        atoms.get_stress(voigt=True),
        -np.array([1.0, 5.0, 9.0, 7.0, 5.0, 3.0]) / atoms.get_volume(),
    )
    # Symmetrizing stress must not discard diagnostic information from the
    # model: callers requesting the virial still receive the original tensor.
    np.testing.assert_array_equal(calculator.results["virial"], virial.reshape(3, 3))


@pytest.mark.parametrize(
    ("relax_positions", "fix_fe"), [(False, False), (True, False), (True, True)]
)
def test_relax_magnetic_atoms(relax_positions: bool, fix_fe: bool) -> None:
    atoms = Atoms("FeO", positions=[[0.2, 0, 0], [0.5, 0, 0]])
    if fix_fe:
        atoms.set_constraint(FixAtoms(indices=[0]))
    initial_positions = atoms.positions.copy()
    initial_moments = np.array([[0.4, -0.2, 0.1], [0.3, 0.5, 0.2]])
    atoms.set_initial_magnetic_moments(initial_moments)

    def evaluate(*, coords, spin, **kwargs):
        positions = coords.reshape(1, 2, 3)
        energy = (np.sum(positions**2) + np.sum(spin[:, 0] ** 2)) / 2
        mask = np.array([[[True], [False]]])
        return np.array([[energy]]), -positions, np.zeros((1, 9)), -spin * mask, mask

    with patch("deepmd.calculator.DeepPot") as factory:
        factory.return_value.has_spin = True
        factory.return_value.eval.side_effect = evaluate
        atoms.calc = DP("spin.pt2", type_dict={"Fe": 0, "O": 1})
        system = SpinFilter(atoms, relax_positions=relax_positions)
        assert BFGS(system, logfile=None).run(fmax=1e-6, steps=100)

    moments = atoms.get_initial_magnetic_moments()
    np.testing.assert_allclose(moments[0], 0, atol=1e-6)
    np.testing.assert_array_equal(moments[1], initial_moments[1])
    if fix_fe:
        np.testing.assert_array_equal(atoms.positions[0], initial_positions[0])
        np.testing.assert_allclose(atoms.positions[1], 0, atol=1e-6)
    else:
        np.testing.assert_allclose(
            atoms.positions, 0 if relax_positions else initial_positions, atol=1e-6
        )
