# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent regression tests for the ASE calculator adapter."""

from unittest.mock import (
    Mock,
    patch,
)

import numpy as np
from ase import (
    Atoms,
)

from deepmd.calculator import (
    DP,
)


def test_stress_symmetrizes_flat_virial() -> None:
    """Convert the model virial to ASE's symmetric Voigt stress convention.

    Real DeePMD models normally produce an already-symmetric total virial, so
    the existing integration tests could not detect that transposing its flat
    nine-component representation was a no-op.  An intentionally asymmetric
    mock output makes each off-diagonal average independently observable.
    """
    model = Mock()
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
