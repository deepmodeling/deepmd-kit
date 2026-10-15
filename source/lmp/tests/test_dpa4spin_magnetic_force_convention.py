# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit checks for the ``dpa4spin`` / ``dpa4spin/kk`` magnetic-force conventions.

The LAMMPS adapter stores the model energy gradient with respect to the
magnetic moment in ``atom->fm`` under one of two explicit conventions,
selected by ``pair_style ... magnetic_force precession|cartesian`` and
reported by ``Pair::extract("fm_cartesian_force", dim)``:

* precession (default, extract value 0): ``fm = scale * force_cvt * |m| / hbar
  * force_mag``, matching the stock LAMMPS SPIN integrator;
* cartesian (extract value 1): ``fm = scale * ener_cvt * force_mag``, the raw
  ``-dE/dS`` gradient with only the energy-per-model-spin unit conversion,
  including at zero moment.

These tests encode that conversion contract without a LAMMPS build. Full
adapter coverage (``dpa4spin`` / ``dpa4spin/kk``, single-rank and MPI,
built-in and plugin, finite-difference comparison of Cartesian ``fm`` to
energy) belongs in CI with a Kokkos-enabled LAMMPS and a native-spin
archive; this module does not substitute for that.
"""

from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)

import numpy as np
import pytest

# Metal-units reduced Planck constant used by pair_dpa4spin{,_kokkos}.cpp.
HBAR_METAL = 6.5821191e-04


def store_magnetic_force(
    force_mag: np.ndarray,
    spin_norm: np.ndarray,
    *,
    cartesian: bool,
    scale: float = 1.0,
    ener_unit_cvt_factor: float = 1.0,
    force_unit_cvt_factor: float = 1.0,
) -> np.ndarray:
    """Apply the adapter's magnetic-force conversion (metal path by default).

    Parameters
    ----------
    force_mag : np.ndarray
        Model energy gradient with respect to the moment, shape (natoms, 3).
    spin_norm : np.ndarray
        Per-atom moment magnitude ``sp[:, 3]``, shape (natoms,).
    cartesian : bool
        True for ``magnetic_force cartesian``, False for precession.
    scale : float
        Pair ``scale[1][1]`` prefactor.
    ener_unit_cvt_factor, force_unit_cvt_factor : float
        LAMMPS unit conversions relative to the model eV / Angstrom system.

    Returns
    -------
    np.ndarray
        Values that the adapter accumulates into ``atom->fm``, shape (natoms, 3).
    """
    force_mag = np.asarray(force_mag, dtype=np.float64)
    spin_norm = np.asarray(spin_norm, dtype=np.float64)
    if cartesian:
        return scale * ener_unit_cvt_factor * force_mag
    return (scale * force_unit_cvt_factor / HBAR_METAL) * spin_norm[:, None] * force_mag


@pytest.mark.parametrize(
    "spin_norm",
    [
        np.array([1.2737, 0.05, 0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([2.5]),
    ],
)
def test_cartesian_preserves_model_gradient_including_zero_moment(
    spin_norm: np.ndarray,
) -> None:
    """Cartesian mode must not multiply by |m| or divide by hbar."""
    rng = np.random.default_rng(0)
    force_mag = rng.normal(size=(spin_norm.shape[0], 3))
    stored = store_magnetic_force(force_mag, spin_norm, cartesian=True)
    np.testing.assert_allclose(stored, force_mag)
    # Zero-moment atoms keep a non-zero Cartesian force when the model does.
    zero = spin_norm == 0.0
    if np.any(zero):
        assert np.any(np.abs(stored[zero]) > 0.0)


@pytest.mark.parametrize(
    "spin_norm",
    [
        np.array([1.2737, 0.05, 0.0, 0.0]),
        np.array([0.0, 0.0]),
    ],
)
def test_precession_scales_by_moment_over_hbar(spin_norm: np.ndarray) -> None:
    """Precession mode multiplies by |m| / hbar; zero moment stays zero."""
    rng = np.random.default_rng(1)
    force_mag = rng.normal(size=(spin_norm.shape[0], 3))
    stored = store_magnetic_force(force_mag, spin_norm, cartesian=False)
    expected = force_mag * (spin_norm / HBAR_METAL)[:, None]
    np.testing.assert_allclose(stored, expected)
    zero = spin_norm == 0.0
    if np.any(zero):
        np.testing.assert_array_equal(stored[zero], 0.0)


def test_precession_matches_cartesian_times_moment_over_hbar() -> None:
    """On metal units the two conventions differ only by |m| / hbar."""
    spin_norm = np.array([1.1, 0.02, 0.0])
    force_mag = np.array(
        [
            [0.3, -0.1, 0.2],
            [1.0, 0.0, -0.5],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    cartesian = store_magnetic_force(force_mag, spin_norm, cartesian=True)
    precession = store_magnetic_force(force_mag, spin_norm, cartesian=False)
    np.testing.assert_allclose(
        precession, cartesian * (spin_norm / HBAR_METAL)[:, None]
    )


def test_adapter_declares_extract_key_and_keyword() -> None:
    """The C++ adapter must keep the discoverable selection and extract API."""
    root = Path(__file__).resolve().parents[1]
    pair_cpp = (root / "pair_dpa4spin.cpp").read_text(encoding="utf-8")
    pair_h = (root / "pair_dpa4spin.h").read_text(encoding="utf-8")
    assert 'strcmp(str, "fm_cartesian_force")' in pair_cpp
    assert "fm_cartesian_force" in pair_h
    assert 'key == "magnetic_force"' in pair_cpp
    assert 'value == "cartesian"' in pair_cpp
    assert 'value == "precession"' in pair_cpp
    assert "dim = 0" in pair_cpp
    assert 'strcmp(update->unit_style, "metal")' in pair_cpp


def test_non_unit_energy_conversion_applies_only_to_cartesian() -> None:
    """Cartesian uses ener_unit_cvt_factor; precession uses force_unit_cvt_factor."""
    spin_norm = np.array([1.0])
    force_mag = np.array([[2.0, 0.0, 0.0]])
    cartesian = store_magnetic_force(
        force_mag,
        spin_norm,
        cartesian=True,
        ener_unit_cvt_factor=23.060549,
        force_unit_cvt_factor=99.0,
    )
    precession = store_magnetic_force(
        force_mag,
        spin_norm,
        cartesian=False,
        ener_unit_cvt_factor=23.060549,
        force_unit_cvt_factor=2.0,
    )
    np.testing.assert_allclose(cartesian, [[46.121098, 0.0, 0.0]])
    np.testing.assert_allclose(precession, force_mag * (2.0 / HBAR_METAL))
