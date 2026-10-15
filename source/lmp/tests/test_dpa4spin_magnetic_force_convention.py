# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checks for the ``dpa4spin`` / ``dpa4spin/kk`` magnetic-force conventions.

The LAMMPS adapter stores the model energy gradient with respect to the
magnetic moment in ``atom->fm`` under one of two explicit conventions,
selected by ``pair_style ... magnetic_force precession|cartesian`` and
reported by ``Pair::extract("fm_cartesian_force", dim)``:

* precession (default, extract value 0): ``fm = scale * force_cvt * |m| / hbar
  * force_mag``, matching the stock LAMMPS SPIN integrator;
* cartesian (extract value 1): ``fm = scale * ener_cvt * force_mag``, the raw
  ``-dE/dS`` gradient with only the energy-per-model-spin unit conversion,
  including at zero moment.

NumPy helpers below document the conversion algebra. Behavioral tests drive
the registered ``dpa4spin`` pair style against
``deeppot_dpa4c_spin_lmp.pt2`` (no fparam/aparam/chg_spin) when PyLammps and
the style are available; they skip otherwise so CI without a dpa4spin build
stays green. ``dpa4spin/kk`` requires a GPU Kokkos backend and a compressed
canonical archive, so the Kokkos case is an explicit skip when unavailable.
"""

from __future__ import (
    annotations,
)

import ctypes
import os
import tempfile
from pathlib import (
    Path,
)

import numpy as np
import pytest

# Metal-units reduced Planck constant used by pair_dpa4spin{,_kokkos}.cpp.
HBAR_METAL = 6.5821191e-04

_MODEL_ENV = "DPA4SPIN_LMP_MODEL"
_DEFAULT_MODEL = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "infer"
    / "deeppot_dpa4c_spin_lmp.pt2"
)

# 6-atom NiO geometry shared with source/tests/infer/gen_dpa4c_spin.py
# (box edge 6 A = DPA4C ghost cutoff). Ni is spin-active; O is not.
_BOX = np.array([0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 0.0, 0.0])
_COORD = np.array(
    [
        [1.0, 1.0, 1.0],
        [3.2, 1.4, 1.1],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ],
    dtype=np.float64,
)
# Atom 0: spin-active Ni at zero moment (cartesian fm must stay nonzero).
# Atoms 1-2: nonzero Ni moments. Atoms 3-5: non-magnetic O.
_SPIN = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.07, 0.09, 0.03],
        [0.02, -0.06, 0.08],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
_TYPE = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)


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
    pair_kk = (root / "pair_dpa4spin_kokkos.cpp").read_text(encoding="utf-8")
    assert 'strcmp(str, "fm_cartesian_force")' in pair_cpp
    assert "fm_cartesian_force" in pair_h
    assert 'key == "magnetic_force"' in pair_cpp
    assert 'value == "cartesian"' in pair_cpp
    assert 'value == "precession"' in pair_cpp
    assert "dim = 0" in pair_cpp
    assert 'strcmp(update->unit_style, "metal")' in pair_cpp
    # Primary issue target is dpa4spin/kk: scatter must keep both conventions.
    assert "fm_cartesian_force" in pair_kk
    assert "ener_unit_cvt_factor" in pair_kk
    assert "cartesian ? 1.0 : sp(i, 3)" in pair_kk


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


# ---------------------------------------------------------------------------
# Behavioral coverage through the real ``dpa4spin`` adapter
# ---------------------------------------------------------------------------


def _resolve_model() -> Path:
    env = os.environ.get(_MODEL_ENV)
    return Path(env) if env else _DEFAULT_MODEL


def _require_dpa4spin_host():
    """Import PyLammps and require a host ``dpa4spin`` registration."""
    try:
        from lammps import (
            lammps,
        )
    except ImportError:
        pytest.skip("PyLammps is not available")
    model = _resolve_model()
    if not model.is_file():
        pytest.skip(
            f"native-spin dpa4spin fixture missing ({model}); "
            f"set {_MODEL_ENV} or run source/tests/infer/gen_dpa4c_spin_lmp.py"
        )
    probe = lammps(cmdargs=["-log", "none", "-screen", "none"])
    try:
        if not probe.has_style("pair", "dpa4spin"):
            pytest.skip("pair style dpa4spin is not registered in this LAMMPS build")
    finally:
        probe.close()
    return lammps, model


def _extract_fm_cartesian_force(lmp) -> int:
    """Read the scalar int from ``Pair::extract("fm_cartesian_force", dim)``."""
    extract = lmp.lib.lammps_extract_pair
    # The Python binding defaults restype to c_int, which truncates pointers.
    previous = extract.restype
    extract.restype = ctypes.c_void_p
    try:
        c_ptr = extract(lmp.lmp, b"fm_cartesian_force")
    finally:
        extract.restype = previous
    if not c_ptr:
        raise AssertionError("extract(fm_cartesian_force) returned null")
    return int(ctypes.cast(c_ptr, ctypes.POINTER(ctypes.c_int)).contents.value)


def _open_dpa4spin(lammps_cls, data_file: Path, model: Path, pair_args: str):
    """Build a single-rank metal/spin system with ``pair_style dpa4spin``."""
    lmp = lammps_cls(cmdargs=["-log", "none", "-screen", "none"])
    try:
        for cmd in (
            "units metal",
            "dimension 3",
            "boundary p p p",
            "atom_style spin",
            "atom_modify map yes",
            f"read_data {data_file.resolve()}",
            "mass 1 58.6934",
            "mass 2 15.9994",
            f"pair_style dpa4spin {model.resolve()} {pair_args}".rstrip(),
            "pair_coeff * * Ni O",
            "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            "compute fmprop all property/atom fmx fmy fmz",
            "compute spprop all property/atom spx spy spz sp",
            "run 0",
        ):
            lmp.command(cmd)
    except Exception:
        lmp.close()
        raise
    return lmp


def _gather_fm_sp(lmp) -> tuple[np.ndarray, np.ndarray]:
    natoms = int(lmp.get_natoms())
    fm = np.array(lmp.gather("c_fmprop", 1, 3), dtype=np.float64).reshape(natoms, 3)
    sp = np.array(lmp.gather("c_spprop", 1, 4), dtype=np.float64).reshape(natoms, 4)
    return fm, sp


@pytest.fixture(scope="module")
def dpa4spin_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the mixed zero-/nonzero-moment NiO data file once per module."""
    from write_lmp_data import (
        write_lmp_data_spin,
    )

    path = tmp_path_factory.mktemp("dpa4spin_mf") / "nio_zero_and_finite.lmp"
    write_lmp_data_spin(_BOX, _COORD, _SPIN, _TYPE, path)
    return path


def test_dpa4spin_extract_reports_magnetic_force_selection(
    dpa4spin_data: Path,
) -> None:
    """``magnetic_force`` selection must be visible via Pair::extract."""
    lammps_cls, model = _require_dpa4spin_host()
    for args, expect in (
        ("", 0),
        ("magnetic_force precession", 0),
        ("magnetic_force cartesian", 1),
    ):
        lmp = _open_dpa4spin(lammps_cls, dpa4spin_data, model, args)
        try:
            assert _extract_fm_cartesian_force(lmp) == expect
        finally:
            lmp.close()


def test_dpa4spin_cartesian_fm_matches_energy_fd_and_keeps_zero_moment(
    dpa4spin_data: Path,
) -> None:
    """Cartesian ``fm`` is ``-dE/dS`` and stays nonzero at zero moment on Ni."""
    lammps_cls, model = _require_dpa4spin_host()
    lmp = _open_dpa4spin(lammps_cls, dpa4spin_data, model, "magnetic_force cartesian")
    try:
        fm, sp = _gather_fm_sp(lmp)
        assert sp[0, 3] == pytest.approx(0.0)
        assert np.linalg.norm(fm[0]) > 1e-8, (
            "spin-active Ni at zero moment must keep a nonzero Cartesian fm"
        )
        assert sp[1, 3] > 0.0
        assert np.linalg.norm(fm[1]) > 1e-8

        # Energy FD on the largest |fm| component of Ni atom 2 (1-based id 2).
        # ``set atom ... spin`` takes magnitude then direction (mag > 0).
        s0 = sp[1, :3] * sp[1, 3]
        eps = 1e-5
        comp = int(np.argmax(np.abs(fm[1])))
        analytic0 = float(fm[1, comp])
        assert abs(analytic0) > 1e-8, (
            f"largest |fm[1]| component must be live for FD: fm[1]={fm[1]}"
        )
        ehat = np.zeros(3, dtype=np.float64)
        ehat[comp] = 1.0

        def energy_at(moment: np.ndarray) -> float:
            mag = float(np.linalg.norm(moment))
            assert mag > 0.0
            u = moment / mag
            lmp.command(
                f"set atom 2 spin {mag:.16e} {u[0]:.16e} {u[1]:.16e} {u[2]:.16e}"
            )
            lmp.command("run 0")
            return float(lmp.get_thermo("pe"))

        ep = energy_at(s0 + eps * ehat)
        em = energy_at(s0 - eps * ehat)
        energy_at(s0)
        fm_restored, _ = _gather_fm_sp(lmp)
        fd = -(ep - em) / (2.0 * eps)
        analytic = float(fm_restored[1, comp])
        assert abs(analytic) > 1e-8, (
            f"analytic fm[{comp}] vanished after restore: analytic={analytic}"
        )
        # Relative only: an absolute escape of ~1e-8 would let a scatter
        # sign flip (+= -> -=) pass when |fm| is tiny; with |analytic| > 1e-8,
        # |analytic - (-analytic)| = 2|analytic| fails both rel and a tight abs.
        rel = abs(fd - analytic) / max(1e-12, abs(analytic) + abs(fd))
        assert rel < 5e-3, (
            f"cartesian fm[{comp}] vs energy FD: analytic={analytic}, fd={fd}, rel={rel}"
        )
    finally:
        lmp.close()


def test_dpa4spin_default_precession_scales_cartesian_by_moment_over_hbar(
    dpa4spin_data: Path,
) -> None:
    """Default precession ``fm`` equals cartesian ``fm`` times ``|m| / hbar``."""
    lammps_cls, model = _require_dpa4spin_host()
    lmp_c = _open_dpa4spin(lammps_cls, dpa4spin_data, model, "magnetic_force cartesian")
    try:
        fm_c, sp = _gather_fm_sp(lmp_c)
    finally:
        lmp_c.close()

    lmp_p = _open_dpa4spin(lammps_cls, dpa4spin_data, model, "")
    try:
        assert _extract_fm_cartesian_force(lmp_p) == 0
        fm_p, _ = _gather_fm_sp(lmp_p)
    finally:
        lmp_p.close()

    # Zero-moment spin-active Ni: precession must vanish; cartesian must not.
    np.testing.assert_allclose(fm_p[0], 0.0, atol=1e-14)
    assert np.linalg.norm(fm_c[0]) > 1e-8

    # Nonzero Ni moments: ratio must be |m|/hbar.
    for i in (1, 2):
        m = sp[i, 3]
        expect = m / HBAR_METAL
        for d in range(3):
            if abs(fm_c[i, d]) < 1e-18:
                assert abs(fm_p[i, d]) < 1e-14
                continue
            ratio = fm_p[i, d] / fm_c[i, d]
            np.testing.assert_allclose(ratio, expect, rtol=1e-8, atol=0.0)

    # Non-magnetic O: both conventions stay at zero.
    np.testing.assert_allclose(fm_c[3:], 0.0, atol=1e-14)
    np.testing.assert_allclose(fm_p[3:], 0.0, atol=1e-14)


def test_dpa4spin_kokkos_cartesian_skipped_without_gpu_backend() -> None:
    """Exercise ``dpa4spin/kk`` when a GPU Kokkos build is present; else skip."""
    try:
        from lammps import (
            lammps,
        )
    except ImportError:
        pytest.skip("PyLammps is not available")

    model = _resolve_model()
    if not model.is_file():
        pytest.skip(f"fixture missing: {model}")

    probe = lammps(cmdargs=["-log", "none", "-screen", "none"])
    try:
        has_kk = probe.has_style("pair", "dpa4spin/kk")
        has_kokkos = probe.has_package("KOKKOS")
    finally:
        probe.close()
    if not has_kk or not has_kokkos:
        pytest.skip("dpa4spin/kk or KOKKOS package not available")

    # dpa4spin/kk rejects host Kokkos backends and graph-lower (uncompressed)
    # archives; a GPU + compressed canonical model is required.
    try:
        lmp = lammps(
            cmdargs=[
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-log",
                "none",
                "-screen",
                "none",
            ]
        )
    except Exception as exc:
        pytest.skip(f"GPU Kokkos backend unavailable: {exc}")

    try:
        if not lmp.has_style("pair", "dpa4spin/kk"):
            pytest.skip("dpa4spin/kk not registered under Kokkos")
        # Probe whether this archive is accepted; graph-lower must skip.
        from write_lmp_data import (
            write_lmp_data_spin,
        )

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "nio.lmp"
            write_lmp_data_spin(_BOX, _COORD, _SPIN, _TYPE, data)
            try:
                for cmd in (
                    "units metal",
                    "dimension 3",
                    "boundary p p p",
                    "atom_style spin",
                    "atom_modify map yes",
                    f"read_data {data.resolve()}",
                    "mass 1 58.6934",
                    "mass 2 15.9994",
                    f"pair_style dpa4spin/kk {model.resolve()} "
                    "magnetic_force cartesian",
                    "pair_coeff * * Ni O",
                    "neighbor 2.0 bin",
                    "neigh_modify every 1 delay 0 check yes",
                    "compute fmprop all property/atom fmx fmy fmz",
                    "run 0",
                ):
                    lmp.command(cmd)
            except Exception as exc:
                msg = str(exc)
                if "GPU backend only" in msg or "compact canonical" in msg:
                    pytest.skip(f"dpa4spin/kk cannot run here: {msg}")
                raise
            assert _extract_fm_cartesian_force(lmp) == 1
            natoms = int(lmp.get_natoms())
            fm = np.array(lmp.gather("c_fmprop", 1, 3), dtype=np.float64).reshape(
                natoms, 3
            )
            assert np.linalg.norm(fm[0]) > 1e-8
            assert np.linalg.norm(fm[1]) > 1e-8
    finally:
        lmp.close()
