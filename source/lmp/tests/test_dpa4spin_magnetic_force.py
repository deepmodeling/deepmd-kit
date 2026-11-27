# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check the stored magnetic force through the registered LAMMPS adapter.

The native-spin pair styles store the precession magnetic force of the
LAMMPS SPIN package, |m|/hbar times -dE/dS.

Generate the CPU graph fixture with ``source/tests/infer/gen_dpa4c_spin_lmp.py``.
PyTorch-enabled LAMMPS CI sets DPA4SPIN_LMP_REQUIRED=1 so missing fixtures,
Python bindings, or plugin registration fail instead of hiding regressions.
These tests exercise the host adapter; the GPU-only Kokkos adapter requires a
separate compressed canonical archive and is not covered by this CPU fixture.
"""

from __future__ import (
    annotations,
)

import ctypes
import importlib
import os
import shutil
import subprocess
from contextlib import (
    contextmanager,
)
from pathlib import (
    Path,
)

import numpy as np
import pytest
from write_lmp_data import (
    write_lmp_data_spin,
)

HBAR_METAL = 6.5821191e-04
_MODEL = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "infer"
    / "deeppot_dpa4c_spin_lmp.pt2"
)
_BOX = np.array([0.0, 6.0, 0.0, 6.0, 0.0, 6.0, 0.0, 0.0, 0.0])
_COORD = np.array(
    [
        [1.0, 1.0, 1.0],
        [3.2, 1.4, 1.1],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ]
)
_SPIN = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.07, 0.09, 0.03],
        [0.02, -0.06, 0.08],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
_TYPE = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)


def _unavailable(reason: str) -> None:
    if os.environ.get("DPA4SPIN_LMP_REQUIRED") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def adapter():
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip("PyTorch backend is disabled")
    try:
        module = importlib.import_module("lammps")
    except ImportError as exc:
        _unavailable(f"LAMMPS Python bindings unavailable: {exc}")
        raise  # Keep the import failure path explicit for static analysis.
    model = Path(os.environ.get("DPA4SPIN_LMP_MODEL", _MODEL))
    if not model.is_file():
        _unavailable(f"Generate the native-spin LAMMPS fixture first: {model}")
    with module.lammps(cmdargs=["-log", "none", "-screen", "none"]) as lmp:
        if not lmp.has_style("pair", "dpa4spin"):
            _unavailable("LAMMPS plugin did not register pair style dpa4spin")
    return module.lammps, model


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "nio.lmp"
    write_lmp_data_spin(_BOX, _COORD, _SPIN, _TYPE, path)
    return path


@contextmanager
def _system(adapter, data_file: Path, style: str = "dpa4spin"):
    lammps_cls, model = adapter
    with lammps_cls(cmdargs=["-log", "none", "-screen", "none"]) as lmp:
        for command in (
            "units metal",
            "atom_style spin",
            "atom_modify map yes",
            f"read_data {data_file.resolve()}",
            "mass 1 58.6934",
            "mass 2 15.9994",
            f"pair_style {style} {model.resolve()}",
            "pair_coeff * * Ni O",
            "neighbor 2.0 bin",
            "compute fmprop all property/atom fmx fmy fmz",
            "compute spprop all property/atom spx spy spz sp",
            "run 0",
        ):
            lmp.command(command)
        yield lmp


def _pair_pointer(lmp, name: str, pointer_type):
    # extract_pair() reads one double, whereas scale is a matrix of doubles.
    extract = lmp.lib.lammps_extract_pair
    previous = extract.restype
    extract.restype = pointer_type
    try:
        pointer = extract(lmp.lmp, name.encode())
    finally:
        extract.restype = previous
    assert pointer, f"Pair::extract({name!r}) returned null"
    return pointer


def _gather(lmp, name: str, columns: int) -> np.ndarray:
    return np.array(lmp.gather(name, 1, columns)).reshape(-1, columns)


@pytest.mark.parametrize("magnitude", [0.0, 1e-8, 0.2], ids=["zero", "tiny", "finite"])
def test_precession_matches_energy_gradient(adapter, data_file: Path, magnitude: float):
    """Check every component of |m|/hbar * (-dE/dS), including at S=0."""
    spin = _SPIN.copy()
    spin[0] = magnitude * np.array([2.0, -1.0, 2.0]) / 3.0
    write_lmp_data_spin(_BOX, _COORD, spin, _TYPE, data_file)
    with _system(adapter, data_file) as lmp:
        analytic = _gather(lmp, "c_fmprop", 3)[0]
        initial = _gather(lmp, "c_spprop", 4)[0]
        moment = initial[:3] * initial[3]
        eps = 1e-5
        gradient = np.empty(3)
        for component in range(3):
            energies = []
            for sign in (1, -1):
                displaced = moment.copy()
                displaced[component] += sign * eps
                norm = np.linalg.norm(displaced)
                direction = displaced / norm
                # Both displaced norms are positive even when the centre is S=0.
                # LAMMPS's set command rejects an exactly zero magnitude.
                lmp.command(
                    f"set atom 1 spin/atom {norm:.17g} "
                    + " ".join(f"{value:.17g}" for value in direction)
                )
                lmp.command("run 0")
                energies.append(lmp.get_thermo("pe"))
            gradient[component] = -(energies[0] - energies[1]) / (2 * eps)
    assert np.all(np.abs(gradient) > 1e-9), (
        "Every tested spin component must be live",
        gradient,
    )
    # Check components separately: a norm-only comparison could hide a sign
    # error in a small transverse component. A zero moment stores exactly zero.
    factor = initial[3] / HBAR_METAL
    np.testing.assert_allclose(
        analytic, factor * gradient, rtol=5e-3, atol=1e-11 * factor
    )


def test_deepspin_matches_dpa4spin(adapter, data_file: Path):
    """pair_style deepspin stores the outputs dpa4spin does."""
    with _system(adapter, data_file) as lmp:
        expected = {name: _gather(lmp, name, 3) for name in ("c_fmprop", "f")}
        energy = lmp.get_thermo("pe")
    assert np.linalg.norm(expected["c_fmprop"]) > 1e-8
    np.testing.assert_array_equal(expected["c_fmprop"][0], 0.0)
    with _system(adapter, data_file, style="deepspin") as lmp:
        for name, value in expected.items():
            np.testing.assert_allclose(
                _gather(lmp, name, 3), value, rtol=1e-10, atol=1e-14
            )
        assert lmp.get_thermo("pe") == pytest.approx(energy, rel=1e-12)


def test_pair_scale_and_repeated_compute(adapter, data_file: Path):
    """Exercise the actual extract/scale and force accumulation paths."""
    with _system(adapter, data_file) as lmp:
        magnetic_force = _gather(lmp, "c_fmprop", 3)
        force = _gather(lmp, "f", 3)
        energy = lmp.get_thermo("pe")
        scale = _pair_pointer(
            lmp, "scale", ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
        )
        scale[1][1] = 2.75
        for _ in range(2):
            lmp.command("run 0")
            np.testing.assert_allclose(
                _gather(lmp, "c_fmprop", 3),
                2.75 * magnetic_force,
                rtol=1e-10,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                _gather(lmp, "f", 3), 2.75 * force, rtol=1e-10, atol=1e-14
            )
            assert lmp.get_thermo("pe") == pytest.approx(2.75 * energy, rel=1e-12)


@pytest.mark.parametrize("kokkos", [False, True], ids=["host", "kokkos"])
def test_change_units_after_pair_style(tmp_path, kokkos):
    required = "DPA4SPIN_KOKKOS_REQUIRED" if kokkos else "DPA4SPIN_LMP_REQUIRED"
    if os.environ.get(required) != "1":
        pytest.skip(f"requires the matching LAMMPS build ({required}=1)")
    model = _MODEL.with_name("deeppot_dpa4c_spin_lmp_kokkos.pt2") if kokkos else _MODEL
    lmp = os.environ["DEEPMD_KOKKOS_LMP"] if kokkos else shutil.which("lmp")
    assert lmp and model.is_file()
    args = [lmp]
    if kokkos:
        args.extend(
            [
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "newton",
                "on",
                "neigh",
                "half",
                "gpu/aware",
                "off",
            ]
        )
    script = tmp_path / "units.in"
    style = "dpa4spin/kk" if kokkos else "dpa4spin"
    script.write_text(
        "units metal\natom_style spin\natom_modify map yes\n"
        f"pair_style {style} {model}\nunits real\n"
        "region box block 0 10 0 10 0 10\ncreate_box 2 box\n"
        "create_atoms 1 single 3 3 3\nmass * 12\npair_coeff * * Ni O\nrun 0\n"
    )
    run = subprocess.run(
        [*args, "-in", str(script), "-log", "none"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode != 0
    assert "requires units metal" in run.stdout + run.stderr
