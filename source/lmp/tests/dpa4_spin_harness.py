# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared harness for the DPA4 native-spin LAMMPS ``.pt2`` tests.

Owns the mechanics common to every DPA4 native-spin variant -- the 4-atom
NiO system, the live ``DeepPot`` reference, LAMMPS construction, magnetic
force gathering, the MPI subprocess/parser, and the single-rank and
rank-parity assertions -- so each variant module carries only its archive,
its geometry and its variant-specific claims.

Narrowly scoped rather than folded into ``lammps_test_utils``: the pieces
here are DPA4-native-spin-specific (``atom_modify map yes`` for the graph
route's atom map, the ``spin_norm / hbar`` ``fm`` convention, and a
timeout-bounded runner that SIGKILLs the whole mpirun process group) and
would change behaviour for the unrelated tests that share that module.

Consumers: ``test_lammps_dpa4_spin_graph_pt2.py`` (plain native spin) and
``test_lammps_dpa4_spin_zbl_pt2.py`` (native spin + ZBL bridging).
"""

import json
import os
import signal
import subprocess as sp
import sys
import tempfile
import textwrap
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import constants
import numpy as np
import pytest
from lammps import (
    PyLammps,
)

MPI_DEFAULT_TIMEOUT = 120.0

# LAMMPS's ``fm`` (what ``compute property/atom fmx fmy fmz`` reports) is
# NOT the raw DeepEval force_mag: pair_deepspin.cpp scales it by
# ``spin_norm / hbar`` per atom (metal-units ``hbar = 6.5821191e-04``, see
# ``source/lmp/pair_deepspin.cpp:531,535``).  ``spin_norm`` is 0 for the two
# non-magnetic O atoms, so the scaling is a no-op there (0 stays 0).
HBAR_METAL = 6.5821191e-04

# The 4-atom NiO system every DPA4 native-spin variant shares (box,
# coordinates and LAMMPS type ordering reused verbatim from
# test_lammps_spin_pt2.py): 2 Ni atoms (LAMMPS type 1, deepmd atype 0,
# spin-active) + 2 O atoms (LAMMPS type 2, deepmd atype 1, non-magnetic) --
# matching the archives' ``type_map=["Ni", "O"]`` and
# ``use_spin=[True, False]``.  The Ni-Ni pair (atoms 0 and 1) sits ~0.978 A
# apart, which is inside the ZBL bridging transition zone (0.8, 1.2) used by
# the bridged variant.
BOX = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
COORD = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
SPIN = np.array(
    [
        [0, 0, 1.2737],
        [0, 0, 1.2737],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
TYPE_NIO = np.array([1, 1, 2, 2])

# LAMMPS Voigt-ish component order used by the virial/pressure checks.
_VIRIAL_ORDER = [0, 4, 8, 3, 6, 7, 1, 2, 5]


def cell_from_lammps_box(lmp_box: np.ndarray) -> np.ndarray:
    """Convert a LAMMPS ``xlo xhi ylo yhi zlo zhi xy xz yz`` box spec to a
    flat, row-major 3x3 cell matrix (deepmd's ``box`` convention).
    """
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = lmp_box
    return np.array(
        [
            xhi - xlo,
            0.0,
            0.0,
            xy,
            yhi - ylo,
            0.0,
            xz,
            yz,
            zhi - zlo,
        ]
    )


def compute_expected(
    pb_file: Path,
    *,
    box: np.ndarray = BOX,
    coord: np.ndarray = COORD,
    spin: np.ndarray = SPIN,
    type_map: np.ndarray = TYPE_NIO,
) -> dict[str, Any]:
    """Evaluate ``pb_file`` on the fixed system to obtain the reference.

    Runs in a subprocess to avoid importing ``deepmd`` in the LAMMPS test
    process (see ``test_lammps_model_devi_pt2.py``'s ``_compute_expected``
    for the same precaution: the LAMMPS plugin already loads
    ``libdeepmd_op_pt.so`` at the C++ level, and importing the Python
    package on top of that can segfault).

    The archive lives in ``source/tests/infer`` next to ``gen_common.py``,
    whose ``load_custom_ops()`` loads the build-tree ``libdeepmd_op_pt.so``
    (registering ``deepmd::edge_force_virial``, which the graph ``.pt2``
    inference needs). ``import deepmd.pt`` alone only loads the op library
    from SHARED_LIB_DIR, which the build-test env does not populate -- so
    the subprocess reuses that fallback (after importing ``deepmd.pt``, per
    its docstring) before constructing ``DeepPot``.

    Parameters
    ----------
    pb_file : Path
        The ``.pt2`` archive to evaluate.
    box : np.ndarray
        LAMMPS box spec of the system.
    coord : np.ndarray
        Per-atom coordinates.
    spin : np.ndarray
        Per-atom spin vectors.
    type_map : np.ndarray
        LAMMPS 1-based per-atom types.

    Returns
    -------
    dict
        ``e`` (energy), ``ae`` (atom energies), ``f`` (forces), ``fm``
        (magnetic forces, already scaled to LAMMPS's ``fm`` convention) and
        ``v`` (per-atom virial, sign-flipped to LAMMPS's convention).
    """
    cell = cell_from_lammps_box(box)
    atype = (type_map - 1).tolist()  # LAMMPS 1-based -> deepmd 0-based
    infer_dir = str(pb_file.resolve().parent)
    script = textwrap.dedent(f"""\
        import json
        import sys
        import numpy as np

        sys.path.insert(0, {infer_dir!r})
        import deepmd.pt  # noqa: F401  (triggers the base op-library load)
        from gen_common import load_custom_ops

        load_custom_ops()
        from deepmd.infer import DeepPot

        dp = DeepPot({str(pb_file.resolve())!r})
        e, f, v, ae, av, fm, mm = dp.eval(
            np.array({coord.tolist()!r}).reshape(1, -1, 3),
            np.array({cell.tolist()!r}).reshape(1, 9),
            {atype!r},
            atomic=True,
            spin=np.array({spin.tolist()!r}).reshape(1, -1, 3),
        )
        print(json.dumps({{
            "e": float(e[0, 0]),
            "ae": np.asarray(ae[0]).reshape(-1).tolist(),
            "f": np.asarray(f[0]).tolist(),
            "fm": np.asarray(fm[0]).tolist(),
            "av": np.asarray(av[0]).tolist(),
        }}))
    """)
    proc = sp.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to compute expected values:\n{proc.stderr}")
    result = json.loads(proc.stdout.strip())

    # Raw DeepEval force_mag (dE/dspin), scaled by LAMMPS's own
    # spin_norm / hbar unit convention (see HBAR_METAL) before comparison.
    fm_raw = np.array(result["fm"])
    spin_norm = np.linalg.norm(spin, axis=1)
    return {
        "e": result["e"],
        "ae": np.array(result["ae"]),
        "f": np.array(result["f"]),
        "fm": fm_raw * (spin_norm / HBAR_METAL)[:, None],
        # Per-atom virial, sign-flipped (LAMMPS convention) relative to
        # DeepPot's atomic virial output (mirrors test_lammps_spin_pt2.py).
        "v": -np.array(result["av"]),
    }


def make_lammps(data_file: Path, units: str = "metal") -> PyLammps:
    """Standard DeepSpin LAMMPS system, plus ``atom_modify map yes``.

    Mirrors ``lammps_test_utils.make_spin_lammps`` (not reused directly: it
    does not set ``atom_modify``), with the map turned on -- the native-spin
    DPA4 GRAPH ``.pt2`` needs the LAMMPS atom-map to resolve ghost-atom
    indices to local owners for single-rank inference (same requirement as
    the energy graph route; see ``pair_deepspin.cpp``'s
    ``DeePMD-kit Error: Single-rank LAMMPS .pt2 inference requires
    `atom_modify map yes``` check).

    Parameters
    ----------
    data_file : Path
        LAMMPS data file to read.
    units : str
        Unit system; only ``"metal"`` is valid for spin.

    Returns
    -------
    PyLammps
        The constructed LAMMPS instance.

    Raises
    ------
    ValueError
        If ``units`` is not ``"metal"``.
    """
    if units != "metal":
        raise ValueError("units for spin should be metal")

    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("spin")
    lammps.atom_modify("map yes")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 58")
    lammps.mass("2 16")
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


def gather_force_mag(lammps: PyLammps, natoms: int) -> np.ndarray:
    """Extract per-atom force_mag in atom-id order.

    LAMMPS does not expose ``fm`` through the legacy ``extract``/
    ``gather_atoms`` registry (see ``run_mpi_pair_deepmd_spin_dpa3_pt2.py``'s
    module docstring), so go via ``compute property/atom fmx fmy fmz`` +
    ``gather`` (id-ordered on every rank, single-rank included).

    Parameters
    ----------
    lammps : PyLammps
        A LAMMPS instance with an ``fmprop`` compute defined.
    natoms : int
        Number of atoms.

    Returns
    -------
    np.ndarray
        ``(natoms, 3)`` magnetic forces in atom-id order.
    """
    fm_global = lammps.lmp.gather("c_fmprop", 1, 3)
    return np.array(fm_global, dtype=np.float64).reshape(natoms, 3)


def run_mpi_spin_runner(
    mpi_runner: Path,
    pb_file: Path,
    data_path: Path,
    *,
    nprocs: int = 2,
    processors: str | None = None,
    extra_args: list[str] | None = None,
    capture: bool = False,
    timeout: float | None = None,
) -> dict:
    """Invoke the graph-spin MPI runner under ``mpirun -n <nprocs>``.

    With ``capture=True``, return raw subprocess info (``returncode``,
    ``stdout``, ``stderr``, ``timed_out``). Every invocation is bounded by
    ``timeout`` (default ``MPI_DEFAULT_TIMEOUT``) so a should-fail-but-
    doesn't run cannot hang the suite, and on expiry the WHOLE mpirun
    process group is SIGKILLed.

    Parameters
    ----------
    mpi_runner : Path
        The runner script to launch.
    pb_file : Path
        The ``.pt2`` archive under test.
    data_path : Path
        LAMMPS data file for this run.
    nprocs : int
        Number of MPI ranks.
    processors : str, optional
        LAMMPS ``processors`` grid; defaults to ``1 1 1`` when ``nprocs==1``.
    extra_args : list of str, optional
        Extra runner arguments.
    capture : bool
        Return raw subprocess info instead of parsing the output.
    timeout : float, optional
        Wall-clock bound; defaults to ``MPI_DEFAULT_TIMEOUT``.

    Returns
    -------
    dict
        ``{"pe": float, "rows": np.ndarray}``, or the raw subprocess info
        when ``capture`` is true.

    Raises
    ------
    RuntimeError
        If a non-``capture`` run exceeds ``timeout``.
    subprocess.CalledProcessError
        If a non-``capture`` run exits non-zero.
    """
    if timeout is None:
        timeout = MPI_DEFAULT_TIMEOUT
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        out_path = f.name
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(mpi_runner),
            str(data_path.resolve()),
            str(pb_file.resolve()),
            out_path,
        ]
        if processors is not None:
            argv.extend(["--processors", processors])
        elif nprocs == 1:
            argv.extend(["--processors", "1 1 1"])
        if extra_args:
            argv.extend(extra_args)
        proc = sp.Popen(
            argv,
            stdout=sp.PIPE if capture else None,
            stderr=sp.PIPE if capture else None,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except sp.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            if capture:
                return {
                    "returncode": None,
                    "stdout": stdout or "",
                    "stderr": stderr or "",
                    "timed_out": True,
                }
            raise RuntimeError(
                f"mpirun timed out after {timeout}s (process group killed); "
                "a should-succeed MPI regression is deadlocked."
            ) from None
        if capture:
            return {
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": False,
            }
        if proc.returncode != 0:
            raise sp.CalledProcessError(proc.returncode, argv)
        with open(out_path) as fh:
            lines = fh.read().strip().splitlines()
        pe = float(lines[0])
        rows = np.array(
            [list(map(float, line.split())) for line in lines[1:]],
            dtype=np.float64,
        )
        return {"pe": pe, "rows": rows}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def assert_mpi_matches_single_rank(single: dict, multi: dict) -> None:
    """Compare a multi-rank MPI result against the 1-rank one: pe, per-atom
    force and per-atom force_mag, all at rtol=atol=1e-10.

    Both runs report LAMMPS's own ``fm`` (already scaled by
    ``spin_norm / HBAR_METAL`` inside pair_deepspin.cpp), so the scaling
    cancels and the rows compare directly.

    Parameters
    ----------
    single : dict
        Result of the 1-rank run.
    multi : dict
        Result of the multi-rank run.
    """
    # anti-vacuity: a degenerate fixture (all-zero forces) would make the
    # comparison pass for the wrong reason.
    assert np.abs(multi["rows"][:, :3]).max() > 1e-6, "forces are trivially zero"
    assert np.abs(multi["rows"][:, 3:6]).max() > 1e-6, "force_mag is trivially zero"

    np.testing.assert_allclose(
        multi["pe"], single["pe"], rtol=1e-10, atol=1e-10, err_msg="energy"
    )
    np.testing.assert_allclose(
        multi["rows"][:, :3],
        single["rows"][:, :3],
        rtol=1e-10,
        atol=1e-10,
        err_msg="force",
    )
    np.testing.assert_allclose(
        multi["rows"][:, 3:6],
        single["rows"][:, 3:6],
        rtol=1e-10,
        atol=1e-10,
        err_msg="force_mag",
    )


def check_single_rank_energy_force(
    lammps: PyLammps,
    pb_file: Path,
    expected: dict,
    *,
    coord: np.ndarray = COORD,
) -> None:
    """Single-rank LAMMPS energy + force + force_mag vs the DeepEval
    reference, including the native-spin zero-``fm``-on-non-spin invariant.

    Parameters
    ----------
    lammps : PyLammps
        The LAMMPS instance from the module's fixture.
    pb_file : Path
        The ``.pt2`` archive under test.
    expected : dict
        Reference produced by :func:`compute_expected`.
    coord : np.ndarray
        The system coordinates (row count sets the atom count).
    """
    natoms = coord.shape[0]
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("fmprop all property/atom fmx fmy fmz")
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(expected["e"])

    forces = np.array(
        [lammps.atoms[ii].force for ii in range(natoms)], dtype=np.float64
    )
    ids = np.array([lammps.atoms[ii].id for ii in range(natoms)])
    forces = forces[np.argsort(ids)]
    np.testing.assert_allclose(forces, expected["f"], atol=1e-8, rtol=0)

    force_mag = gather_force_mag(lammps, natoms)
    np.testing.assert_allclose(force_mag, expected["fm"], atol=1e-8, rtol=0)
    # Anti-vacuity / native-spin design invariant: force_mag on the two
    # non-spin (O) atoms must be exactly zero, both in the Python reference
    # (baked into expected["fm"]) and as produced by LAMMPS.
    np.testing.assert_array_equal(force_mag[2:], np.zeros((natoms - 2, 3)))

    lammps.run(1)


def check_single_rank_virial(
    lammps: PyLammps,
    pb_file: Path,
    expected: dict,
    *,
    box: np.ndarray = BOX,
    coord: np.ndarray = COORD,
) -> None:
    """Single-rank per-atom pe/pressure/virial via ``pe/atom`` /
    ``pressure`` / ``centroid/stress/atom``, atol=1e-8, rtol=1e-8.

    Parameters
    ----------
    lammps : PyLammps
        The LAMMPS instance from the module's fixture.
    pb_file : Path
        The ``.pt2`` archive under test.
    expected : dict
        Reference produced by :func:`compute_expected`.
    box : np.ndarray
        LAMMPS box spec (sets the cell volume).
    coord : np.ndarray
        The system coordinates (row count sets the atom count).
    """
    natoms = coord.shape[0]
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("peatom all pe/atom pair")
    lammps.compute("pressure all pressure NULL pair")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    lammps.variable("eatom atom c_peatom")
    for ii in range(9):
        jj = _VIRIAL_ORDER[ii]
        lammps.variable(f"pressure{jj} equal c_pressure[{ii + 1}]")
    for ii in range(9):
        jj = _VIRIAL_ORDER[ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(expected["e"])

    forces = np.array(
        [lammps.atoms[ii].force for ii in range(natoms)], dtype=np.float64
    )
    ids = np.array([lammps.atoms[ii].id for ii in range(natoms)])
    forces = forces[np.argsort(ids)]
    np.testing.assert_allclose(forces, expected["f"], atol=1e-8, rtol=0)

    idx_map = lammps.lmp.numpy.extract_atom("id")[:natoms] - 1
    np.testing.assert_allclose(
        np.array(lammps.variables["eatom"].value),
        expected["ae"][idx_map],
        atol=1e-8,
        rtol=1e-8,
    )

    vol = box[1] * box[3] * box[5]
    for ii in range(6):
        jj = _VIRIAL_ORDER[ii]
        pressure_jj = np.array(lammps.variables[f"pressure{jj}"].value) / (
            constants.nktv2p
        )
        expected_pressure_jj = -expected["v"][idx_map, jj].sum(axis=0) / vol
        np.testing.assert_allclose(
            pressure_jj, expected_pressure_jj, atol=1e-8, rtol=1e-8
        )
    for ii in range(9):
        jj = _VIRIAL_ORDER[ii]
        virial_jj = np.array(lammps.variables[f"virial{jj}"].value) / (constants.nktv2p)
        np.testing.assert_allclose(
            virial_jj, expected["v"][idx_map, jj], atol=1e-8, rtol=1e-8
        )
