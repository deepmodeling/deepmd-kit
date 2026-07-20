# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-rank LAMMPS ``pair_style deepspin`` on the native-spin DPA4
NeighborGraph (graph-schema) ``.pt2`` (Task 8's ``deeppot_dpa4_spin_graph.pt2``).

Unlike the virtual-atom ``spin_ener`` scheme exercised by
``test_lammps_spin_pt2.py``, native-spin DPA4 has NO dense/nlist lower at
all -- spin rides the NeighborGraph lower exclusively (see
``deepmd/pt_expt/model/dpa4_native_spin_model.py``'s module docstring and
``source/tests/infer/gen_dpa4_spin.py``). The fixture is also exported with
``has_comm_artifact=false`` unconditionally (no nested with-comm AOTI
artifact), so multi-rank LAMMPS has no cross-rank ghost-feature-exchange
route to fall back to: ``DeepSpinPTExpt::compute`` fails fast on
*any* ``nprocs > 1`` run of a graph-kind spin archive, independent of the
usual ``has_comm_artifact_`` / ``atom_map`` decision matrix used for energy
models and for the virtual-atom spin scheme (see
``source/api_cc/src/DeepSpinPTExpt.cc``, the
``lower_input_is_graph_ && multi_rank && has_message_passing_`` guard).

Reference (energy / force / force_mag / virial) values are computed LIVE at
test-setup time via ``deepmd.infer.DeepPot.eval`` on
``deeppot_dpa4_spin_graph.pt2`` for the fixed 4-atom NiO system reused from
``test_lammps_spin_pt2.py`` (box 13x13x13, same coordinates/type ordering: 2
spin-active Ni + 2 non-magnetic O) -- i.e. exactly the Task 7 graph-spin
Python eval path, driven here through LAMMPS instead.

The reference is deliberately NOT hardcoded (a previous revision hardcoded
it and went stale within ~1e-6 the moment master's ``dpa4_nn``
physical-null-mass-attention change shifted DPA4 numerics -- exactly the
fragility flagged in the Task 10 review). Nor is it read from a sidecar
``.expected`` file produced by ``source/tests/infer/gen_dpa4_spin.py``:
that script's own PBC eval uses a DIFFERENT 6-atom (3 Ni + 3 O) system in a
6x6x6 box (see its module docstring / ``_COORDS`` / ``_CELL`` / ``_SPINS``),
and that box's edge length (6.0) exactly equals DPA4's ghost cutoff
(rcut(4.0)+skin(2.0)=6.0) -- not a safe geometry to reuse for a LAMMPS
periodic run. Instead, ``_compute_expected`` below loads the archive and
evaluates it, at test-setup time, on THIS module's own fixed geometry --
mirroring ``test_lammps_model_devi_pt2.py``'s ``_compute_expected`` pattern
(subprocess-isolated, so importing ``deepmd``'s Python package does not
share a process with the LAMMPS plugin's own loaded ``libdeepmd_op_pt.so``).
This keeps the reference correct-by-construction: it always reflects
whatever the current archive produces, so a real DPA4 numerics shift is
caught by comparing against the *previous* run's output changing (reviewed
in the PR), not by a silently-stale hardcoded array.
"""

import importlib.util
import json
import os
import shutil
import signal
import subprocess as sp
import sys
import tempfile
import textwrap
from pathlib import (
    Path,
)

import constants
import numpy as np
import pytest
from lammps import (
    PyLammps,
)
from write_lmp_data import (
    write_lmp_data_spin,
)

pb_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_spin_graph.pt2"
)
data_file = Path(__file__).parent / "data_dpa4_spin_graph_pt2.lmp"
# The MPI runner is graph-spin-specific (no aparam / no NULL-type
# extras, unlike run_mpi_pair_deepmd_spin_dpa3_pt2.py's virtual-atom-scheme
# runner): the native-spin DPA4 fixture takes no fparam/aparam.
mpi_runner = Path(__file__).parent / "run_mpi_pair_deepmd_spin_graph_dpa4_pt2.py"

_MPI_DEFAULT_TIMEOUT = 120.0

# Same 4-atom NiO system as test_lammps_spin_pt2.py (box, coordinates, and
# LAMMPS type ordering all reused verbatim): 2 Ni atoms (LAMMPS type 1,
# deepmd atype 0, spin-active) + 2 O atoms (LAMMPS type 2, deepmd atype 1,
# non-magnetic) -- matches ``deeppot_dpa4_spin_graph.pt2``'s
# ``type_map=["Ni", "O"]`` and ``use_spin=[True, False]`` (gen_dpa4_spin.py).
box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
spin = np.array(
    [
        [0, 0, 1.2737],
        [0, 0, 1.2737],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
type_NiO = np.array([1, 1, 2, 2])

# LAMMPS's ``fm`` (what ``compute property/atom fmx fmy fmz`` reports) is
# NOT the raw DeepEval force_mag: pair_deepspin.cpp scales it by
# ``spin_norm / hbar`` per atom (metal-units ``hbar = 6.5821191e-04``, see
# ``source/lmp/pair_deepspin.cpp:531,535`` -- same convention already
# implicit, if untested against a raw LAMMPS ``fm`` read, in
# test_lammps_spin_pt2.py). ``spin_norm`` is 0 for the two non-magnetic O
# atoms, so the scaling is a no-op there (0 stays 0).
_HBAR_METAL = 6.5821191e-04

# Reference values (energy / atom-energy / force / force_mag / virial),
# populated by ``_compute_expected`` in ``setup_module`` -- see the module
# docstring for why these are computed live via a DeepPot subprocess call
# rather than hardcoded or read from a sidecar file.
expected_e = None
expected_ae = None
expected_f = None
expected_fm = None
expected_v = None


def _cell_from_lammps_box(lmp_box: np.ndarray) -> np.ndarray:
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


def _compute_expected() -> None:
    """Load ``deeppot_dpa4_spin_graph.pt2`` via ``DeepPot`` and evaluate the
    module's fixed 4-atom NiO system to obtain the Python reference.

    Runs in a subprocess to avoid importing ``deepmd`` in the LAMMPS test
    process (see ``test_lammps_model_devi_pt2.py``'s ``_compute_expected``
    for the same precaution: the LAMMPS plugin already loads
    ``libdeepmd_op_pt.so`` at the C++ level, and importing the Python
    package on top of that can segfault).
    """
    global expected_e, expected_ae, expected_f, expected_fm, expected_v

    cell = _cell_from_lammps_box(box)
    atype = (type_NiO - 1).tolist()  # LAMMPS 1-based -> deepmd 0-based (Ni=0, O=1)

    script = textwrap.dedent(f"""\
        import json
        import numpy as np
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
    proc = sp.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to compute expected values:\n{proc.stderr}")
    result = json.loads(proc.stdout.strip())

    expected_e = result["e"]
    expected_ae = np.array(result["ae"])
    expected_f = np.array(result["f"])
    # Raw DeepEval force_mag (dE/dspin), scaled by LAMMPS's own
    # spin_norm / hbar unit convention (see the comment on ``_HBAR_METAL``
    # above) before comparison.
    fm_raw = np.array(result["fm"])
    spin_norm = np.linalg.norm(spin, axis=1)
    expected_fm = fm_raw * (spin_norm / _HBAR_METAL)[:, None]
    # Per-atom virial, sign-flipped (LAMMPS convention) relative to DeepPot's
    # atomic virial output (mirrors test_lammps_spin_pt2.py's convention).
    expected_v = -np.array(result["av"])


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    if not pb_file.exists():
        pytest.skip("deeppot_dpa4_spin_graph.pt2 not found")
    _compute_expected()
    write_lmp_data_spin(box, coord, spin, type_NiO, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _lammps(data_file, units="metal") -> PyLammps:
    """Standard DeepSpin LAMMPS system, plus ``atom_modify map yes``.

    Mirrors ``lammps_test_utils.make_spin_lammps`` (not reused directly: it
    does not set ``atom_modify``), with the map turned on -- the native-spin
    DPA4 GRAPH ``.pt2`` needs the LAMMPS atom-map to resolve ghost-atom
    indices to local owners for single-rank inference (same requirement as
    the energy graph route; see ``pair_deepspin.cpp``'s
    ``DeePMD-kit Error: Single-rank LAMMPS .pt2 inference requires
    `atom_modify map yes``` check).
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


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


def _gather_force_mag(lammps: PyLammps, natoms: int) -> np.ndarray:
    """Extract per-atom force_mag in atom-id order.

    LAMMPS does not expose ``fm`` through the legacy ``extract``/
    ``gather_atoms`` registry (see ``run_mpi_pair_deepmd_spin_dpa3_pt2.py``'s
    module docstring), so go via ``compute property/atom fmx fmy fmz`` +
    ``gather`` (id-ordered on every rank, single-rank included).
    """
    fm_global = lammps.lmp.gather("c_fmprop", 1, 3)
    return np.array(fm_global, dtype=np.float64).reshape(natoms, 3)


def test_pair_deepspin(lammps) -> None:
    """Single-rank LAMMPS energy + force + force_mag vs the Python DeepEval
    graph-spin reference (Task 7 path), on the same 4-atom NiO system.
    """
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("fmprop all property/atom fmx fmy fmz")
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(expected_e)

    forces = np.array([lammps.atoms[ii].force for ii in range(4)], dtype=np.float64)
    ids = np.array([lammps.atoms[ii].id for ii in range(4)])
    order = np.argsort(ids)
    forces = forces[order]
    np.testing.assert_allclose(forces, expected_f, atol=1e-8, rtol=0)

    force_mag = _gather_force_mag(lammps, coord.shape[0])
    np.testing.assert_allclose(force_mag, expected_fm, atol=1e-8, rtol=0)
    # Anti-vacuity / native-spin design invariant: force_mag on the two
    # non-spin (O) atoms must be exactly zero, both in the Python reference
    # (baked into expected_fm above) and as produced by LAMMPS.
    np.testing.assert_array_equal(force_mag[2:], np.zeros((2, 3)))

    lammps.run(1)


def test_pair_deepspin_virial(lammps) -> None:
    """Single-rank per-atom pe/pressure/virial via
    ``pe/atom`` / ``pressure`` / ``centroid/stress/atom``, atol=1e-8,
    rtol=1e-8.
    """
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("peatom all pe/atom pair")
    lammps.compute("pressure all pressure NULL pair")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    lammps.variable("eatom atom c_peatom")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"pressure{jj} equal c_pressure[{ii + 1}]")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(expected_e)

    forces = np.array([lammps.atoms[ii].force for ii in range(4)], dtype=np.float64)
    ids = np.array([lammps.atoms[ii].id for ii in range(4)])
    order = np.argsort(ids)
    forces = forces[order]
    np.testing.assert_allclose(forces, expected_f, atol=1e-8, rtol=0)

    idx_map = lammps.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    np.testing.assert_allclose(
        np.array(lammps.variables["eatom"].value),
        expected_ae[idx_map],
        atol=1e-8,
        rtol=1e-8,
    )

    vol = box[1] * box[3] * box[5]
    for ii in range(6):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        pressure_jj = np.array(lammps.variables[f"pressure{jj}"].value) / (
            constants.nktv2p
        )
        expected_pressure_jj = -expected_v[idx_map, jj].sum(axis=0) / vol
        np.testing.assert_allclose(
            pressure_jj, expected_pressure_jj, atol=1e-8, rtol=1e-8
        )
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        virial_jj = np.array(lammps.variables[f"virial{jj}"].value) / (constants.nktv2p)
        np.testing.assert_allclose(
            virial_jj, expected_v[idx_map, jj], atol=1e-8, rtol=1e-8
        )


# ---------------------------------------------------------------------------
# Multi-rank fail-fast (native-spin graph .pt2 has no with-comm artifact at
# all -- has_comm_artifact=false unconditionally, see gen_dpa4_spin.py --
# so any nprocs > 1 run must abort loudly rather than silently produce
# wrong-but-plausible numbers or hang).
# ---------------------------------------------------------------------------


def _run_mpi_subprocess(
    extra_args: list[str] | None = None,
    nprocs: int = 2,
    data_path: Path | None = None,
    processors: str | None = None,
    capture: bool = False,
    timeout: float | None = None,
) -> dict:
    """Invoke the graph-spin MPI runner under ``mpirun -n <nprocs>`` against
    the native-spin DPA4 graph ``.pt2``.

    Copied (module-global closure, not imported) from
    ``test_lammps_dpa4_graph_pt2.py``'s twin. With ``capture=True``, return
    raw subprocess info (``returncode``, ``stdout``, ``stderr``,
    ``timed_out``) -- used by the fail-fast test below; every invocation is
    bounded by ``timeout`` (default ``_MPI_DEFAULT_TIMEOUT``) so a
    should-fail-but-doesn't run cannot hang the suite, and on expiry the
    WHOLE mpirun process group is SIGKILLed.
    """
    if data_path is None:
        data_path = data_file
    if timeout is None:
        timeout = _MPI_DEFAULT_TIMEOUT
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


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepspin_mpi_fails_fast() -> None:
    """A 2-rank MPI run on the native-spin DPA4 graph ``.pt2`` (no
    with-comm artifact, ``has_comm_artifact=false`` unconditionally) must
    ABORT with the Task 9 multi-rank message rather than silently
    succeeding or hanging.

    Mirrors ``test_lammps_dpa4_graph_pt2.py``'s empty-rank fail-fast test:
    assert not-timed-out AND returncode != 0, with a bounded timeout
    (120s) so a regression back into a hang is loud rather than an
    indefinite CI stall.
    """
    out = _run_mpi_subprocess(nprocs=2, capture=True, timeout=120)
    assert not out["timed_out"], (
        "Multi-rank graph-spin run timed out instead of failing promptly: "
        "the has_comm_artifact=false fail-fast guard "
        "(DeepSpinPTExpt::compute) must throw on nprocs > 1 before "
        "any collective communication is attempted."
    )
    assert out["returncode"] != 0, (
        "Expected the multi-rank run on the native-spin DPA4 graph .pt2 "
        "to fail loudly (no with-comm artifact exists for this scheme), "
        "but it exited 0.\n"
        f"stdout:\n{out['stdout'][-2000:]}\nstderr:\n{out['stderr'][-2000:]}"
    )
    combined = out["stdout"] + out["stderr"]
    assert (
        "multi-rank inference is not supported for graph-kind spin .pt2" in combined
    ), (
        "Expected the documented fail-loud message ('multi-rank inference "
        f"is not supported for graph-kind spin .pt2'), got:\n{combined[-2000:]}"
    )
