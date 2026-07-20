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

Reference (energy / force / force_mag / virial) values below were computed
once via ``deepmd.infer.DeepPot.eval`` on ``deeppot_dpa4_spin_graph.pt2``
for the fixed 4-atom NiO system reused from ``test_lammps_spin_pt2.py``
(box 13x13x13, same coordinates/type ordering: 2 spin-active Ni + 2
non-magnetic O) -- i.e. exactly the Task 7 graph-spin Python eval path,
driven here through LAMMPS instead.
"""

import importlib.util
import os
import shutil
import signal
import subprocess as sp
import sys
import tempfile
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

# Reference values from deepmd.infer.DeepPot.eval(..., spin=spin) on
# deeppot_dpa4_spin_graph.pt2 for the system above (computed once offline;
# see the module docstring). expected_av / expected_v mirror
# test_lammps_spin_pt2.py's sign convention: LAMMPS reports the *negative*
# of DeepPot's atomic virial.
#
# Regeneration recipe (run once, offline, from the repo root, after
# ``python source/tests/infer/gen_dpa4_spin.py`` has produced
# ``source/tests/infer/deeppot_dpa4_spin_graph.pt2``):
#
#   import numpy as np
#   from deepmd.infer import DeepPot
#   dp = DeepPot("source/tests/infer/deeppot_dpa4_spin_graph.pt2")
#   e, f, v, ae, av, fm, mm = dp.eval(
#       coord.reshape(1, -1, 3),
#       box.reshape(1, 9) if box is not None else None,
#       type_NiO - 1,  # LAMMPS 1-based type -> deepmd 0-based atype (Ni=0, O=1)
#       atomic=True,
#       spin=spin.reshape(1, -1, 3),
#   )
#
# using the ``box``/``coord``/``spin``/``type_NiO`` arrays defined above
# (this module's fixed 4-atom NiO system). ``e``/``f``/``ae`` map directly to
# ``expected_e``/``expected_f``/``expected_ae`` below (reshaped/squeezed to
# drop the leading frame axis); ``av`` (NOT ``v`` -- the per-atom virial,
# since ``expected_v`` below is shape ``(4, 9)``) is sign-flipped to give
# ``expected_v``, per the LAMMPS-vs-DeepPot atomic-virial sign convention
# noted above. ``fm`` is the RAW ``dE/dspin`` -- exactly ``_expected_fm_raw``
# below, BEFORE the ``spin_norm / hbar`` scaling applied further down to
# produce ``expected_fm`` (the value actually compared against LAMMPS's own
# ``fm`` output; see the comment on that scaling below). ``mm`` (mask_mag) is
# unused here -- the fixed system's spin-active/non-magnetic split is
# already known (2 Ni + 2 O) and hardcoded via ``_spin_norm`` below.
expected_e = 2.3446106979205501e00
expected_ae = np.array(
    [
        5.6007587197408659e-01,
        4.0092286275099903e-01,
        6.9179401747409552e-01,
        6.9181794572136912e-01,
    ]
)
expected_f = np.array(
    [
        [-1.3324305442433643e-01, 5.3499008613075223e-02, -2.1571616033805491e-01],
        [1.2792742332316817e-01, -5.3518911559650446e-02, 2.1519707821828571e-01],
        [1.0822753105286880e00, 1.0065026530991443e00, -1.4738926455238310e00],
        [-1.0769596794275196e00, -1.0064827501525693e00, 1.4744117276436002e00],
    ]
)
# Raw DeepEval force_mag (dE/dspin), BEFORE LAMMPS's own spin-dynamics unit
# convention is applied.
_expected_fm_raw = np.array(
    [
        [2.0524302733427091e-02, -8.4015035747638245e-03, 3.8224710881061427e-02],
        [-2.0164441845905290e-01, 8.4351600022754797e-02, 1.8298077837080096e-01],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
    ]
)
# LAMMPS's ``fm`` (what ``compute property/atom fmx fmy fmz`` reports) is
# NOT the raw DeepEval force_mag: pair_deepspin.cpp scales it by
# ``spin_norm / hbar`` per atom (metal-units ``hbar = 6.5821191e-04``, see
# ``source/lmp/pair_deepspin.cpp:531,535`` -- same convention already
# implicit, if untested against a raw LAMMPS ``fm`` read, in
# test_lammps_spin_pt2.py). ``spin_norm`` is 0 for the two non-magnetic O
# atoms, so the scaling is a no-op there (0 stays 0).
_HBAR_METAL = 6.5821191e-04
_spin_norm = np.linalg.norm(spin, axis=1)
expected_fm = _expected_fm_raw * (_spin_norm / _HBAR_METAL)[:, None]
# Per-atom virial, sign-flipped (LAMMPS convention) relative to DeepPot's
# atomic virial output.
expected_v = -np.array(
    [
        -4.2918792055893495e-03,
        8.3110521134303061e-03,
        1.7241815457408119e-02,
        1.1926757510913758e-04,
        -2.2730896013015287e-05,
        -3.1745300418736588e-05,
        1.1142878725624991e-01,
        -4.6116975570851690e-02,
        -8.3115587171492200e-02,
        -7.4305849714402294e-02,
        3.1128126231708988e-02,
        5.6231453837925868e-02,
        3.9551912767818186e-02,
        -1.6569044537869740e-02,
        -2.9931177229700148e-02,
        -2.6928649990912962e-01,
        1.1280920942139185e-01,
        2.0378437830961091e-01,
        -4.0583473930473013e-01,
        -3.8244570231040009e-01,
        5.6053129139996127e-01,
        -3.8220789092383661e-01,
        -3.5706837580311901e-01,
        5.2303030431741482e-01,
        5.6081442964078232e-01,
        5.2342390128425265e-01,
        -7.6665623649745862e-01,
        -4.0916165894703893e-01,
        -3.8224312875315491e-01,
        5.5990542803278998e-01,
        -3.8271294213750534e-01,
        -3.5753445910214326e-01,
        5.2371244713553355e-01,
        5.6026058034045267e-01,
        5.2340133163384406e-01,
        -7.6667237309746128e-01,
    ]
).reshape(4, 9)


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
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
