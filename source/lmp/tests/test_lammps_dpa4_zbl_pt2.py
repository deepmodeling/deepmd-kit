# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-rank LAMMPS ``pair_style deepmd`` on the ZBL-BRIDGED DPA4
NeighborGraph (graph-schema) ``.pt2`` (``deeppot_dpa4_zbl_graph.pt2``,
``source/tests/infer/gen_dpa4_zbl.py``).

``bridging_method: ZBL`` builds a COMPOSITION -- ``LinearEnergyModel`` over
``[learned DPA4, InnerPotentialAtomicModel]`` with ``weights="sum"`` -- so
this drives the graph lower of a linear composition through the LAMMPS pair
style.  The archive already had a C++ gtest
(``source/api_cc/tests/test_deeppot_dpa4_zbl_ptexpt.cc``) but NO LAMMPS
coverage at all; the pair style is a distinct consumer (its own nlist/ghost
handling, per-atom virial accumulation and unit conversion), so a regression
confined to it was invisible.

Single-rank only, deliberately
------------------------------
Bridging enables the descriptor's Source Freeze Propagation Gate, whose
per-node ``eta_j = prod_{e: src_e = j} w_e`` folds a node's FULL outgoing-edge
set.  Edges exist only for owned centres, so eta is incomplete on every rank
and the freeze exports NO with-comm artifact (``gen_dpa4_zbl.py`` asserts
``has_comm_artifact is False`` and that no nested
``forward_lower_with_comm.pt2`` entry exists).  There is therefore no
correct multi-rank answer to compare against; what this file pins instead is
that a multi-rank run FAILS LOUDLY rather than silently returning
wrong-but-plausible numbers -- see
``test_pair_deepmd_mpi_dpa4_zbl_fails_fast``.

Reference values are computed LIVE at test-setup time via
``deepmd.infer.DeepPot.eval`` on the archive itself, mirroring
``test_lammps_dpa4_spin_graph_pt2.py``'s ``_compute_expected`` (which explains
the reasoning in full).  Two reasons, both load-bearing here:

- A hardcoded array goes stale the moment DPA4 numerics shift, and this
  fixture's energies are dominated by a ~700 eV analytical term, so a stale
  reference would fail in a way that looks like a bridging bug.
- ``gen_dpa4_zbl.py``'s own ``.expected`` sidecar cannot be reused: its
  evaluation uses a 6x6x6 A cell whose edge length equals DPA4's LAMMPS ghost
  cutoff exactly (rcut(4.0) + skin(2.0) = 6.0), which is not a safe geometry
  for a periodic LAMMPS run.  This module keeps the same 6-atom geometry (the
  0.9 A Ni-Ni pair that makes the ZBL term dominant) in a 13x13x13 A box
  instead.
"""

import importlib.util
import json
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
    write_lmp_data,
)

pb_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_zbl_graph.pt2"
)
data_file = Path(__file__).parent / "data_dpa4_zbl_pt2.lmp"
# The MPI runner is backend-agnostic (DATAFILE PB_FILE OUTPUT + flags); reuse
# the DPA3 driver verbatim rather than duplicate it (same pattern as
# test_lammps_dpa4_graph_pt2.py).  Only the fail-fast test below uses it.
mpi_runner = Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"

# Ceiling for the mpirun invocation.  The fail-fast under test is expected to
# throw on EVERY rank before any collective, so a timeout means the guard
# regressed into a deadlock -- which is a test failure, not a slow machine.
_MPI_DEFAULT_TIMEOUT = 300.0

# 6-atom NiO system, coordinates verbatim from
# ``source/tests/infer/gen_dpa4_zbl.py``'s ``_COORDS``: atoms 0 and 1 sit
# 0.9 A apart, inside ``bridging_r_outer``, so the analytical ZBL term
# contributes a large, unmistakable repulsion (~1.4e3 eV/A forces) rather
# than a numerical afterthought.  The box is 13x13x13 A (NOT the generator's
# 6x6x6 -- see the module docstring).
box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.9, 1.0, 1.0],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ]
)
# Model ``type_map`` is ["Ni", "O"]; gen_dpa4_zbl.py's atype [0,0,0,1,1,1]
# -> LAMMPS types [1,1,1,2,2,2] under identity ``pair_coeff * *``.
type_NiO = np.array([1, 1, 1, 2, 2, 2])

# Reference values, populated by ``_compute_expected`` in ``setup_module``.
expected_e = None
expected_ae = None
expected_f = None
expected_v = None


def _cell_from_lammps_box(lmp_box: np.ndarray) -> np.ndarray:
    """Convert a LAMMPS ``xlo xhi ylo yhi zlo zhi xy xz yz`` box spec to a
    flat, row-major 3x3 cell matrix (deepmd's ``box`` convention).
    """
    xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = lmp_box
    return np.array([xhi - xlo, 0.0, 0.0, xy, yhi - ylo, 0.0, xz, yz, zhi - zlo])


def _compute_expected() -> None:
    """Load ``deeppot_dpa4_zbl_graph.pt2`` via ``DeepPot`` and evaluate this
    module's fixed 6-atom system to obtain the Python reference.

    Runs in a subprocess to avoid importing ``deepmd`` in the LAMMPS test
    process (the LAMMPS plugin already loads ``libdeepmd_op_pt.so`` at the C++
    level, and importing the Python package on top of that can segfault) --
    the same precaution as ``test_lammps_dpa4_spin_graph_pt2.py`` and
    ``test_lammps_model_devi_pt2.py``.
    """
    global expected_e, expected_ae, expected_f, expected_v

    cell = _cell_from_lammps_box(box)
    atype = (type_NiO - 1).tolist()  # LAMMPS 1-based -> deepmd 0-based

    # The archive lives in ``source/tests/infer`` next to ``gen_common.py``,
    # whose ``load_custom_ops()`` loads the build-tree ``libdeepmd_op_pt.so``
    # (registering ``deepmd::edge_force_virial``, which graph ``.pt2``
    # inference needs).  ``import deepmd.pt`` alone only loads the op library
    # from SHARED_LIB_DIR, which the build-test env does not populate.
    infer_dir = str(pb_file.resolve().parent)
    script = (
        "import json, sys\n"
        "import numpy as np\n"
        f"sys.path.insert(0, {infer_dir!r})\n"
        "import deepmd.pt  # noqa: F401  (triggers the base op-library load)\n"
        "from gen_common import load_custom_ops\n"
        "load_custom_ops()\n"
        "from deepmd.infer import DeepPot\n"
        f"dp = DeepPot({str(pb_file.resolve())!r})\n"
        "e, f, v, ae, av = dp.eval(\n"
        f"    np.array({coord.tolist()!r}).reshape(1, -1, 3),\n"
        f"    np.array({cell.tolist()!r}).reshape(1, 9),\n"
        f"    {atype!r},\n"
        "    atomic=True,\n"
        ")\n"
        "print(json.dumps({\n"
        '    "e": float(e[0, 0]),\n'
        '    "ae": np.asarray(ae[0]).reshape(-1).tolist(),\n'
        '    "f": np.asarray(f[0]).tolist(),\n'
        '    "av": np.asarray(av[0]).tolist(),\n'
        "}))\n"
    )
    proc = sp.run([sys.executable, "-c", script], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to compute expected values:\n{proc.stderr}")
    result = json.loads(proc.stdout.strip())

    expected_e = result["e"]
    expected_ae = np.array(result["ae"])
    expected_f = np.array(result["f"])
    # LAMMPS uses the opposite sign convention for the virial vs DeepPot.
    expected_v = -np.array(result["av"])

    # Anti-vacuity, checked once here so every test below is known to compare
    # against a non-degenerate reference: the 0.9 A Ni-Ni pair must drive a
    # large analytical ZBL repulsion.  A fresh (unjittered) DPA4 would give
    # identically zero forces, and a composition that lost its analytical
    # child would give small ones.
    assert np.max(np.abs(expected_f)) > 1e2, (
        "the ZBL-bridged reference forces are too small to be the analytical "
        f"term on a 0.9 A pair (max |f| = {np.max(np.abs(expected_f)):.3e}); "
        "the fixture or the bridging composition is degenerate."
    )


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip("Skip test because PyTorch support is not enabled.")
    if not pb_file.exists():
        pytest.skip(
            "deeppot_dpa4_zbl_graph.pt2 not found (run "
            "source/tests/infer/gen_dpa4_zbl.py)."
        )
    _compute_expected()
    write_lmp_data(box, coord, type_NiO, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _lammps(data_file, units="metal") -> PyLammps:
    """Standard LAMMPS system plus ``atom_modify map yes``.

    DPA4 message-passes within a rank, so the single-rank graph ``.pt2``
    route needs the LAMMPS atom-map to resolve ghost-atom indices to their
    local owners (``DeepPotPTExpt.cc``: "Single-rank LAMMPS .pt2 inference
    requires `atom_modify map yes`").
    """
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
    lammps.atom_modify("map yes")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 58")  # Ni
    lammps.mass("2 16")  # O
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


def test_pair_deepmd(lammps) -> None:
    """Single-rank energy + per-atom force vs the Python DeepEval reference.

    ``rel=1e-10`` on the energy and ``atol=1e-8`` on the force: both sides are
    fp64 and run the SAME compiled artifact, so this is a cross-consumer
    (LAMMPS pair style vs Python DeepEval) check rather than a cross-backend
    one.  The absolute force bound is ~1e-11 relative at this fixture's ~1.4e3
    eV/A magnitude, matching the bound the sibling DPA4 graph LAMMPS tests
    use.
    """
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(expected_e, rel=1e-10)

    ids = np.array([lammps.atoms[ii].id for ii in range(6)])
    forces = np.array([lammps.atoms[ii].force for ii in range(6)])
    np.testing.assert_allclose(forces, expected_f[ids - 1], atol=1e-8, rtol=0)

    # A second MD step: the ZBL repulsion is large but bounded (~0.03 A of
    # displacement at dt = 0.5 fs), so this is a stability smoke test of the
    # per-step dispatch, not a dynamics check.
    lammps.run(1)


def test_pair_deepmd_atom_energy_and_virial(lammps) -> None:
    """Single-rank per-atom energy and per-atom virial.

    ``centroid/stress/atom`` is the pair style's own virial accumulation
    path, which the C++ gtest does not exercise; the atomic energies pin that
    the composition's two per-atom terms are summed per atom rather than only
    in the total.
    """
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("peatom all pe/atom pair")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    lammps.variable("eatom atom c_peatom")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)

    assert lammps.eval("pe") == pytest.approx(expected_e, rel=1e-10)

    idx_map = lammps.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    np.testing.assert_allclose(
        np.array(lammps.variables["eatom"].value),
        expected_ae[idx_map],
        atol=1e-8,
        rtol=0,
    )
    for ii in range(9):
        np.testing.assert_allclose(
            np.array(lammps.variables[f"virial{ii}"].value) / constants.nktv2p,
            expected_v[idx_map, ii],
            atol=1e-8,
            rtol=0,
        )


# ---------------------------------------------------------------------------
# Multi-rank: NOT a correctness test -- a fail-fast test.
# ---------------------------------------------------------------------------


def _run_mpi_subprocess(nprocs: int, processors: str, timeout: float) -> dict:
    """Run the (backend-agnostic) DPA3 MPI runner against the bridged archive
    and return ``{"returncode", "stdout", "stderr", "timed_out"}``.

    Always bounded: on expiry the WHOLE mpirun process group is SIGKILLed
    (killing only mpirun can leave orphaned ranks blocking in a collective).
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        out_path = f.name
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(mpi_runner),
            str(data_file.resolve()),
            str(pb_file.resolve()),
            out_path,
            "--processors",
            processors,
        ]
        proc = sp.Popen(
            argv, stdout=sp.PIPE, stderr=sp.PIPE, text=True, start_new_session=True
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except sp.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            return {
                "returncode": None,
                "stdout": stdout or "",
                "stderr": stderr or "",
                "timed_out": True,
            }
        return {
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
        }
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa4_zbl_fails_fast() -> None:
    """A multi-rank run of a BRIDGED archive must fail loudly, not answer.

    The bridged model is single-rank only by construction (see the module
    docstring), so its freeze exports no with-comm artifact while still
    declaring ``has_message_passing``.  ``DeepPotPTExpt::compute_inner``'s
    dispatch reads exactly that combination -- graph lower + ``nprocs > 1`` +
    message passing + no with-comm artifact -- and throws before building any
    tensors.  Without the guard the run would fall through to the plain
    single-rank artifact on a per-rank subdomain, where the bridging gate's
    per-node eta is incomplete: wrong, finite, plausible numbers.

    The failure is uniform across ranks (every rank evaluates the same
    metadata-only predicate before any collective), so a TIMEOUT is a failure
    of this test: it would mean the guard regressed into a deadlock.

    This is deliberately the ONLY multi-rank test in this file; there is no
    correct multi-rank reference for a bridged model to compare against.
    """
    out = _run_mpi_subprocess(
        nprocs=2, processors="2 1 1", timeout=_MPI_DEFAULT_TIMEOUT
    )
    assert not out["timed_out"], (
        "Multi-rank run of the bridged archive timed out instead of failing "
        "promptly; the dispatch guard must throw on every rank BEFORE any "
        "collective."
    )
    assert out["returncode"] != 0, (
        "Expected the multi-rank run of a bridged (no with-comm artifact) "
        "archive to fail loudly, but it exited 0.\n"
        f"stdout:\n{out['stdout'][-2000:]}\nstderr:\n{out['stderr'][-2000:]}"
    )
    combined = out["stdout"] + out["stderr"]
    assert "with-comm artifact" in combined, (
        "Expected the documented fail-loud message (mentioning the missing "
        f"'with-comm artifact'), got:\n{combined[-2000:]}"
    )
