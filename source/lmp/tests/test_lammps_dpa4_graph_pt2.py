# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with the NeighborGraph (graph-schema) .pt2 DPA4 model.

The model ``deeppot_dpa4_graph.pt2`` is a DPA4/SeZM descriptor exported with
``lower_kind="graph"`` (gen_dpa4.py Section B). DPA4 is graph-native
end-to-end (no dense-only sub-block to gate off, unlike DPA2's
``use_three_body``), so the same config used for the dense ``deeppot_dpa4.pt2``
fixture is graph-eligible; the weights are freshly jittered (see gen_dpa4.py
Section B.1) so the fixture is geometry-sensitive rather than the
architecturally edge-independent output of a fresh, untrained DPA4.

DPA4's SeZM descriptor reads ghost-neighbour features at every interaction
block (``has_message_passing_across_ranks()`` returns ``self.bridging_switch
is None`` -- true for the plain (non-bridging) config exercised here), so
the GRAPH export auto-embeds a nested with-comm AOTI artifact
(``forward_lower_with_comm.pt2``) alongside the plain graph forward, the
same as DPA2's repformer. The DENSE (nlist) ``.pt2`` remains a single
comm-less artifact -- see ``test_lammps_dpa4_pt2.py``.

Single-rank LAMMPS folds ghosts onto local owners (``fold_to_local=True``,
``N == nloc``) and uses the plain graph artifact, exactly as before.
Multi-rank LAMMPS keeps the extended region (``N == nall_real``) and routes
to the with-comm artifact: the C++ ``DeepPotPTExpt`` drives
``deepmd_export::border_op`` once per interaction block to exchange ghost
node/edge features across ranks, masks the fitting reduction to owned nodes
only, and LAMMPS reverse-comm folds the returned per-extended-atom forces
back onto their owners -- the same generic dispatch (``has_message_passing_``
+ ``has_comm_artifact_`` + ``lower_input_kind == "graph"``) that already
serves DPA2, with zero C++ changes required for DPA4 (see
``source/api_cc/src/DeepPotPTExpt.cc``).

Reference values come from ``source/tests/infer/gen_dpa4.py`` (the same
``deeppot_dpa4_graph.expected`` the C++ gtest uses). A second, independent
oracle -- ``deeppot_dpa4_graph_nlist_ref.pt2`` (same weights, dense-nlist
lower, NOT graph) -- is also exercised directly through LAMMPS so a
regression that only breaks the *C++* graph ingestion (not the Python
export path already cross-checked at gen-time) still gets caught.
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
from expected_ref import (
    read_expected_ref,
)
from lammps import (
    PyLammps,
)
from write_lmp_data import (
    write_lmp_data,
)

pb_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa4_graph.pt2"
)
# Independent dense-nlist oracle exported from the SAME (jittered) weights
# (gen_dpa4.py B.1/B.2, lower_kind="nlist"); at non-binding sel graph and
# dense math are equivalent (gen-time cross-check already enforces atomic
# energy / force / total-virial agreement within 1e-8 at the Python level --
# see gen_dpa4.py B.4). Comparing through LAMMPS as well exercises the C++
# graph ingestion path (edge tensors, node atype slicing) independently of
# that Python-level check.
nlist_ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_graph_nlist_ref.pt2"
)
ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa4_graph.expected"
)
# The MPI runner is backend-agnostic (DATAFILE PB_FILE OUTPUT + flags); reuse
# the DPA3 driver verbatim rather than duplicate it (same pattern as
# test_lammps_dpa1_graph_pt2.py / test_lammps_dpa2_graph_pt2.py).
mpi_runner = Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"

# Ceiling for EVERY mpirun invocation (parse mode included): a with-comm
# desync hangs the collective forever, so an unbounded should-succeed
# regression would hang the whole suite.
_MPI_DEFAULT_TIMEOUT = 600.0

# Reference values written by source/tests/infer/gen_dpa4.py (PBC case).
# Guarded with try/except because gen_dpa4.py only runs when PyTorch is built,
# and the graph section is itself skipped under LeakSanitizer (see
# gen_dpa4.py Section B's module comment) -- either way this file must still
# be collectible, with the affected tests skipping cleanly.
try:
    _ref = read_expected_ref(ref_file)["pbc"]
    expected_e = float(np.sum(_ref["expected_e"]))
    expected_f = _ref["expected_f"].reshape(6, 3)
    # LAMMPS uses the opposite sign convention for virial vs DeepPot.
    expected_v = -_ref["expected_v"].reshape(6, 9)
except FileNotFoundError:
    expected_e = expected_f = expected_v = None

_HAS_REF = expected_e is not None

# Same 6-atom water system as the dense DPA4 fixture
# (source/tests/infer/gen_dpa4.py / test_lammps_dpa4_pt2.py): type_map
# [O, H], box 13x13x13.
box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
# Model type_map is ["O", "H"]; gen_dpa4.py atype = [0, 1, 1, 0, 1, 1] ->
# LAMMPS types [1, 2, 2, 1, 2, 2] under identity ``pair_coeff * *``.
type_OH = np.array([1, 2, 2, 1, 2, 2])

data_file = Path(__file__).parent / "data_dpa4_graph_pt2.lmp"
# Wide-box, 3-way x-split variant for the genuinely-empty-rank MPI corner
# (``processors 3 1 1``), same construction as
# ``test_lammps_dpa2_graph_pt2.py``'s ``data_file_empty_rank`` fixture but
# with DPA4's ghost cutoff: rcut(4.0)+skin(2.0)=6.0 (vs dpa2's 8.0). Atoms
# stay in x in [0.25, 12.83] near the left edge of a [0, 90] box. With 3
# even x-slabs of width 30, rank 0 owns [0, 30) (all atoms), rank 2 owns
# [60, 90) (empty of local atoms but picks up a periodic ghost of the
# x~0.25 atoms wrapped around the box's x=90/x=0 seam, since that distance
# ~0.25 is well within the ghost cutoff), and rank 1 (the MIDDLE slab,
# [30, 60)) borders neither the real atoms directly (nearest real atom at
# distance 30-12.83 ~= 17.17 > 6) nor the periodic seam -- so rank 1 is the
# genuinely empty rank (zero owned AND zero ghost atoms) this fixture is
# built to produce.
data_file_empty_rank = Path(__file__).parent / "data_dpa4_graph_pt2_empty_rank.lmp"


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data(box, coord, type_OH, data_file)
    box_empty_rank = np.array([0, 90, 0, 13, 0, 13, 0, 0, 0])
    write_lmp_data(box_empty_rank, coord, type_OH, data_file_empty_rank)


def teardown_module() -> None:
    for f in [data_file, data_file_empty_rank]:
        if f.exists():
            os.remove(f)


def _lammps(data_file, units="metal", atom_map: str = "yes") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
    if atom_map != "no":
        lammps.atom_modify(f"map {atom_map}")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 16")
    lammps.mass("2 2")
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


@pytest.mark.skipif(
    not _HAS_REF, reason="gen_dpa4.py graph .expected fixture not generated"
)
def test_pair_deepmd(lammps) -> None:
    """Single-rank serial run (``atom_modify map yes``): the graph .pt2
    folds ghosts onto local owners (``fold_to_local=True``) and must match
    the gen_dpa4.py reference for energy and per-atom force.
    """
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


@pytest.mark.skipif(
    not _HAS_REF, reason="gen_dpa4.py graph .expected fixture not generated"
)
def test_pair_deepmd_virial(lammps) -> None:
    """Single-rank per-atom virial via ``centroid/stress/atom``."""
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    idx_map = lammps.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    for ii in range(9):
        assert np.array(
            lammps.variables[f"virial{ii}"].value
        ) / constants.nktv2p == pytest.approx(expected_v[idx_map, ii])


@pytest.mark.skipif(
    not nlist_ref_file.exists(),
    reason="gen_dpa4.py deeppot_dpa4_graph_nlist_ref.pt2 fixture not generated",
)
def test_pair_deepmd_graph_matches_nlist_ref() -> None:
    """Single-rank graph .pt2 vs the independent dense-nlist oracle
    (``deeppot_dpa4_graph_nlist_ref.pt2``, same weights, ``lower_kind="nlist"``)
    through LAMMPS, energy/force within 1e-6.

    gen_dpa4.py already cross-checks graph vs nlist at the Python
    ``DeepPot.eval`` level (B.4, 1e-8) at generation time; this test instead
    drives BOTH artifacts through the C++ ``DeepPotPTExpt`` / LAMMPS pair
    style, so a regression confined to the C++ graph-ingestion seam (edge
    tensor construction, node atype slicing) that the Python-level gen-time
    check cannot see is still caught. The per-atom virial is deliberately
    NOT compared here: the graph path assigns each edge's force/virial
    contribution fully to the source atom (edge_force_virial full-to-src), a
    different (equally valid) decomposition than the dense path's -- only
    energy and force (and, at the Python level already, the *total* virial)
    are convention-independent.
    """
    lmp_graph = _lammps(data_file=data_file)
    lmp_graph.pair_style(f"deepmd {pb_file.resolve()}")
    lmp_graph.pair_coeff("* *")
    lmp_graph.run(0)
    e_graph = lmp_graph.eval("pe")
    f_graph = np.array([lmp_graph.atoms[ii].force for ii in range(6)])
    id_graph = [lmp_graph.atoms[ii].id for ii in range(6)]
    lmp_graph.close()

    lmp_nlist = _lammps(data_file=data_file)
    lmp_nlist.pair_style(f"deepmd {nlist_ref_file.resolve()}")
    lmp_nlist.pair_coeff("* *")
    lmp_nlist.run(0)
    e_nlist = lmp_nlist.eval("pe")
    f_nlist = np.array([lmp_nlist.atoms[ii].force for ii in range(6)])
    id_nlist = [lmp_nlist.atoms[ii].id for ii in range(6)]
    lmp_nlist.close()

    # Same data file, single rank -> identical atom-id ordering; assert this
    # rather than silently re-sorting so an ordering change is loud.
    assert id_graph == id_nlist
    assert e_graph == pytest.approx(e_nlist, rel=0, abs=1e-6)
    np.testing.assert_allclose(f_graph, f_nlist, atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# Multi-rank tests (message-passing with-comm graph route).
#
# DPA4's SeZM descriptor participates in per-block ghost exchange, so
# multi-rank LAMMPS routes to the nested with-comm artifact instead of the
# plain graph artifact used above. These tests are the correctness gate for
# that machinery on DPA4, mirroring ``test_lammps_dpa2_graph_pt2.py``'s
# multi-rank section (the SAME generic C++ dispatch; DPA4 required zero C++
# changes, see the module docstring).
# ---------------------------------------------------------------------------


def _run_mpi_subprocess(
    extra_args: list[str] | None = None,
    nprocs: int = 2,
    data_path: Path | None = None,
    processors: str | None = None,
    runner_args: list[str] | None = None,
    pb: Path | None = None,
    capture: bool = False,
    timeout: float | None = None,
) -> dict:
    """Invoke the (backend-agnostic) DPA3 MPI runner under
    ``mpirun -n <nprocs>`` against the dpa4 graph .pt2 and return
    ``{"pe": float, "forces": (n, 3), "virials": (n, 9)}``.

    ``nprocs == 1`` forces ``--processors 1 1 1`` so the C++ side sees
    ``nprocs == 1`` and routes to the plain (single-rank) graph artifact --
    a same-archive reference for the multi-rank comparison.  ``pb``
    overrides the model archive (defaults to ``deeppot_dpa4_graph.pt2``).

    EVERY run is bounded: ``timeout`` (seconds, default
    ``_MPI_DEFAULT_TIMEOUT``) covers the parse path too, so a deadlocked
    collective in a should-succeed regression cannot hang the suite -- on
    expiry the WHOLE mpirun process group is SIGKILLed (killing only mpirun
    can leave orphaned ranks blocking in the collective and holding the
    GPU).  With ``capture=True``, return raw subprocess info
    (``returncode``, ``stdout``, ``stderr``, ``timed_out``) instead of
    parsed output -- used by the fail-fast tests; a timeout there returns
    ``timed_out=True`` with ``returncode=None`` for the caller to assert
    on.  In parse mode a timeout raises ``RuntimeError`` and a nonzero exit
    raises ``subprocess.CalledProcessError`` (matching the old
    ``check_call`` behavior).
    """
    if data_path is None:
        data_path = data_file
    if pb is None:
        pb = pb_file
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
            str(pb.resolve()),
            out_path,
        ]
        if processors is not None:
            argv.extend(["--processors", processors])
        elif nprocs == 1:
            argv.extend(["--processors", "1 1 1"])
        if extra_args:
            argv.extend(extra_args)
        if runner_args:
            argv.extend(runner_args)
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
            # Kill the whole process group: killing only mpirun can
            # leave the deadlocked ranks orphaned (still blocking in
            # the collective and holding the GPU).
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
        forces = rows[:, :3]
        virials = rows[:, 3:]
        return {"pe": pe, "forces": forces, "virials": virials}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa4_graph_matches_single_rank() -> None:
    """Multi-rank (``-n 2``) with-comm graph route must equal single-rank
    (``-n 1``, plain graph artifact) on the SAME archive and trajectory.

    THE gate on the DPA4 with-comm graph machinery: per-block
    ``deepmd_export::border_op`` ghost exchange, the owned-node energy
    mask (extended region includes ghost nodes that must not contribute
    to the reduced energy), and the reverse-comm force fold back onto
    owners. A wrong-but-finite divergence in any of the three would show
    up here even though there is no hardcoded reference value.
    """
    out_mpi = _run_mpi_subprocess(nprocs=2)
    out_ref = _run_mpi_subprocess(nprocs=1)
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    # Same tolerance as test_lammps_dpa2_graph_pt2.py's twin: a relative
    # component absorbs the tiny ordering-dependent floating-point
    # divergence the with-comm route's per-layer ghost exchange plus atomic
    # index_add can introduce on CUDA, without loosening the CPU-exact
    # (bit-reproducible) case.
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=1e-8
    )


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa4_graph_empty_rank_does_not_silently_succeed() -> None:
    """A genuinely empty rank (zero owned AND zero ghost atoms) under the
    message-passing with-comm graph route must NOT silently produce
    wrong-but-plausible numbers.

    DPA4's with-comm route needs every rank to participate in the per-block
    MPI ghost exchange (``border_op``); a rank with zero nodes has nothing
    to export in the traced graph (violates the exported
    ``Dim("n_node_total", min=1)`` and would desync the collective ghost
    exchange across ranks). The C++ side (``DeepPotPTExpt.cc``, the SAME
    model-agnostic guard already exercised by
    ``test_lammps_dpa2_graph_pt2.py``) throws a clear, actionable error on
    the empty rank instead of running.

    The failure is COLLECTIVE and PROMPT: before entering the per-layer
    ``border_op`` collectives, every rank participates in a communicator-
    wide min-reduction of its node count (``deepmd_export::
    allreduce_min_int``), so the non-empty peers detect the empty rank and
    throw the same error instead of blocking forever waiting for it.  A
    timeout is therefore a FAILURE of this test (it would mean the
    preflight regressed back into the historical deadlock), and the
    documented error message must appear on a nonzero exit.

    ``data_file_empty_rank`` (3-way x-split, ``processors 3 1 1``) was
    verified (see the module-level comment above the fixture) to put the
    MIDDLE rank in a genuinely empty state, using DPA4's own ghost cutoff
    (rcut(4.0)+skin(2.0)=6.0, vs dpa2's 8.0).
    """
    out = _run_mpi_subprocess(
        nprocs=3,
        data_path=data_file_empty_rank,
        processors="3 1 1",
        capture=True,
        timeout=120,
    )
    assert not out["timed_out"], (
        "Multi-rank graph run with an empty rank timed out instead of "
        "failing promptly: the collective empty-rank preflight "
        "(allreduce_min_int) must make every rank throw BEFORE the "
        "per-layer border_op collectives."
    )
    assert out["returncode"] != 0, (
        "Expected the multi-rank message-passing graph run to fail loudly "
        "on a genuinely empty rank, but it exited 0.\n"
        f"stdout:\n{out['stdout'][-2000:]}\nstderr:\n{out['stderr'][-2000:]}"
    )
    combined = out["stdout"] + out["stderr"]
    assert "zero owned+ghost atoms" in combined, (
        "Expected the documented fail-loud message ('zero owned+ghost "
        f"atoms'), got:\n{combined[-2000:]}"
    )
