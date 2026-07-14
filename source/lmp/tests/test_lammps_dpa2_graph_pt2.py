# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with the NeighborGraph (graph-schema) .pt2 DPA2 model.

The model ``deeppot_dpa2_graph.pt2`` is a dpa2 descriptor (repformer
message-passing, ``use_three_body=False`` so the graph-ineligible
three-body repinit sub-block is off) exported with ``lower_kind="graph"``
(gen_dpa2.py section B).  Unlike dpa1 (non-message-passing), dpa2's
repformer block exchanges ghost-atom features every layer, so the graph
export automatically embeds a nested with-comm AOTI artifact
(``forward_lower_with_comm.pt2``) alongside the plain graph forward.

Single-rank LAMMPS folds ghosts onto local owners (``fold_to_local=True``,
``N == nloc``) and uses the plain graph artifact.  Multi-rank LAMMPS keeps
the extended region (``N == nall_real``) and routes to the with-comm
artifact: the C++ ``DeepPotPTExpt`` drives ``deepmd_export::border_op``
once per repformer layer to exchange ghost node/edge features across
ranks, masks the fitting reduction to owned nodes only (an owned-atom
energy mask, since the extended region includes ghost nodes that must
not double-count energy), and LAMMPS reverse-comm folds the returned
per-extended-atom forces back onto their owners.  This is the first live
exercise of the whole with-comm graph chain (per-layer border_op +
owned-energy mask + reverse force fold) end-to-end through real MPI.

Reference values come from ``source/tests/infer/gen_dpa2.py`` (the same
``deeppot_dpa2_graph.expected`` the C++ gtest uses).  A second, independent
oracle -- ``deeppot_dpa2_graph_nlist_ref.pt2`` (same weights, dense-nlist
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
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa2_graph.pt2"
)
# Independent dense-nlist oracle exported from the SAME weights
# (gen_dpa2.py B.2, lower_kind="nlist"); at non-binding sel graph and dense
# math are equivalent (gen-time cross-check already enforces atomic energy /
# force / total-virial agreement within 1e-8 at the Python level -- see
# gen_dpa2.py B.4). Comparing through LAMMPS as well exercises the C++ graph
# ingestion path (edge tensors, node atype slicing, pair-exclusion seam)
# independently of that Python-level check.
nlist_ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa2_graph_nlist_ref.pt2"
)
ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa2_graph.expected"
)
# The MPI runner is backend-agnostic (DATAFILE PB_FILE OUTPUT + flags); reuse
# the DPA3 driver verbatim rather than duplicate it (same pattern as
# test_lammps_dpa1_graph_pt2.py).
mpi_runner = Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"

data_file = Path(__file__).parent / "data_dpa2_graph_pt2.lmp"
# Wide-box, 3-way x-split variant for the genuinely-empty-rank MPI corner
# (``processors 3 1 1``): atoms stay in x in [0.25, 12.83] near the left
# edge of a [0, 90] box.  With 3 even x-slabs of width 30, rank 0 owns
# [0, 30) (all atoms), rank 2 owns [60, 90) (empty of local atoms but
# picks up a periodic ghost of the x~0.25 atoms wrapped around the box's
# x=90/x=0 seam, since that distance ~0.25 is well within the ghost cutoff
# rcut(6.0)+skin(2.0)=8.0), and rank 1 (the MIDDLE slab, [30, 60)) borders
# neither the real atoms directly (nearest real atom at distance 30-12.83
# ~= 17.17 > 8) nor the periodic seam -- so rank 1 is the genuinely empty
# rank (zero owned AND zero ghost atoms) this fixture is built to produce.
# A naive dpa1-style 2-way split (single periodic seam shared by both
# ranks) cannot achieve this: the "empty" rank always picks up a
# wrapped-around ghost of the near-edge atoms (see
# test_lammps_dpa1_graph_pt2.py's empty-subdomain comment) since with only
# two ranks in a periodic dimension the same pair of ranks borders on
# both sides.
data_file_empty_rank = Path(__file__).parent / "data_dpa2_graph_pt2_empty_rank.lmp"

# Reference values written by source/tests/infer/gen_dpa2.py (PBC case).
# Guarded with try/except because gen_dpa2.py only runs when PyTorch is built.
try:
    _ref = read_expected_ref(ref_file)["pbc"]
    expected_e = float(np.sum(_ref["expected_e"]))
    expected_f = _ref["expected_f"].reshape(6, 3)
    # LAMMPS uses the opposite sign convention for virial vs DeepPot.
    expected_v = -_ref["expected_v"].reshape(6, 9)
except FileNotFoundError:
    expected_e = expected_f = expected_v = None

# Gate the reference-comparison tests on the generated ``.expected`` fixture so
# they skip cleanly (rather than failing with a ``TypeError`` on ``None``) when
# gen_dpa2.py has not run (e.g. PyTorch not built). The MPI multi-rank tests
# compare against a single-rank run of the same archive and do not need it.
_HAS_REF = expected_e is not None

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
# Model type_map is ["O", "H"]; gtest atype = [0, 1, 1, 0, 1, 1] -> LAMMPS
# types [1, 2, 2, 1, 2, 2] under identity ``pair_coeff * *``.
type_OH = np.array([1, 2, 2, 1, 2, 2])


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


@pytest.mark.skipif(not _HAS_REF, reason="gen_dpa2.py .expected fixture not generated")
def test_pair_deepmd(lammps) -> None:
    """Single-rank serial run (``atom_modify map yes``): the graph .pt2
    folds ghosts onto local owners (``fold_to_local=True``) and must match
    the gen_dpa2.py reference for energy and per-atom force.
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


@pytest.mark.skipif(not _HAS_REF, reason="gen_dpa2.py .expected fixture not generated")
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
    reason="gen_dpa2.py deeppot_dpa2_graph_nlist_ref.pt2 fixture not generated",
)
def test_pair_deepmd_graph_matches_nlist_ref() -> None:
    """Single-rank graph .pt2 vs the independent dense-nlist oracle
    (``deeppot_dpa2_graph_nlist_ref.pt2``, same weights, ``lower_kind="nlist"``)
    through LAMMPS, energy/force within 1e-6.

    gen_dpa2.py already cross-checks graph vs nlist at the Python
    ``DeepPot.eval`` level (B.4, 1e-8) at generation time; this test instead
    drives BOTH artifacts through the C++ ``DeepPotPTExpt`` / LAMMPS pair
    style, so a regression confined to the C++ graph-ingestion seam (edge
    tensor construction, node atype slicing, pair-exclusion application)
    that the Python-level gen-time check cannot see is still caught. The
    per-atom virial is deliberately NOT compared here: the graph path
    assigns each edge's virial contribution fully to the source atom, a
    different (equally valid) decomposition than the dense path's for a
    message-passing model -- only energy and force (and, at the Python
    level already, the *total* virial) are convention-independent.
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
# dpa2's repformer participates in per-layer ghost exchange, so multi-rank
# LAMMPS routes to the nested with-comm artifact instead of the plain graph
# artifact used above and by dpa1's (non-MP) multi-rank path.  These tests
# are the correctness gate for that new machinery: per-layer border_op,
# owned-energy mask, and reverse force fold.
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
    ``mpirun -n <nprocs>`` against the dpa2 graph .pt2 and return
    ``{"pe": float, "forces": (n, 3), "virials": (n, 9)}``.

    ``nprocs == 1`` forces ``--processors 1 1 1`` so the C++ side sees
    ``nprocs == 1`` and routes to the plain (single-rank) graph artifact --
    a same-archive reference for the multi-rank comparison.  ``pb``
    overrides the model archive (defaults to ``deeppot_dpa2_graph.pt2``).

    With ``capture=True``, return raw subprocess info (``returncode``,
    ``stdout``, ``stderr``, ``timed_out``) instead of parsed output -- used
    by the empty-rank test.  ``timeout`` (seconds, capture mode only) bounds
    the run: on expiry the WHOLE mpirun process group is SIGKILLed (a
    deadlocked collective otherwise leaves orphaned ranks holding the GPU)
    and ``timed_out=True`` is returned with ``returncode=None``.
    """
    if data_path is None:
        data_path = data_file
    if pb is None:
        pb = pb_file
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
        if capture:
            proc = sp.Popen(
                argv,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
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
        sp.check_call(argv)
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
def test_pair_deepmd_mpi_dpa2_graph_matches_single_rank() -> None:
    """Multi-rank (``-n 2``) with-comm graph route must equal single-rank
    (``-n 1``, plain graph artifact) on the SAME archive and trajectory.

    THE gate on the new with-comm graph machinery: per-layer
    ``deepmd_export::border_op`` ghost exchange, the owned-node energy
    mask (extended region includes ghost nodes that must not contribute
    to the reduced energy), and the reverse-comm force fold back onto
    owners.  A wrong-but-finite divergence in any of the three would show
    up here even though there is no hardcoded reference value.
    """
    out_mpi = _run_mpi_subprocess(nprocs=2)
    out_ref = _run_mpi_subprocess(nprocs=1)
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    # dpa2 per-atom virial components reach magnitudes ~3.5e3 (unlike dpa1,
    # whose small magnitudes let the copied atol=1e-8, rtol=0 criterion pass);
    # on CUDA the with-comm route's per-layer ghost exchange plus atomic
    # index_add reorders the edge-virial summation, giving an observed max
    # abs diff ~1e-6 = 6.9e-10 RELATIVE (10 significant digits agree; forces
    # and energy match, and the energy check above already uses rel=1e-8).
    # An absolute-only tolerance cannot absorb that at these magnitudes, so
    # allow the same 1e-8 relative slack as the energy check.
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=1e-8
    )


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa2_graph_empty_rank_does_not_silently_succeed() -> None:
    """A genuinely empty rank (zero owned AND zero ghost atoms) under the
    message-passing with-comm graph route must NOT silently produce
    wrong-but-plausible numbers.

    Design note (departs from ``test_lammps_dpa1_graph_pt2.py``'s
    ``..._empty_subdomain`` test, which asserts the empty-rank run MATCHES
    the single-rank reference): dpa1 is non-message-passing, so its
    "empty" rank still runs the plain graph artifact over its (non-empty)
    ghost region.  dpa2's with-comm route instead needs every rank to
    participate in the per-layer MPI halo exchange (``border_op``); a rank
    with zero nodes has nothing to export in the traced graph (violates
    the exported ``Dim("n_node_total", min=1)`` and would desync the
    collective halo exchange across ranks). The C++ side
    (``DeepPotPTExpt.cc``, guard added alongside the with-comm graph route)
    throws a clear, actionable error on the empty rank instead of running.

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
    MIDDLE rank in a genuinely empty state -- unlike a naive 2-way
    dpa1-style split, whose "empty" rank always picks up a ghost wrapped
    around the periodic seam.
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
