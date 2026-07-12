# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with the NeighborGraph (graph-schema) .pt2 DPA1 model.

The model ``deeppot_dpa1_graph.pt2`` is a dpa1(attn_layer=0) descriptor
exported with ``lower_kind="graph"`` (gen_dpa1.py section B).  dpa1 is
NON-message-passing, so the SAME single-rank graph .pt2 also drives the
multi-rank path: the C++ ``DeepPotPTExpt`` builds an EXTENDED-region graph
(``fold_to_local=False``; ghosts are distinct nodes whose features come from
their real halo types) and returns per-extended-atom forces, which LAMMPS
reverse-comm folds back to their owners.  There is NO with-comm artifact and
NO ``border_op`` (that is the message-passing PR-G path) — hence no
``use_loc_mapping=False`` variant.

Reference values come from ``source/tests/infer/gen_dpa1.py`` (the same
``deeppot_dpa1_graph.expected`` the C++ gtest uses); the multi-rank run must
match the single-rank reference for energy, per-atom force, and per-atom
virial.  This is the core multi-rank correctness gate for the non-MP graph
path implemented in B3.1.
"""

import importlib.util
import os
import shutil
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
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa1_graph.pt2"
)
# Graph-lower dpa1 with model-level pair_exclude_types=[[0,1]] and its
# same-weights no-exclusion baseline (source/tests/infer/gen_dpa1_pairexcl.py).
# Used to prove exclusion survives the extended-region multi-rank graph path.
pb_file_pairexcl = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa1_pairexcl_graph.pt2"
)
pb_file_pairexcl_none = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa1_pairexcl_none.pt2"
)
ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa1_graph.expected"
)
# The MPI runner is backend-agnostic (DATAFILE PB_FILE OUTPUT + flags); reuse
# the DPA3 driver verbatim rather than duplicate it.
mpi_runner = Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"

data_file = Path(__file__).parent / "data_dpa1_graph_pt2.lmp"
# Elongated-box variant for the empty-subdomain MPI corner: x extended to
# 30 A while atoms stay in x in [0.25, 12.83]; with ``processors 2 1 1`` the
# split at x = 15 leaves rank 1 with zero local atoms.
data_file_empty_subdomain = (
    Path(__file__).parent / "data_dpa1_graph_pt2_empty_subdomain.lmp"
)

# Reference values written by source/tests/infer/gen_dpa1.py (PBC case).
# Guarded with try/except because gen_dpa1.py only runs when PyTorch is built.
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
# gen_dpa1.py has not run (e.g. PyTorch not built). The MPI multi-rank tests
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
    box_empty_subdomain = np.array([0, 30, 0, 13, 0, 13, 0, 0, 0])
    write_lmp_data(box_empty_subdomain, coord, type_OH, data_file_empty_subdomain)


def teardown_module() -> None:
    for f in [data_file, data_file_empty_subdomain]:
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


@pytest.mark.skipif(not _HAS_REF, reason="gen_dpa1.py .expected fixture not generated")
def test_pair_deepmd(lammps) -> None:
    """Single-rank serial run (``atom_modify map yes``): the graph .pt2
    folds ghosts onto local owners (``fold_to_local=True``) and must match
    the gen_dpa1.py reference for energy and per-atom force.
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


@pytest.mark.skipif(not _HAS_REF, reason="gen_dpa1.py .expected fixture not generated")
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


# ---------------------------------------------------------------------------
# Multi-rank test (non-MP extended-region graph path; B3.1).
#
# dpa1 is non-message-passing, so multi-rank uses the SAME single-rank graph
# .pt2 on the extended region.  The expected energy/force/virial are the
# single-rank reference: each rank evaluates its local atoms over the extended
# graph; ghost reaction forces fold back via LAMMPS reverse-comm.
# ---------------------------------------------------------------------------


def _run_mpi_subprocess(
    extra_args: list[str] | None = None,
    nprocs: int = 2,
    data_path: Path | None = None,
    processors: str | None = None,
    runner_args: list[str] | None = None,
    pb: Path | None = None,
) -> dict:
    """Invoke the (backend-agnostic) DPA3 MPI runner under
    ``mpirun -n <nprocs>`` against the dpa1 graph .pt2 and return
    ``{"pe": float, "forces": (n, 3), "virials": (n, 9)}``.

    ``nprocs == 1`` forces ``--processors 1 1 1`` so the C++ side sees
    ``nprocs == 1`` and routes to the single-rank graph path — a
    same-archive reference for the multi-rank comparison.  ``pb`` overrides
    the model archive (defaults to the no-exclusion ``deeppot_dpa1_graph.pt2``).
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
def test_pair_deepmd_mpi_dpa1_graph() -> None:
    """Multi-rank LAMMPS run for the dpa1 graph .pt2 must match the
    single-rank reference within numerical tolerance for energy, forces,
    and per-atom virial.

    This is the core correctness gate for the non-MP extended-region
    multi-rank C++ path (B3.1): the extended graph + reverse-comm
    fold-back must reproduce the folded single-rank result.
    """
    out = _run_mpi_subprocess()
    assert out["pe"] == pytest.approx(expected_e, rel=0, abs=1e-8)
    for ii in range(6):
        np.testing.assert_allclose(out["forces"][ii], expected_f[ii], atol=1e-8, rtol=0)
    # ``centroid/stress/atom`` column order [xx, yy, zz, xy, xz, yz, yx, zx,
    # zy]; the inverse permutation maps it back to the expected_v columns.
    expected_v_to_lammps = [0, 6, 7, 3, 1, 8, 4, 5, 2]
    np.testing.assert_allclose(
        out["virials"][:, expected_v_to_lammps] / constants.nktv2p,
        expected_v,
        atol=1e-8,
        rtol=0,
    )


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa1_graph_matches_single_rank() -> None:
    """Multi-rank (``-n 2``) ≡ single-rank (``-n 1``) on the SAME archive
    and trajectory — isolates the extended-region multi-rank C++ path from
    the .pt2 reference values (a wrong-but-finite divergence would show up
    here even if the hardcoded reference drifted).
    """
    out_mpi = _run_mpi_subprocess(nprocs=2)
    out_ref = _run_mpi_subprocess(nprocs=1)
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=0
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa1_graph_empty_subdomain() -> None:
    """Multi-rank with one rank owning zero local atoms (elongated box,
    ``processors 2 1 1``, split at x = 15).  The extended-region graph path
    must still produce correct forces/virial on the populated rank and a
    zero contribution from the empty rank — compared against a same-archive
    single-rank reference of the same fixture.
    """
    # Force ``processors 2 1 1`` so the split is along x at 15 and rank 1 is
    # genuinely empty -- otherwise LAMMPS may auto-pick a grid where neither
    # rank is empty and the branch under test is not exercised.
    out_mpi = _run_mpi_subprocess(
        nprocs=2, data_path=data_file_empty_subdomain, processors="2 1 1"
    )
    out_ref = _run_mpi_subprocess(nprocs=1, data_path=data_file_empty_subdomain)
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=0
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
@pytest.mark.skipif(
    not pb_file_pairexcl.exists(),
    reason="gen_dpa1_pairexcl.py .pt2 fixtures not generated",
)
def test_pair_deepmd_mpi_dpa1_pairexcl_graph_matches_single_rank() -> None:
    """Model-level ``pair_exclude_types`` must survive the extended-region
    multi-rank graph path (cell 5).

    Exclusion is a BUILD-time transform applied at the C++ ingestion seam
    (``applyPairExclusion`` on ``edge_mask``); the SAME seam serves the
    single-rank (``n_node == nloc``) and multi-rank (``n_node == nall_real``,
    extended region) graph routes, so multi-rank must equal single-rank on the
    excluded model. A regression that skipped the seam on the extended-region
    path -- or fed it the wrong (local vs extended) atypes -- would diverge here.

    Two checks:
      1. MP (``-n 2``) ≡ SP (``-n 1``) on the excluded archive.
      2. The excluded run differs from the SAME-weights no-exclusion baseline
         (``deeppot_dpa1_pairexcl_none.pt2``), so a silently-dropped exclusion
         on BOTH ranks cannot pass check 1 trivially.
    """
    out_mpi = _run_mpi_subprocess(nprocs=2, pb=pb_file_pairexcl)
    out_ref = _run_mpi_subprocess(nprocs=1, pb=pb_file_pairexcl)
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=0
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)

    # Exclusion must be ACTIVE on the multi-rank path: the excluded energy must
    # differ from the same-weights no-exclusion baseline (O-H pairs dropped).
    out_none = _run_mpi_subprocess(nprocs=2, pb=pb_file_pairexcl_none)
    assert abs(out_mpi["pe"] - out_none["pe"]) > 1e-6, (
        "pair_exclude_types had no effect on the multi-rank graph path "
        f"(|E_excl - E_none| = {abs(out_mpi['pe'] - out_none['pe']):.2e})"
    )
