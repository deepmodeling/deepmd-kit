# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-rank LAMMPS ``pair_style deepspin`` on the native-spin DPA4
NeighborGraph (graph-schema) ``.pt2`` (Task 8's ``deeppot_dpa4_spin_graph.pt2``).

Unlike the virtual-atom ``spin_ener`` scheme exercised by
``test_lammps_spin_pt2.py``, native-spin DPA4 has NO dense/nlist lower at
all -- spin rides the NeighborGraph lower exclusively (see
``deepmd/pt_expt/model/dpa4_native_spin_model.py``'s module docstring and
``source/tests/infer/gen_dpa4_spin.py``). The fixture also carries the
nested with-comm AOTI artifact, so multi-rank LAMMPS drives a real
domain-decomposed run through ``DeepSpinPTExpt::run_model_graph_with_comm``:
per-block ghost FEATURE refresh via ``border_op``, ghost SPINS via the
LAMMPS ``sp`` forward-comm (see ``source/api_cc/src/DeepSpinPTExpt.cc``).

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
periodic run. Instead, ``dpa4_spin_harness.compute_expected`` loads the archive and
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
import os
import shutil
from pathlib import (
    Path,
)

import numpy as np
import pytest
from dpa4_spin_harness import (
    BOX,
    COORD,
    SPIN,
    TYPE_NIO,
    assert_mpi_matches_single_rank,
    check_single_rank_energy_force,
    check_single_rank_virial,
    compute_expected,
    make_lammps,
    run_mpi_spin_runner,
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
# Wide-box, 3-way x-split variant for the OWNED-EMPTY-rank MPI corner
# (``processors 3 1 1``), adapted from ``test_lammps_dpa4_graph_pt2.py``'s
# ``data_file_empty_rank`` construction to this file's 4-atom spin system
# (issue #5906 Task 12b) -- with one load-bearing difference.  The DeepSpin
# phantom path (PR #5485, ``DeepSpinPTExpt.cc``) engages when a rank owns
# ZERO local atoms but still holds ghosts (``nloc_real == 0 &&
# nall_real > 0``); a GENUINELY empty rank (zero owned AND zero ghost) is
# instead rejected by the same collective fail-fast preflight as the energy
# route (``DeepSpinPTExpt.cc``'s "zero owned+ghost atoms" throw), so the
# genuinely-empty construction of the energy twin cannot exercise the
# phantom logic.  This fixture therefore shifts the Ni pair to
# x ~= 26.8/26.1, within DPA4's ghost cutoff (rcut(4.0)+skin(2.0)=6.0) of
# the x=30 slab boundary of a [0, 90] box.  With 3 even x-slabs of width
# 30: rank 0 owns [0, 30) (all 4 atoms), rank 1 ([30, 60)) owns NOTHING but
# receives ghosts of the two Ni atoms (3.2 and 3.9 from its lower
# boundary), and rank 2 ([60, 90)) owns nothing but receives periodic
# ghosts of the x < 6 O atoms (x=3.51 and x=4.27, wrapped around the box's
# x=90/x=0 seam) -- BOTH atom-less ranks carry ghosts, so both take the
# phantom path and none trips the genuinely-empty fail-fast.  The shifted
# coordinates (``coord_empty_rank``) are defined below, after ``coord``.
data_file_empty_rank = Path(__file__).parent / "data_dpa4_spin_graph_pt2_empty_rank.lmp"
# The MPI runner is graph-spin-specific (no aparam / no NULL-type
# extras, unlike run_mpi_pair_deepmd_spin_dpa3_pt2.py's virtual-atom-scheme
# runner): the native-spin DPA4 fixture takes no fparam/aparam.
mpi_runner = Path(__file__).parent / "run_mpi_pair_deepmd_spin_graph_dpa4_pt2.py"

# The shared 4-atom NiO system (dpa4_spin_harness).
box, coord, spin, type_NiO = BOX, COORD, SPIN, TYPE_NIO

# Owned-empty-rank variant of ``coord`` (see the comment above
# ``data_file_empty_rank``): the two Ni atoms shift to x ~= 26.8/26.1 so
# rank 1 of the ``processors 3 1 1`` split owns nothing but holds their
# ghosts; the O atoms stay at x < 6 so rank 2 holds their periodic-seam
# ghosts.  The Ni-Ni and O-O pair geometries are internally unchanged
# (rigid x-shift of the Ni pair only), and in the wide [0, 90] box the two
# pairs sit far beyond rcut(4.0) of each other with or without the shift,
# so each pair still interacts internally and the system stays
# non-degenerate for the anti-vacuity checks below.
_EMPTY_RANK_NI_X_SHIFT = 14.0
coord_empty_rank = coord.copy()
coord_empty_rank[:2, 0] += _EMPTY_RANK_NI_X_SHIFT

# Reference values, populated by ``compute_expected`` in ``setup_module`` --
# see the module docstring for why these are computed live via a DeepPot
# subprocess call rather than hardcoded or read from a sidecar file.
expected: dict = {}


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    if not pb_file.exists():
        pytest.skip("deeppot_dpa4_spin_graph.pt2 not found")
    expected.update(compute_expected(pb_file))
    write_lmp_data_spin(box, coord, spin, type_NiO, data_file)
    box_empty_rank = np.array([0, 90, 0, 13, 0, 13, 0, 0, 0])
    write_lmp_data_spin(
        box_empty_rank, coord_empty_rank, spin, type_NiO, data_file_empty_rank
    )


def teardown_module() -> None:
    for path in (data_file, data_file_empty_rank):
        if path.exists():
            os.remove(path)


@pytest.fixture
def lammps():
    lmp = make_lammps(data_file)
    yield lmp
    lmp.close()


def test_pair_deepspin(lammps) -> None:
    """Single-rank LAMMPS energy + force + force_mag vs the Python
    graph-spin DeepEval reference (Task 7 path).
    """
    check_single_rank_energy_force(lammps, pb_file, expected)


def test_pair_deepspin_virial(lammps) -> None:
    """Single-rank per-atom pe/pressure/virial on the native-spin archive."""
    check_single_rank_virial(lammps, pb_file, expected)


def _run_mpi(data_path: Path, nprocs: int, processors: str) -> dict:
    """This module's binding of the shared runner (archive + runner fixed)."""
    return run_mpi_spin_runner(
        mpi_runner, pb_file, data_path, nprocs=nprocs, processors=processors
    )


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepspin_mpi_matches_single_rank() -> None:
    """A 2-rank MPI run must reproduce the 1-rank result on the SAME archive.

    The native-spin DPA4 graph ``.pt2`` now carries the nested with-comm
    artifact, so ``DeepSpinPTExpt::compute_inner`` drives a real
    domain-decomposed run through ``run_model_graph_with_comm``: the
    per-block ghost FEATURE refresh rides ``border_op`` while ghost SPINS
    arrive via the LAMMPS ``sp`` forward-comm.  Both the conservative force
    and the MAGNETIC force must be rank-count invariant -- force_mag is the
    output that only exists on this route, so comparing it is what proves
    the spin leaf survived the with-comm lower.

    Replaces the previous fail-fast test, which asserted the C++ throw that
    existed only while native spin was excluded from the with-comm export.
    """
    single = _run_mpi(data_file, 1, "1 1 1")
    multi = _run_mpi(data_file, 2, "2 1 1")
    assert_mpi_matches_single_rank(single, multi)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepspin_mpi_empty_rank_phantom_pads_and_matches() -> None:
    """A rank that owns ZERO local atoms (but holds ghosts) must SUCCEED
    through the DeepSpin phantom path -- the first coverage of that path
    (issue #5906 Task 12b).

    This is the deliberate divergence from the energy route: where
    ``DeepPotPTExpt`` has no owned-empty special case,
    ``DeepSpinPTExpt::compute_inner`` phantom-pads the owned-empty rank
    route-agnostically -- it prepends 2 phantom atoms with an empty nlist
    row (contributing exactly zero energy/force/virial), because the
    inductor specialization assumes ``nloc >= 2`` (PR #5485).  A crash,
    fail-fast exit, or hang here would therefore be a real regression of
    the phantom logic, not the expected behaviour.  (A GENUINELY empty
    rank -- zero owned AND zero ghost -- is a different corner: it is
    rejected by the same collective "zero owned+ghost atoms" preflight as
    the energy route, because the phantom path requires ``nall_real > 0``;
    see the fixture comment.)

    ``data_file_empty_rank`` (3-way x-split, ``processors 3 1 1``) was
    verified (see the module-level comment above the fixture) to put BOTH
    non-first ranks in the owned-empty-with-ghosts state; the 1-rank run on
    the SAME wide-box data file is the same-archive reference.  Energy,
    conservative force AND magnetic force must all be rank-count invariant
    at the file's MPI tolerances -- force_mag only exists on this route, so
    comparing it is what proves the spin leaf survives phantom padding.
    """
    single = _run_mpi(data_file_empty_rank, 1, "1 1 1")
    multi = _run_mpi(data_file_empty_rank, 3, "3 1 1")
    assert_mpi_matches_single_rank(single, multi)
