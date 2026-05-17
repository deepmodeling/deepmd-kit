# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with .pt2 (AOTInductor) DPA3 model.

Mirrors test_lammps_pt2.py (se_e2_a) but for the DPA3 descriptor.
Reference values from source/tests/infer/gen_dpa3.py / C++ test.
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

pb_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa3.pt2"
# Multi-rank-capable variant (use_loc_mapping=False; carries the
# nested forward_lower_with_comm.pt2 artifact). Produced alongside
# deeppot_dpa3.pt2 by source/tests/infer/gen_dpa3.py.
pb_file_mpi = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa3_mpi.pt2"
)
ref_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa3.expected"
)
data_file = Path(__file__).parent / "data_dpa3_pt2.lmp"
data_file_si = Path(__file__).parent / "data_dpa3_pt2.si"
data_type_map_file = Path(__file__).parent / "data_type_map_dpa3_pt2.lmp"
# Elongated-box variant for the empty-subdomain MPI test: x is
# extended to 30 Å while atoms remain in x ∈ [0.25, 12.83]. Combined
# with ``processors 2 1 1`` this leaves rank 1 (x ≥ 15) with zero
# local atoms — a corner case the comm-dispatch path must handle
# without crashing or producing wrong forces.
data_file_empty_subdomain = Path(__file__).parent / "data_dpa3_pt2_empty_subdomain.lmp"
# NULL-type variant: 6 real atoms (types 1,2) + 2 type-3 atoms straddling
# the x=6.5 rank boundary. With ``pair_coeff * * O H NULL`` LAMMPS type 3
# maps to deepmd atype=-1, so those atoms are filtered by
# ``select_real_atoms_coord`` and the comm tensors must be remapped via
# ``fwd_map`` before being handed to the with-comm artifact. Forces on
# the 6 real atoms must match the no-NULL baseline; NULL atoms get zero
# force from the deepmd model.
data_file_null_type = Path(__file__).parent / "data_dpa3_pt2_null_type.lmp"
# Isolated-NULL fixture: box=30 Å in x so rank 0 (x ∈ [0, 15]) has a
# subdomain interior that is NOT within rcut of any boundary. With
# rcut=6, boundary-adjacent regions are [0, 6] (PBC of right wall)
# and [9, 15] (left wall of rank 1) — atoms in x in (6, 9) are LOCAL
# but not in any sendlist. Place 1 NULL atom at x=7.5 (in this gap)
# so the remap branch is reached but the sendlists contain no NULL
# entries — exercises ``has_null_atoms=true`` with no-op remap.
data_file_null_isolated = Path(__file__).parent / "data_dpa3_pt2_null_isolated.lmp"
# All-NULL-rank fixture: box=30 Å in x. 6 real atoms in rank 0
# (x < 13). 2 NULL atoms in rank 1 (x ∈ {20, 25}). Under
# ``processors 2 1 1`` rank 1 owns ONLY NULL atoms, so after
# ``select_real_atoms_coord`` rank 1 has nloc_real=0 (intersection
# of empty-subdomain and NULL-type paths).
data_file_all_null_rank = Path(__file__).parent / "data_dpa3_pt2_all_null_rank.lmp"

# Reference values written by source/tests/infer/gen_dpa3.py (PBC case).
# Guarded with try/except because gen_dpa3.py only runs when PyTorch is built;
# matrices that disable PyTorch (e.g. paddle-only) skip the test in
# setup_module but still load this file at pytest collection time.
try:
    _ref = read_expected_ref(ref_file)["pbc"]
    expected_e = float(np.sum(_ref["expected_e"]))
    expected_f = _ref["expected_f"].reshape(6, 3)
    # LAMMPS uses opposite sign convention for virial vs DeepPot atom_virial.
    expected_v = -_ref["expected_v"].reshape(6, 9)
except FileNotFoundError:
    expected_e = expected_f = expected_v = None

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
type_OH = np.array([1, 2, 2, 1, 2, 2])
type_HO = np.array([2, 1, 1, 2, 1, 1])


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data(box, coord, type_OH, data_file)
    write_lmp_data(box, coord, type_HO, data_type_map_file)
    write_lmp_data(
        box * constants.dist_metal2si,
        coord * constants.dist_metal2si,
        type_OH,
        data_file_si,
    )
    # Elongated x-axis; atoms unchanged. With ``processors 2 1 1`` the
    # split is at x = 15 Å and rank 1 owns x ≥ 15, which is empty.
    box_empty_subdomain = np.array([0, 30, 0, 13, 0, 13, 0, 0, 0])
    write_lmp_data(box_empty_subdomain, coord, type_OH, data_file_empty_subdomain)
    # NULL-type fixture: original 6 real atoms (types 1,2) plus 2 LAMMPS
    # type-3 atoms placed within rcut (~6 Å) of real atoms on BOTH sides
    # of the x=6.5 rank boundary. The NULL atoms appear in real atoms'
    # neighbour lists and in the cross-rank sendlists, so the comm-tensor
    # remap (``fwd_map``-based) is genuinely exercised — not trivial.
    coord_null_type = np.concatenate(
        [
            coord,
            np.array(
                [
                    [5.5, 6.0, 6.0],  # rank 0 side, near boundary
                    [7.5, 7.0, 7.0],  # rank 1 side, near boundary
                ]
            ),
        ]
    )
    type_null = np.concatenate([type_OH, np.array([3, 3])])
    write_lmp_data(box, coord_null_type, type_null, data_file_null_type)
    # Isolated-NULL fixture: same elongated box as empty-subdomain
    # plus one NULL atom in rank 0's subdomain interior (x ∈ (6, 9)).
    coord_null_isolated = np.concatenate([coord, np.array([[7.5, 6.5, 6.5]])])
    type_null_isolated = np.concatenate([type_OH, np.array([3])])
    write_lmp_data(
        box_empty_subdomain,
        coord_null_isolated,
        type_null_isolated,
        data_file_null_isolated,
    )
    # All-NULL-rank fixture: box=30 in x. Real atoms in rank 0
    # (their original coords; all x < 13). NULL atoms placed in
    # rank 1 (x ∈ {20, 25}). Rank 1 owns ONLY NULL atoms.
    coord_all_null_rank = np.concatenate(
        [
            coord,
            np.array(
                [
                    [20.0, 6.5, 6.5],
                    [25.0, 6.5, 6.5],
                ]
            ),
        ]
    )
    type_all_null_rank = np.concatenate([type_OH, np.array([3, 3])])
    write_lmp_data(
        box_empty_subdomain,
        coord_all_null_rank,
        type_all_null_rank,
        data_file_all_null_rank,
    )


def teardown_module() -> None:
    for f in [
        data_file,
        data_type_map_file,
        data_file_si,
        data_file_empty_subdomain,
        data_file_null_type,
        data_file_null_isolated,
        data_file_all_null_rank,
    ]:
        if f.exists():
            os.remove(f)


def _lammps(data_file, units="metal") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
    lammps.atom_modify("map yes")
    if units == "metal" or units == "real":
        lammps.neighbor("2.0 bin")
    elif units == "si":
        lammps.neighbor("2.0e-10 bin")
    else:
        raise ValueError("units should be metal, real, or si")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    if units == "metal" or units == "real":
        lammps.mass("1 16")
        lammps.mass("2 2")
    elif units == "si":
        lammps.mass("1 %.10e" % (16 * constants.mass_metal2si))
        lammps.mass("2 %.10e" % (2 * constants.mass_metal2si))
    else:
        raise ValueError("units should be metal, real, or si")
    if units == "metal":
        lammps.timestep(0.0005)
    elif units == "real":
        lammps.timestep(0.5)
    elif units == "si":
        lammps.timestep(5e-16)
    else:
        raise ValueError("units should be metal, real, or si")
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_type_map():
    lmp = _lammps(data_file=data_type_map_file)
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_real():
    lmp = _lammps(data_file=data_file, units="real")
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_si():
    lmp = _lammps(data_file=data_file_si, units="si")
    yield lmp
    lmp.close()


def test_pair_deepmd(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_virial(lammps) -> None:
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


def test_pair_deepmd_type_map(lammps_type_map) -> None:
    lammps_type_map.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_type_map.pair_coeff("* * H O")
    lammps_type_map.run(0)
    assert lammps_type_map.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps_type_map.atoms[ii].force == pytest.approx(
            expected_f[lammps_type_map.atoms[ii].id - 1]
        )
    lammps_type_map.run(1)


def test_pair_deepmd_real(lammps_real) -> None:
    lammps_real.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_real.pair_coeff("* *")
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    lammps_real.run(1)


def test_pair_deepmd_virial_real(lammps_real) -> None:
    lammps_real.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_real.pair_coeff("* *")
    lammps_real.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps_real.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps_real.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    idx_map = lammps_real.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    for ii in range(9):
        assert np.array(
            lammps_real.variables[f"virial{ii}"].value
        ) / constants.nktv2p_real == pytest.approx(
            expected_v[idx_map, ii] * constants.ener_metal2real
        )


def test_pair_deepmd_si(lammps_si) -> None:
    lammps_si.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_si.pair_coeff("* *")
    lammps_si.run(0)
    assert lammps_si.eval("pe") == pytest.approx(expected_e * constants.ener_metal2si)
    for ii in range(6):
        assert lammps_si.atoms[ii].force == pytest.approx(
            expected_f[lammps_si.atoms[ii].id - 1] * constants.force_metal2si
        )
    lammps_si.run(1)


# ---------------------------------------------------------------------------
# Multi-rank test (Phase 5 of GNN MPI)
#
# Drives the .pt2 model under ``mpirun -n 2`` so the C++ ``DeepPotPTExpt``
# routes to the with-comm AOTI artifact (Phase 4) and ``border_op`` does
# real MPI ghost exchange between two ranks.  The expected energy/forces
# are the same as the single-rank reference (single-rank LAMMPS would
# need ``atom_modify map yes`` to use the regular artifact; multi-rank
# uses the with-comm artifact whose graph reproduces the gather via
# MPI exchange).
# ---------------------------------------------------------------------------


def _run_mpi_subprocess(
    extra_args: list[str] | None = None,
    nprocs: int = 2,
    data_path: Path | None = None,
    processors: str | None = None,
    runner_args: list[str] | None = None,
) -> dict:
    """Helper: invoke run_mpi_pair_deepmd_dpa3_pt2.py under
    ``mpirun -n <nprocs>`` and return
    ``{"pe": float, "forces": (n, 3) array, "virials": (n, 9) array}``.

    With ``nprocs == 1`` the runner is invoked with ``--processors 1 1 1``
    so the C++ side sees ``nswap == 0`` and routes to the regular
    (single-rank) artifact of the dual-artifact .pt2 — useful as a
    same-archive reference for multi-rank comparisons.

    ``data_path`` (default ``data_file``) selects the LAMMPS data file —
    the empty-subdomain test points at a non-default elongated-box file.

    ``processors`` overrides the runner's default decomposition string
    (``"2 1 1"``); used by the ``test_*_decomposition`` variants to
    exercise 2D / 3D processor grids (Px*Py*Pz must equal nprocs).
    """
    if data_path is None:
        data_path = data_file
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        out_path = f.name
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"),
            str(data_path.resolve()),
            str(pb_file_mpi.resolve()),
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
        # Each row is (3 force) + (9 virial); see runner script.
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
def test_pair_deepmd_mpi_dpa3() -> None:
    """Multi-rank LAMMPS run for DPA3 .pt2 must match the single-rank
    reference within numerical tolerance for energy, forces, and virial.

    Forces are the autograd output of energy through the with-comm
    graph, so they implicitly validate the backward path of
    ``deepmd_export::border_op``. Per-atom virial is gathered from
    ``compute centroid/stress/atom NULL pair`` (parallel-safe) — the
    earlier deadlock comment was specific to ``compute pressure NULL
    virial`` + ``lammps.eval(...)``, which we sidestep entirely.

    Requires the .pt2 archive to carry a with-comm artifact (Phase 3
    output for GNN models). If the archive lacks it, the C++ falls
    back to the regular artifact and produces wrong cross-rank values
    — which the assertion would catch (loud test failure, not silent).
    """
    out = _run_mpi_subprocess()
    # Energy matches single-rank reference.
    assert out["pe"] == pytest.approx(expected_e)
    # Per-atom forces match (atoms in id-sorted order from the
    # subprocess script).
    for ii in range(6):
        np.testing.assert_allclose(
            out["forces"][ii],
            expected_f[ii],
            atol=1e-8,
            rtol=0,
        )
    # Per-atom virial matches the gen_dpa3.py reference. LAMMPS
    # centroid/stress/atom returns components in [xx, yy, zz, xy, xz,
    # yz, yx, zx, zy] order; ``expected_v`` columns follow the same
    # column-major flattening as the single-rank ``test_pair_deepmd_virial``
    # (which uses idx_map [0, 4, 8, 3, 6, 7, 1, 2, 5] from c_virial[1..9]
    # to expected_v columns). The inverse permutation maps
    # ``out["virials"]`` columns back to ``expected_v`` columns.
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
def test_pair_deepmd_mpi_dpa3_nlist_rebuild() -> None:
    """Multi-rank with neighbor-list rebuilds, validated against a
    single-rank reference of the same archive and trajectory.

    Uses ``neigh_modify every 1`` so a rebuild happens on every step,
    then runs 3 steps — yields 3 rebuilds in roughly 1/8 the wall
    time of a 25-step ``every 10`` run. The same trajectory is then
    run under ``mpirun -n 1`` (regular-artifact dispatch on the same
    dual-artifact .pt2) to obtain a reference; comparing the two
    catches a wrong-but-finite force from a dispatch bug.

    NVE is deterministic up to floating-point summation order, so
    the cross-rank divergence after 3 steps is bounded by accumulated
    round-off — small for a 6-atom system but non-zero, hence the
    relaxed (but still tight) tolerances.
    """
    runner_args = ["--neigh-every", "1"]
    out_mpi = _run_mpi_subprocess(
        extra_args=["--nsteps", "3"], nprocs=2, runner_args=runner_args
    )
    out_ref = _run_mpi_subprocess(
        extra_args=["--nsteps", "3"], nprocs=1, runner_args=runner_args
    )
    np.testing.assert_allclose(
        out_mpi["forces"],
        out_ref["forces"],
        atol=1e-6,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        out_mpi["virials"],
        out_ref["virials"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa3_empty_subdomain() -> None:
    """Multi-rank DPA3 with one rank owning zero local atoms.

    Runs 5 MD steps with ``neigh_modify every 100`` so the nlist is
    rebuilt only once (at step 0, ago=0) and the next 4 force
    evaluations exercise the cached ``mapping_tensor`` /
    ``firstneigh_tensor`` path (PR 5407 caching) under empty
    subdomain. Atoms move ~0 (v=0 default) so positions only differ
    by tiny round-off, but the C++ dispatch path with cached state
    on rank 1 (which has nloc=0) must still produce correct
    cross-rank forces.

    Uses a 30 x 13 x 13 box with all six atoms clustered in x in
    [0.25, 12.83]. Under ``processors 2 1 1`` the split is at x = 15
    so rank 1 owns an empty subdomain. The comm-dispatch path must
    still produce correct forces and virial (compared against a
    same-archive single-rank reference of the same trajectory).

    This catches: zero-length send/recv lists in the comm tensors,
    division-by-zero in nlocal-dependent reshapes, silent drop of a
    rank's contribution when it has no atoms to evaluate, AND
    cache-hit (ago>0) bugs specific to the empty-subdomain rank.
    """
    runner_args = ["--neigh-every", "100"]
    out_mpi = _run_mpi_subprocess(
        nprocs=2,
        data_path=data_file_empty_subdomain,
        extra_args=["--nsteps", "5"],
        runner_args=runner_args,
    )
    out_ref = _run_mpi_subprocess(
        nprocs=1,
        data_path=data_file_empty_subdomain,
        extra_args=["--nsteps", "5"],
        runner_args=runner_args,
    )
    np.testing.assert_allclose(
        out_mpi["forces"], out_ref["forces"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-6, rtol=1e-6
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
@pytest.mark.parametrize(
    "nprocs,processors",
    [
        # 2D ``2 2 1`` is omitted: ``8 @ 2 2 2`` already exercises 2D
        # face exchange (it's a superset, in 3D), so the 2D-only case
        # is redundant. The two kept variants give complementary
        # coverage: 1D-deep sendlist chains vs 3D border exchange.
        (4, "4 1 1"),  # 1D-deep chain; sendlist depth = 3 (each pair is 1+2 swaps)
        (8, "2 2 2"),  # 3D decomposition; full xyz border exchange
    ],
)
def test_pair_deepmd_mpi_dpa3_decomposition(nprocs, processors) -> None:
    """Multi-rank DPA3 .pt2 must match the single-rank reference under
    deeper / 3D processor grids beyond the canonical 2x1x1 (N=2) layout.

    Production MD typically runs with 8/16/32+ ranks and 2D/3D
    decompositions. Bugs that don't fire at N=2 (deeper sendlist
    chains, 3D border swaps, asymmetric subdomains, multiple empty
    cells in the 2x2x2 split of a small fixture) have zero coverage
    without this test.

    The 6-atom 13x13x13 fixture is intentionally small relative to
    the rank count: in the 2x2x2 split each subdomain is
    ~6.5x6.5x6.5 A, so several subdomains are empty — exercising the
    empty-subdomain ``copy_from_nlist`` guard fix in 3D.
    """
    out_mpi = _run_mpi_subprocess(nprocs=nprocs, processors=processors)
    # Step-0 evaluation; bit-exact match expected against the
    # gen_dpa3.py-derived reference.
    assert out_mpi["pe"] == pytest.approx(expected_e, rel=0, abs=1e-8)
    for ii in range(6):
        np.testing.assert_allclose(
            out_mpi["forces"][ii], expected_f[ii], atol=1e-8, rtol=0
        )
    expected_v_to_lammps = [0, 6, 7, 3, 1, 8, 4, 5, 2]
    np.testing.assert_allclose(
        out_mpi["virials"][:, expected_v_to_lammps] / constants.nktv2p,
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
def test_pair_deepmd_mpi_dpa3_null_type() -> None:
    """Multi-rank DPA3 .pt2 with NULL-type atoms.

    Exercises ``select_real_atoms_coord`` filtering AND
    ``build_comm_tensors_positional_with_virtual_atoms`` remapping
    under multi-rank dispatch — neither path was reachable in any
    previous test fixture.

    Setup: 6 real atoms (types 1,2) at the canonical positions plus
    2 LAMMPS type-3 atoms straddling the x=6.5 rank boundary. With
    ``pair_coeff * * O H NULL`` the type-3 atoms map to deepmd
    atype=-1 and are filtered before model evaluation. Because the
    NULL atoms sit within rcut of real atoms on BOTH sides of the
    boundary, they appear in cross-rank sendlists — forcing the
    ``fwd_map``-based remap (which translates unfiltered LAMMPS
    indices into filtered real-atom indices, dropping ``-1`` slots).

    Assertions:
    - Forces on the 6 real atoms (ids 1..6, id-sorted output) match
      the no-NULL baseline ``expected_f`` exactly. NULL atoms don't
      contribute to the deepmd model so real-atom forces are
      identical to the 6-atom baseline.
    - NULL-atom forces (ids 7,8) are zero — the deepmd model is the
      only pair_style and skips them entirely.
    - Total energy matches ``expected_e``.
    - Per-atom virial on real atoms matches ``expected_v``.
    """
    out_mpi = _run_mpi_subprocess(
        nprocs=2,
        data_path=data_file_null_type,
        runner_args=["--pair-coeff", "* * O H NULL", "--mass3", "5.0"],
    )
    # Forces on real atoms (ids 1..6) match the no-NULL baseline.
    real_forces = out_mpi["forces"][:6]
    for ii in range(6):
        np.testing.assert_allclose(real_forces[ii], expected_f[ii], atol=1e-8, rtol=0)
    # NULL atoms (ids 7,8) get zero force from the deepmd model.
    null_forces = out_mpi["forces"][6:]
    np.testing.assert_allclose(null_forces, 0.0, atol=1e-12, rtol=0)
    # Total potential energy unchanged (NULL atoms contribute 0).
    assert out_mpi["pe"] == pytest.approx(expected_e, rel=0, abs=1e-8)
    # Per-atom virial on real atoms matches expected_v with the same
    # column permutation as test_pair_deepmd_mpi_dpa3.
    expected_v_to_lammps = [0, 6, 7, 3, 1, 8, 4, 5, 2]
    real_virials = out_mpi["virials"][:6]
    np.testing.assert_allclose(
        real_virials[:, expected_v_to_lammps] / constants.nktv2p,
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
def test_pair_deepmd_mpi_dpa3_null_isolated() -> None:
    """NULL atom local on a rank but absent from every sendlist.

    Box is 30x13x13 with split at x=15. With rcut=6 the boundary
    rcut-windows on rank 0 are x ∈ [0, 6] (PBC of right wall via
    x=30) and x ∈ [9, 15] (left wall of rank 1). Atoms in
    x ∈ (6, 9) are LOCAL on rank 0 but never appear in any
    cross-rank sendlist. Placing a NULL atom at x=7.5 puts it in
    that gap.

    Coverage: ``has_null_atoms == True`` triggers the
    ``_with_virtual_atoms`` branch, but the remap encounters NO
    NULL entries in any sendlist (no-op remap). The
    ``test_pair_deepmd_mpi_dpa3_null_type`` test exercises the
    remap-with-NULLs case; this one pins the
    remap-with-no-NULLs-in-sendlist case.

    Comparison is mpi-vs-single-rank on the same fixture (no hardcoded
    reference because the box differs from the canonical 13x13x13).
    """
    out_mpi = _run_mpi_subprocess(
        nprocs=2,
        data_path=data_file_null_isolated,
        runner_args=["--pair-coeff", "* * O H NULL", "--mass3", "5.0"],
    )
    out_ref = _run_mpi_subprocess(
        nprocs=1,
        data_path=data_file_null_isolated,
        runner_args=["--pair-coeff", "* * O H NULL", "--mass3", "5.0"],
    )
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=0
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=0, abs=1e-8)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa3_all_null_rank() -> None:
    """Rank that owns ONLY NULL atoms (intersection of empty-subdomain
    and NULL-type paths).

    Box=30x13x13, split at x=15. Real atoms (types 1,2) are all in
    rank 0 (x < 13). NULL atoms (type 3) are at x ∈ {20, 25},
    both in rank 1. After ``select_real_atoms_coord``:

    - Rank 0: nloc_real=6 (all real local), receives NULL atoms as
      ghosts via PBC -> filtered -> nall_real ≤ nall.
    - Rank 1: nloc_real=0 (all local atoms filtered out — empty
      subdomain after filter), receives real atoms as ghosts.

    Tests that the comm-dispatch path handles a rank with zero real
    locals correctly. The empty-subdomain ``copy_from_nlist`` guard
    must fire on rank 1, AND the ``_with_virtual_atoms`` remap must
    handle the case where the local section of the sendlist is
    entirely NULL.
    """
    out_mpi = _run_mpi_subprocess(
        nprocs=2,
        data_path=data_file_all_null_rank,
        runner_args=["--pair-coeff", "* * O H NULL", "--mass3", "5.0"],
    )
    out_ref = _run_mpi_subprocess(
        nprocs=1,
        data_path=data_file_all_null_rank,
        runner_args=["--pair-coeff", "* * O H NULL", "--mass3", "5.0"],
    )
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=0
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=0, abs=1e-8)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa3_null_type_nlist_rebuild() -> None:
    """NULL atoms cross the boundary in OPPOSITE directions while
    real atoms move randomly via thermal motion — sendlist
    composition changes both ways per rebuild.

    Initial conditions:
    - Real atoms (types 1, 2): thermal velocities at T=10000 K
      (``--real-temp 10000``). Each real atom gets a different
      random direction; mass-weighted RMS speed is roughly
      3 - 9 A/ps so motion in 3 steps is ~0.005 - 0.015 A. Tiny
      but enough to perturb sendlist composition under
      ``every 1`` rebuilds.
    - NULL atom 7 (id=7) at x=5.5: gets ``v_x = -2000 A/ps`` via
      ``--null-vx 2000 --null-vx-split`` (odd id -> negative).
      Wraps via PBC: x = 5.5 -> 4.5 -> 3.5 -> 2.5 (stays in rank 0
      but drifts deeper into the PBC ghost region of rank 1).
    - NULL atom 8 (id=8) at x=7.5: gets ``v_x = +2000 A/ps``
      (even id -> positive). x = 7.5 -> 8.5 -> 9.5 -> 10.5 (stays
      in rank 1 but drifts deeper).

    The +x/-x split means each rebuild sees NULL atoms entering
    different sendlists (rank 0's right-edge sendlist gains NULL 7
    even as it loses NULL 8 deeper into rank 1's domain, and vice
    versa). Real-atom thermal motion provides additional sendlist
    perturbation per atom.

    Coverage: ``has_null_atoms`` must remain True; the
    ``_with_virtual_atoms`` remap must produce correct outputs as
    NULL atoms migrate in mixed directions and real-atom positions
    shift. Compares mpi-2-rank vs mpi-1-rank trajectories
    deterministically (both use the same velocity seed 12345).
    """
    runner_args = [
        "--pair-coeff",
        "* * O H NULL",
        "--mass3",
        "5.0",
        "--neigh-every",
        "1",
        "--null-vx",
        "2000.0",
        "--null-vx-split",
        "--real-temp",
        "10000.0",
    ]
    out_mpi = _run_mpi_subprocess(
        nprocs=2,
        data_path=data_file_null_type,
        extra_args=["--nsteps", "3"],
        runner_args=runner_args,
    )
    out_ref = _run_mpi_subprocess(
        nprocs=1,
        data_path=data_file_null_type,
        extra_args=["--nsteps", "3"],
        runner_args=runner_args,
    )
    np.testing.assert_allclose(
        out_mpi["forces"], out_ref["forces"], atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-6, rtol=1e-6
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)
