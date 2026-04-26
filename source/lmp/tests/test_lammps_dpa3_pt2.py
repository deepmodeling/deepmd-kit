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
data_file = Path(__file__).parent / "data_dpa3_pt2.lmp"
data_file_si = Path(__file__).parent / "data_dpa3_pt2.si"
data_type_map_file = Path(__file__).parent / "data_type_map_dpa3_pt2.lmp"

# Reference values from gen_dpa3.py / test_deeppot_dpa3_ptexpt.cc (PBC)
expected_ae = np.array(
    [
        2.733142942358297023e-01,
        2.768815473296480922e-01,
        2.781664369968356865e-01,
        2.697839344989072519e-01,
        2.741210600049306945e-01,
        2.752870928812235496e-01,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        -1.962618723134541832e-02,
        4.287158582278347702e-02,
        7.640666386947853050e-03,
        5.554130248696588501e-02,
        -6.501206231527984977e-03,
        -4.524468847893595158e-02,
        -3.851051736663693714e-02,
        -3.620789238677154381e-02,
        3.756162244251591564e-02,
        6.729090678104879264e-02,
        -2.430710555108604037e-02,
        4.496058666120762021e-02,
        9.285825331084011924e-03,
        5.623126339971108029e-02,
        -8.776072674283137698e-02,
        -7.398133000111631330e-02,
        -3.208664505310900028e-02,
        4.284253973109593966e-02,
    ]
).reshape(6, 3)

expected_v = -np.array(
    [
        -2.519191242984861884e-02,
        -7.976296517418629550e-04,
        2.293255716383547221e-02,
        -1.129879902880513709e-04,
        -2.480533869648754441e-02,
        5.147545203263749480e-03,
        2.250634701911344987e-02,
        5.288887046140826331e-03,
        -2.010244267109611085e-02,
        -1.779331319768159489e-02,
        3.093850189397499839e-03,
        1.469388965841003300e-02,
        -3.857294749719837688e-03,
        1.122172669801067097e-03,
        3.015485878866499582e-03,
        1.588838841470147090e-02,
        -2.814760933954751562e-03,
        -1.277216714527013713e-02,
        -8.763367643346370306e-03,
        -1.305889135368112908e-02,
        1.181350951828694096e-02,
        -6.506014073233991855e-03,
        -6.021216432246893902e-03,
        6.406967309407277100e-03,
        1.054423249710041179e-02,
        1.210616766999832172e-02,
        -1.127472660426425549e-02,
        -3.873334330831591787e-02,
        -3.620067664760272686e-03,
        1.173198873109224322e-03,
        -3.979800321914496279e-03,
        -1.483777776121806245e-02,
        2.311848485249741111e-02,
        1.659292900032220339e-03,
        2.315104663227764842e-02,
        -3.645194750481960122e-02,
        -1.668107738824501848e-04,
        -7.331929353596922626e-03,
        1.141573012886789966e-02,
        -1.498650485705460686e-03,
        -1.339178008942835431e-02,
        2.104129816063767672e-02,
        2.247013447171188061e-03,
        2.035538814221872148e-02,
        -3.195007182084359104e-02,
        -2.339460083073257798e-02,
        -1.001949167693141039e-02,
        1.320033846426920537e-02,
        -1.577941189045228843e-02,
        -6.283307183655661120e-03,
        8.237968913765561507e-03,
        2.238394952866012630e-02,
        8.881021761757389166e-03,
        -1.162377795308391741e-02,
    ]
).reshape(6, 9)

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


def teardown_module() -> None:
    for f in [data_file, data_type_map_file, data_file_si]:
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
) -> dict:
    """Helper: invoke run_mpi_pair_deepmd_dpa3_pt2.py under
    ``mpirun -n <nprocs>`` and return ``{"pe": float, "forces": (n, 3) array}``.

    With ``nprocs == 1`` the runner is invoked with ``--processors 1 1 1``
    so the C++ side sees ``nswap == 0`` and routes to the regular
    (single-rank) artifact of the dual-artifact .pt2 — useful as a
    same-archive reference for multi-rank comparisons.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        out_path = f.name
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"),
            str(data_file.resolve()),
            str(pb_file_mpi.resolve()),
            out_path,
        ]
        if nprocs == 1:
            argv.extend(["--processors", "1 1 1"])
        if extra_args:
            argv.extend(extra_args)
        sp.check_call(argv)
        with open(out_path) as fh:
            lines = fh.read().strip().splitlines()
        pe = float(lines[0])
        forces = np.array(
            [list(map(float, line.split())) for line in lines[1:]],
            dtype=np.float64,
        )
        return {"pe": pe, "forces": forces}
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
    reference within numerical tolerance for energy and forces.

    Forces are the autograd output of energy through the with-comm
    graph, so they implicitly validate the backward path of
    ``deepmd_export::border_op``.  Virial requires a separate
    ``compute pressure NULL virial`` which interacts poorly with
    PyLammps multi-rank (hangs); deferred to a follow-up.

    Requires the .pt2 archive to carry a with-comm artifact (Phase 3
    output for GNN models).  If the archive lacks it, the C++ falls
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


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa3_nlist_rebuild() -> None:
    """Multi-rank with neighbor-list rebuilds, validated against a
    single-rank reference of the same archive and trajectory.

    Runs 25 MD steps with ``neigh_modify every 10 delay 0 check no``,
    so the multi-rank trajectory crosses two nlist rebuilds (at steps
    10 and 20) before the final force evaluation. The same trajectory
    is then run under ``mpirun -n 1`` (regular-artifact dispatch on
    the same dual-artifact .pt2) to obtain a reference; comparing the
    two catches a wrong-but-finite force from a dispatch bug that the
    previous finite/bounded check would miss.

    NVE is deterministic up to floating-point summation order, so the
    cross-rank divergence after 25 steps is bounded by accumulated
    round-off — small for a 6-atom system but non-zero, hence the
    relaxed (but still tight) tolerances.
    """
    out_mpi = _run_mpi_subprocess(extra_args=["--nsteps", "25"], nprocs=2)
    out_ref = _run_mpi_subprocess(extra_args=["--nsteps", "25"], nprocs=1)
    np.testing.assert_allclose(
        out_mpi["forces"],
        out_ref["forces"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-8, abs=1e-10)
