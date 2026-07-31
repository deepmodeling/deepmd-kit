# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for cutoff-based compact ``pair_style deepmd`` evaluation."""

import os
import shutil
import subprocess as sp
from pathlib import (
    Path,
)

import numpy as np
import pytest
from lammps import (
    PyLammps,
)
from lammps_test_utils import (
    require_backend,
)
from model_convert import (
    ensure_converted_pb,
)

pbtxt_file = Path(__file__).parents[2] / "tests" / "infer" / "deeppot.pbtxt"
pbtxt_file2 = Path(__file__).parents[2] / "tests" / "infer" / "deeppot-1.pbtxt"


def setup_module() -> None:
    require_backend("ENABLE_TENSORFLOW", "TensorFlow")


@pytest.fixture(scope="module")
def compact_models(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Convert the two standard energy models used by the LAMMPS tests."""
    model_dir = tmp_path_factory.mktemp("lammps_compact")
    model = model_dir / "graph.pb"
    model2 = model_dir / "graph2.pb"
    ensure_converted_pb(pbtxt_file, model)
    ensure_converted_pb(pbtxt_file2, model2)
    return model, model2


def _make_system(
    models: tuple[Path, ...],
    *,
    with_inactive_molecule: bool,
    compact: bool,
    triclinic: bool,
    deviation_file: Path | None = None,
) -> PyLammps:
    """Build a core, one selected molecule, and an optional distant molecule.

    Atom 2 is across the periodic x boundary from the core. Atom 3 belongs to
    the same molecule but lies outside both the environment and model cutoffs;
    retaining its nonzero atomic bias verifies whole-molecule promotion.
    """
    # At y=z=5, the triclinic tilt shifts the x origin by 19/30.  Apply that
    # shift to the boundary-crossing pair so both points remain inside the
    # tilted prism while preserving their one-angstrom minimum-image distance.
    boundary_shift = 19.0 / 30.0 if triclinic else 0.0
    atom_rows = [
        (1, 0.5 + boundary_shift, 5.0, 5.0, 1),
        (2, 29.5 + boundary_shift, 5.0, 5.0, 10),
        (1, 20.0, 10.0, 5.0, 10),
    ]
    if with_inactive_molecule:
        atom_rows.extend(
            [
                (2, 12.0, 20.0, 5.0, 20),
                (1, 13.0, 20.0, 5.0, 20),
            ]
        )

    lammps = PyLammps()
    if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
        lammps.lmp.command(f"plugin load {plugin}")
    lammps.units("metal")
    lammps.boundary("p p p")
    lammps.atom_style("molecular")
    lammps.atom_modify("map array")
    if triclinic:
        lammps.region("box prism 0 30 0 30 0 30 3 1 2 units box")
    else:
        lammps.region("box block 0 30 0 30 0 30 units box")
    lammps.create_box("2 box")
    for atom_type, x, y, z, _ in atom_rows:
        lammps.create_atoms(f"{atom_type} single {x} {y} {z} units box")
    for atom_id, (_, _, _, _, molecule_id) in enumerate(atom_rows, start=1):
        lammps.set(f"atom {atom_id} mol {molecule_id}")
    assert lammps.lmp.get_natoms() == len(atom_rows)
    lammps.group("qm id 1")
    lammps.mass("1 16")
    lammps.mass("2 2")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")

    style = "deepmd " + " ".join(str(model.resolve()) for model in models)
    if deviation_file is not None:
        style += f" out_file {deviation_file.resolve()} out_freq 1 atomic"
    if compact:
        style += " center_group qm environment_cutoff 1.5 include_molecule yes"
    lammps.pair_style(style)
    lammps.pair_coeff("* *")
    lammps.compute("peatom all pe/atom pair")
    lammps.variable("peatom atom c_peatom")
    lammps.run(0)
    return lammps


def _snapshot(lammps: PyLammps, natoms: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Return energy, force, and atomic energy ordered by global atom ID."""
    atom_ids = np.array(
        lammps.lmp.numpy.extract_atom("id")[:natoms], dtype=np.int64, copy=True
    )
    order = np.argsort(atom_ids)
    force = np.array(
        lammps.lmp.numpy.extract_atom("f")[:natoms], dtype=np.float64, copy=True
    )[order]
    atom_energy = np.asarray(lammps.variables["peatom"].value, dtype=np.float64)[order]
    return float(lammps.eval("pe")), force, atom_energy


def _run_compact_mpi_scenario(
    model: Path, tmp_path: Path, scenario: str, nprocs: int
) -> tuple[float, np.ndarray]:
    """Run a compact selection scenario through the MPI LAMMPS executable."""
    mpirun = shutil.which("mpirun")
    lmp = shutil.which("lmp")
    if mpirun is None or lmp is None:
        pytest.skip("MPI compact tests require mpirun and the lmp executable")

    if scenario == "cross_domain_molecule":
        # The cutoff hit crosses the x-domain boundary, while atom 3 verifies
        # that the complete molecule is selected on the remote rank.
        atom_rows = [
            (1, 14.5, 5.0, 5.0, 1),
            (2, 15.5, 5.0, 5.0, 10),
            (1, 25.0, 5.0, 5.0, 10),
            (2, 2.0, 20.0, 5.0, 20),
            (1, 3.0, 20.0, 5.0, 20),
        ]
    elif scenario == "empty_selected_rank":
        # With a 2x1x1 processor grid, all selected atoms are owned by rank 0;
        # rank 1 must still participate in the backend call and reductions.
        atom_rows = [
            (1, 2.0, 5.0, 5.0, 1),
            (2, 3.0, 5.0, 5.0, 10),
            (1, 4.0, 5.0, 5.0, 10),
            (2, 22.0, 20.0, 5.0, 20),
            (1, 23.0, 20.0, 5.0, 20),
        ]
    else:
        raise ValueError(f"unknown compact MPI scenario: {scenario}")

    run_dir = tmp_path / f"{scenario}_{nprocs}"
    run_dir.mkdir()
    input_file = run_dir / "in.compact"
    energy_file = run_dir / "energy.out"
    force_file = run_dir / "forces.dump"
    commands = []
    if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
        commands.append(f"plugin load {Path(plugin).resolve()}")
    commands.extend(
        [
            "units metal",
            f"processors {nprocs} 1 1",
            "boundary p p p",
            "atom_style molecular",
            "atom_modify map array",
            "region box block 0 30 0 30 0 30 units box",
            "create_box 2 box",
        ]
    )
    for atom_type, x, y, z, _ in atom_rows:
        commands.append(f"create_atoms {atom_type} single {x} {y} {z} units box")
    for atom_id, (_, _, _, _, molecule_id) in enumerate(atom_rows, start=1):
        commands.append(f"set atom {atom_id} mol {molecule_id}")
    commands.extend(
        [
            "group qm id 1",
            "mass 1 16",
            "mass 2 2",
            "neighbor 2.0 bin",
            "neigh_modify every 10 delay 0 check no",
            f"pair_style deepmd {model.resolve()} center_group qm "
            "environment_cutoff 1.5 include_molecule yes",
            "pair_coeff * *",
            "run 0",
            "variable compact_energy equal pe",
            f'print "${{compact_energy}}" file {energy_file} screen no',
            f"write_dump all custom {force_file} id fx fy fz modify sort id",
        ]
    )
    input_file.write_text("\n".join(commands) + "\n")

    result = sp.run(
        [mpirun, "-n", str(nprocs), lmp, "-in", str(input_file)],
        cwd=run_dir,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    energy = float(energy_file.read_text().strip())
    force_rows = np.loadtxt(force_file, skiprows=9, ndmin=2)
    assert force_rows[:, 0].astype(np.int64).tolist() == list(
        range(1, len(atom_rows) + 1)
    )
    return energy, force_rows[:, 1:]


def _make_selection_transition_system(
    model: Path, environment_x: float, *, move_environment: bool
) -> PyLammps:
    """Create a two-atom system for selection cache-invalidation coverage."""
    lammps = PyLammps()
    if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
        lammps.lmp.command(f"plugin load {plugin}")
    lammps.lmp.commands_list(
        [
            "units metal",
            "boundary p p p",
            "atom_style molecular",
            "atom_modify map array",
            "region box block 0 20 0 20 0 20 units box",
            "create_box 2 box",
            "create_atoms 1 single 5 5 5 units box",
            f"create_atoms 2 single {environment_x} 5 5 units box",
            "set atom 1 mol 1",
            "set atom 2 mol 10",
            "group qm id 1",
            "group environment id 2",
            "mass 1 16",
            "mass 2 2",
            "neighbor 2.0 bin",
            "neigh_modify every 10 delay 0 check no",
            f"pair_style deepmd {model.resolve()} center_group qm "
            "environment_cutoff 1.5 include_molecule yes",
            "pair_coeff * *",
            "compute peatom all pe/atom pair",
            "variable peatom atom c_peatom",
        ]
    )
    if move_environment:
        lammps.lmp.commands_list(
            [
                "timestep 1.0",
                "fix mover environment move linear -1.0 0.0 0.0 units box",
                "run 1",
            ]
        )
    else:
        lammps.run(0)
    return lammps


@pytest.mark.parametrize("triclinic", [False, True])
def test_compact_matches_explicit_selected_subsystem(
    compact_models: tuple[Path, Path], triclinic: bool
) -> None:
    """Compact evaluation preserves whole molecules and scatters zero outputs."""
    model, _ = compact_models
    compact_lmp = _make_system(
        (model,),
        with_inactive_molecule=True,
        compact=True,
        triclinic=triclinic,
    )
    reference_lmp = _make_system(
        (model,),
        with_inactive_molecule=False,
        compact=False,
        triclinic=triclinic,
    )
    try:
        compact_energy, compact_force, compact_atom_energy = _snapshot(compact_lmp, 5)
        reference_energy, reference_force, reference_atom_energy = _snapshot(
            reference_lmp, 3
        )
        assert compact_energy == pytest.approx(reference_energy)
        assert compact_force[:3] == pytest.approx(reference_force)
        assert compact_atom_energy[:3] == pytest.approx(reference_atom_energy)
        assert compact_force[3:] == pytest.approx(0.0)
        assert compact_atom_energy[3:] == pytest.approx(0.0)
        # Atom 3 is outside the cutoff, so its nonzero atomic bias can enter
        # only through whole-molecule promotion from atom 2.
        assert abs(compact_atom_energy[2]) > 1.0
    finally:
        compact_lmp.close()
        reference_lmp.close()


def test_compact_model_deviation_uses_selected_atoms_only(
    compact_models: tuple[Path, Path], tmp_path: Path
) -> None:
    """Deviation summaries use the compact count while excluded atoms stay zero."""
    compact_output = tmp_path / "compact_devi.out"
    reference_output = tmp_path / "reference_devi.out"
    compact_lmp = _make_system(
        compact_models,
        with_inactive_molecule=True,
        compact=True,
        triclinic=False,
        deviation_file=compact_output,
    )
    reference_lmp = _make_system(
        compact_models,
        with_inactive_molecule=False,
        compact=False,
        triclinic=False,
        deviation_file=reference_output,
    )
    try:
        compact_deviation = np.loadtxt(compact_output, ndmin=1)
        reference_deviation = np.loadtxt(reference_output, ndmin=1)
        assert compact_deviation[:7] == pytest.approx(reference_deviation[:7])
        assert compact_deviation[7:10] == pytest.approx(reference_deviation[7:])
        assert compact_deviation[10:] == pytest.approx(0.0)
    finally:
        compact_lmp.close()
        reference_lmp.close()


def test_compact_selection_change_rebuilds_backend_cache(
    compact_models: tuple[Path, Path],
) -> None:
    """An atom may enter the compact set between LAMMPS neighbor rebuilds."""
    model, _ = compact_models
    moving_lmp = _make_selection_transition_system(model, 7.2, move_environment=True)
    reference_lmp = _make_selection_transition_system(
        model, 6.2, move_environment=False
    )
    try:
        moving = _snapshot(moving_lmp, 2)
        reference = _snapshot(reference_lmp, 2)
        assert moving[0] == pytest.approx(reference[0])
        assert moving[1] == pytest.approx(reference[1])
        assert moving[2] == pytest.approx(reference[2])
    finally:
        moving_lmp.close()
        reference_lmp.close()


def test_compact_rejects_center_type_not_represented_by_model(
    compact_models: tuple[Path, Path],
) -> None:
    """A nonnegative pair mapping is insufficient if the model lacks the type."""
    model, _ = compact_models
    lammps = PyLammps()
    if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
        lammps.lmp.command(f"plugin load {plugin}")
    lammps.lmp.commands_list(
        [
            "units metal",
            "boundary p p p",
            "atom_style atomic",
            "region box block 0 20 0 20 0 20 units box",
            "create_box 3 box",
            "create_atoms 3 single 5 5 5 units box",
            "group qm id 1",
            "mass 1 16",
            "mass 2 2",
            "mass 3 1",
            f"pair_style deepmd {model.resolve()} center_group qm "
            "environment_cutoff 1.5 include_molecule no",
            "pair_coeff * *",
        ]
    )
    try:
        with pytest.raises(
            Exception,
            match=r"center_group.*type is not represented by the DeepMD model",
        ):
            lammps.run(0)
    finally:
        lammps.close()


@pytest.mark.parametrize("scenario", ["cross_domain_molecule", "empty_selected_rank"])
def test_compact_mpi_matches_serial(
    compact_models: tuple[Path, Path], tmp_path: Path, scenario: str
) -> None:
    """Two-rank selection matches serial, including remote and empty ranks."""
    model, _ = compact_models
    serial_energy, serial_force = _run_compact_mpi_scenario(
        model, tmp_path, scenario, 1
    )
    mpi_energy, mpi_force = _run_compact_mpi_scenario(model, tmp_path, scenario, 2)
    assert mpi_energy == pytest.approx(serial_energy)
    assert mpi_force == pytest.approx(serial_force)
