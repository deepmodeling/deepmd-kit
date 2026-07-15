# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared infrastructure for the LAMMPS Python tests.

Model paths, expected values, and scenario-specific assertions stay in their
test modules. This module owns setup that must remain identical across model
formats and backends, so changes to the LAMMPS test system have one source of
truth.
"""

from __future__ import (
    annotations,
)

import os
import subprocess as sp
import sys
import tempfile
from pathlib import (
    Path,
)
from typing import (
    Any,
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


def require_backend(environment_variable: str, backend_name: str) -> None:
    """Skip the current module when its compiled backend is unavailable."""
    if os.environ.get(environment_variable, "1") != "1":
        pytest.skip(f"Skip test because {backend_name} support is not enabled.")


def remove_test_files(*paths: Path) -> None:
    """Remove generated test files, tolerating partial setup and prior cleanup."""
    for path in paths:
        path.unlink(missing_ok=True)


def write_water_data_variants(
    box: np.ndarray,
    coord: np.ndarray,
    type_oh: np.ndarray,
    type_ho: np.ndarray,
    data_file: Path,
    type_map_file: Path,
    si_file: Path,
) -> None:
    """Write the standard metal, type-map, and SI water test fixtures."""
    write_lmp_data(box, coord, type_oh, data_file)
    write_lmp_data(box, coord, type_ho, type_map_file)
    write_lmp_data(
        box * constants.dist_metal2si,
        coord * constants.dist_metal2si,
        type_oh,
        si_file,
    )


def make_atomic_lammps(
    data_file: Path,
    units: str = "metal",
    *,
    boundary: str = "p p p",
    atom_map: str | None = None,
    masses: tuple[float, ...] = (16, 2),
) -> PyLammps:
    """Create the standard two-type atomic LAMMPS test system.

    ``atom_map="no"`` deliberately omits ``atom_modify`` because LAMMPS
    rejects ``atom_modify map no``; this preserves the no-map failure-path
    tests used by the graph-model fixtures.
    """
    if units not in {"metal", "real", "si"}:
        raise ValueError("units should be metal, real, or si")

    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary(boundary)
    lammps.atom_style("atomic")
    if atom_map is not None and atom_map != "no":
        lammps.atom_modify(f"map {atom_map}")
    lammps.neighbor("2.0e-10 bin" if units == "si" else "2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    for atom_type, mass in enumerate(masses, start=1):
        if units == "si":
            lammps.mass(f"{atom_type} {mass * constants.mass_metal2si:.10e}")
        else:
            lammps.mass(f"{atom_type} {mass:g}")
    lammps.timestep({"metal": 0.0005, "real": 0.5, "si": 5e-16}[units])
    lammps.fix("1 all nve")
    return lammps


def make_spin_lammps(
    data_file: Path,
    units: str = "metal",
    *,
    boundary: str = "p p p",
) -> PyLammps:
    """Create the standard two-type DeepSpin LAMMPS test system."""
    if units != "metal":
        raise ValueError("units for spin should be metal")

    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary(boundary)
    lammps.atom_style("spin")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 58")
    lammps.mass("2 16")
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


def run_mpi_pair_runner(
    runner: Path,
    data_file: Path,
    model_file: Path,
    *,
    nprocs: int = 2,
    processors: str | None = None,
    extra_args: list[str] | None = None,
    runner_args: list[str] | None = None,
    output_columns: tuple[tuple[str, int], ...] = (("forces", 3), ("virials", 9)),
    capture: bool = False,
) -> dict[str, Any]:
    """Invoke a DPA MPI runner and parse its energy/per-atom output.

    The runner output contract is one energy line followed by a rectangular
    per-atom table. ``output_columns`` names and slices that table while each
    model-specific wrapper retains its own defaults and explanatory docstring.

    If ``capture`` is true, skip parsing and return the subprocess result as
    ``{"returncode": int, "stdout": str, "stderr": str}``.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        output_path = Path(f.name)
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(runner),
            str(data_file.resolve()),
            str(model_file.resolve()),
            str(output_path),
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
            proc = sp.run(argv, capture_output=True, text=True)
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

        sp.check_call(argv)
        lines = output_path.read_text().strip().splitlines()
        rows = np.array(
            [list(map(float, line.split())) for line in lines[1:]],
            dtype=np.float64,
        )
        result: dict[str, Any] = {"pe": float(lines[0])}
        start = 0
        for name, width in output_columns:
            result[name] = rows[:, start : start + width]
            start += width
        if rows.shape[1] != start:
            raise ValueError(
                f"MPI runner produced {rows.shape[1]} columns; expected {start}"
            )
        return result
    finally:
        output_path.unlink(missing_ok=True)


def run_mpi_model_deviation(
    runner: Path,
    data_file: Path,
    model_file: Path,
    second_model_file: Path,
    deviation_file: Path,
    *,
    extra_args: list[str] | None = None,
) -> float:
    """Run the two-rank model-deviation driver and return rank-zero energy."""
    with tempfile.NamedTemporaryFile() as output:
        argv = [
            "mpirun",
            "-n",
            "2",
            sys.executable,
            str(runner),
            str(data_file),
            str(model_file),
            str(second_model_file),
            str(deviation_file),
            output.name,
        ]
        if extra_args:
            argv.extend(extra_args)
        sp.check_call(argv)
        return float(np.loadtxt(output.name, ndmin=1)[0])
