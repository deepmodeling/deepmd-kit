# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test DeepMD fparam from fix and numeric dE/dfparam from the current state."""

import os
from pathlib import (
    Path,
)

import numpy as np
import pytest
from lammps import (
    PyLammps,
)
from model_convert import (
    ensure_converted_pb,
)
from write_lmp_data import (
    write_lmp_data,
)

pbtxt_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "fparam_aparam.pbtxt"
)
pb_file = Path(__file__).parent / "fparam_aparam.pb"
data_file = Path(__file__).parent / "data.lmp"

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
type_OH = np.array([1, 1, 1, 1, 1, 1])


def setup_module() -> None:
    """Create the converted model and data file needed by this test module."""
    if os.environ.get("ENABLE_TENSORFLOW", "1") != "1":
        pytest.skip("Skip test because TensorFlow support is not enabled.")
    ensure_converted_pb(pbtxt_file, pb_file)
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    """Remove the temporary LAMMPS data file after the tests finish."""
    if data_file.exists():
        os.remove(data_file)


def _lammps(fp_value, units="metal") -> PyLammps:
    """Build a LAMMPS instance configured for frame-parameter derivative tests."""
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
    lammps.neighbor("2.0 bin")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    lammps.mass("1 16")
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    lammps.variable("fp equal " + str(fp_value))
    lammps.variable("dummy equal 0.0")
    lammps.fix("fpfix all ave/time 1 1 1 v_dummy v_fp")
    lammps.pair_style(
        f"deepmd {pb_file.resolve()} fparam_from_fix fpfix 2 aparam 0.25852028"
    )
    lammps.pair_coeff("* *")
    lammps.compute("dedn all deepmd/fparam/dedn f_fpfix[2]")
    return lammps


@pytest.fixture
def lammps():
    """Provide a ready-to-run LAMMPS instance for the default frame parameter."""
    lmp = _lammps(fp_value=0.25852028)
    yield lmp
    lmp.close()


def _energy_at_fp(fp_value):
    """Evaluate the potential energy at a chosen frame-parameter value."""
    lmp = _lammps(fp_value=fp_value)
    try:
        lmp.run(0)
        return lmp.eval("pe")
    finally:
        lmp.close()


def _energy_with_direct_fparam(fp_value):
    """Evaluate the potential energy using a direct fparam setting."""
    lmp = PyLammps()
    try:
        lmp.units("metal")
        lmp.boundary("p p p")
        lmp.atom_style("atomic")
        lmp.neighbor("2.0 bin")
        lmp.neigh_modify("every 10 delay 0 check no")
        lmp.read_data(data_file.resolve())
        lmp.mass("1 16")
        lmp.timestep(0.0005)
        lmp.fix("1 all nve")
        lmp.pair_style(
            f"deepmd {pb_file.resolve()} fparam {fp_value} aparam 0.25852028"
        )
        lmp.pair_coeff("* *")
        lmp.run(0)
        return lmp.eval("pe")
    finally:
        lmp.close()


def test_pair_fparam_from_fix(lammps) -> None:
    """Check that fparam_from_fix matches the direct fparam path."""
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(_energy_with_direct_fparam(0.25852028))


def test_compute_deepmd_fparam_dedn(lammps) -> None:
    """Compare the reported derivative against a central finite difference."""
    eps = 1.0e-6
    lammps.run(0)
    dedn = lammps.eval("c_dedn")
    ref = (_energy_at_fp(0.25852028 + eps) - _energy_at_fp(0.25852028 - eps)) / (
        2.0 * eps
    )
    assert dedn == pytest.approx(ref, rel=1.0e-4, abs=1.0e-4)
