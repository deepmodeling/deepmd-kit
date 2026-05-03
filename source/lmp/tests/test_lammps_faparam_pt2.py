# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with default_fparam (.pt2 AOTInductor backend)."""

import os
from pathlib import (
    Path,
)

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

pt2_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "fparam_aparam_default.pt2"
)
pt2_file_no_default = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "fparam_aparam.pt2"
)
ref_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "fparam_aparam_default.expected"
)
data_file = Path(__file__).parent / "data.lmp"

# Reference values written by source/tests/infer/gen_fparam_aparam.py for the
# default-fparam case. The .pth and .pt2 produce identical values (verified by
# the gen script's parity check).
# Guarded with try/except because gen_fparam_aparam.py only runs when PyTorch
# is built; matrices that disable PyTorch (e.g. paddle-only) skip the test in
# setup_module but still load this file at pytest collection time.
try:
    _ref = read_expected_ref(ref_file)["default"]
    expected_e = float(np.sum(_ref["expected_e"]))
    expected_f = _ref["expected_f"].reshape(6, 3)
except FileNotFoundError:
    expected_e = expected_f = None

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
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _lammps(data_file, units="metal") -> PyLammps:
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
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


def test_pair_deepmd_default_fparam(lammps) -> None:
    """Test that .pt2 model with default_fparam works without providing fparam."""
    lammps.pair_style(f"deepmd {pt2_file.resolve()} aparam 0.25852028")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_default_fparam_explicit(lammps) -> None:
    """Test that explicit fparam still works with .pt2 default_fparam model."""
    lammps.pair_style(
        f"deepmd {pt2_file.resolve()} fparam 0.25852028 aparam 0.25852028"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_fparam_aparam(lammps) -> None:
    """Test .pt2 model without default_fparam (explicit fparam required)."""
    lammps.pair_style(
        f"deepmd {pt2_file_no_default.resolve()} fparam 0.25852028 aparam 0.25852028"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)
