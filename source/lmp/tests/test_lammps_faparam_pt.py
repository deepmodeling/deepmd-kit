# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with default_fparam (PyTorch backend)."""

import os
from pathlib import (
    Path,
)

import numpy as np
import pytest
from lammps import (
    PyLammps,
)
from write_lmp_data import (
    write_lmp_data,
)

pth_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "fparam_aparam_default.pth"
)
data_file = Path(__file__).parent / "data.lmp"

# expected values from fparam_aparam_default.pth with default_fparam=[0.25852028]
expected_ae = np.array(
    [
        -1.038271223729637e-01,
        -7.285433579124989e-02,
        -9.467600492266426e-02,
        -1.467050207422953e-01,
        -7.660561676973243e-02,
        -7.277296000253175e-02,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        6.622266941151356e-02,
        5.278739714221517e-02,
        2.265728009692279e-02,
        -2.606048291367521e-02,
        -4.538812303131843e-02,
        1.058247419681242e-02,
        1.679392617013225e-01,
        -2.257826240741907e-03,
        -4.490146347357200e-02,
        -1.148364179422036e-01,
        -1.169790528013792e-02,
        6.140403441496690e-02,
        -8.078778123309406e-02,
        -5.838879041789346e-02,
        6.773641084621368e-02,
        -1.247724902386317e-02,
        6.494524782787654e-02,
        -1.174787360813438e-01,
    ]
).reshape(6, 3)

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
    """Test that model with default_fparam works without providing fparam."""
    lammps.pair_style(f"deepmd {pth_file.resolve()} aparam 0.25852028")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_default_fparam_explicit(lammps) -> None:
    """Test that explicit fparam still works with default_fparam model."""
    lammps.pair_style(
        f"deepmd {pth_file.resolve()} fparam 0.25852028 aparam 0.25852028"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)
