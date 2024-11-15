# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import subprocess as sp
import sys
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

pbtxt_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot.pbtxt"
pb_file = Path(__file__).parent / "graph.pb"
pbtxt_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deepdipole_new.pbtxt"
)
pb_file2 = Path(__file__).parent / "deepdipole_new.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
data_file_si = Path(__file__).parent / "data.si"
data_type_map_file = Path(__file__).parent / "data_type_map.lmp"

# this is as the same as python and c++ tests, test_deepdipole.py
expected_d = np.array(
    [
        -1.128427726201255282e-01,
        2.654103846999197880e-01,
        2.625816377288122533e-02,
        3.027556488877700680e-01,
        -7.475444785689989990e-02,
        1.526291164572509684e-01,
    ]
)
# sel_type is 0, it seems that it works here
expected_d[[1, 2, 4, 5]] = 0.0
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
# TODO
# type_HO = np.array([2, 1, 1, 2, 1, 1])


sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file.resolve()} -o {pb_file.resolve()}".split()
)

sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file2.resolve()} -o {pb_file2.resolve()}".split()
)


def setup_module() -> None:
    write_lmp_data(box, coord, type_OH, data_file)
    # TODO
    # write_lmp_data(box, coord, type_HO, data_type_map_file)
    write_lmp_data(
        box * constants.dist_metal2si,
        coord * constants.dist_metal2si,
        type_OH,
        data_file_si,
    )


def teardown_module() -> None:
    os.remove(data_file)
    # os.remove(data_type_map_file)
    os.remove(data_file_si)


def _lammps(data_file, units="metal") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
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


# @pytest.fixture
# def lammps_type_map():
#    yield _lammps(data_file=data_type_map_file)


@pytest.fixture
def lammps_si():
    lmp = _lammps(data_file=data_file_si, units="si")
    yield lmp
    lmp.close()


def test_compute_deeptensor_atom(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute(f"tensor all deeptensor/atom {pb_file2.resolve()}")
    lammps.variable("tensor atom c_tensor[1]")
    lammps.dump("1 all custom 1 dump id c_tensor[1]")
    lammps.run(0)
    idx_map = lammps.lmp.numpy.extract_atom("id") - 1
    assert np.array(lammps.variables["tensor"].value) == pytest.approx(
        expected_d[idx_map]
    )


def test_compute_deeptensor_atom_si(lammps_si) -> None:
    lammps_si.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_si.pair_coeff("* *")
    lammps_si.compute(f"tensor all deeptensor/atom {pb_file2.resolve()}")
    lammps_si.variable("tensor atom c_tensor[1]")
    lammps_si.dump("1 all custom 1 dump id c_tensor[1]")
    lammps_si.run(0)
    idx_map = lammps_si.lmp.numpy.extract_atom("id") - 1
    assert np.array(lammps_si.variables["tensor"].value) == pytest.approx(
        expected_d[idx_map] * constants.dist_metal2si
    )
