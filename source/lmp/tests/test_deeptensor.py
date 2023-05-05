import os
import subprocess as sp
import sys
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

pbtxt_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot.pbtxt"
pb_file = Path(__file__).parent / "graph.pb"
pbtxt_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deepdipole_new.pbtxt"
)
pb_file2 = Path(__file__).parent / "deepdipole_new.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
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
    "{} -m deepmd convert-from pbtxt -i {} -o {}".format(
        sys.executable,
        pbtxt_file.resolve(),
        pb_file.resolve(),
    ).split()
)

sp.check_output(
    "{} -m deepmd convert-from pbtxt -i {} -o {}".format(
        sys.executable,
        pbtxt_file2.resolve(),
        pb_file2.resolve(),
    ).split()
)


def setup_module():
    write_lmp_data(box, coord, type_OH, data_file)
    # TODO
    # write_lmp_data(box, coord, type_HO, data_type_map_file)


def teardown_module():
    os.remove(data_file)
    # os.remove(data_type_map_file)


def _lammps(data_file) -> PyLammps:
    lammps = PyLammps()
    lammps.units("metal")
    lammps.boundary("p p p")
    lammps.atom_style("atomic")
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
    yield _lammps(data_file=data_file)


# @pytest.fixture
# def lammps_type_map():
#    yield _lammps(data_file=data_type_map_file)


def test_compute_deeptensor_atom(lammps):
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute(f"tensor all deeptensor/atom {pb_file2.resolve()}")
    lammps.variable("tensor atom c_tensor[1]")
    lammps.dump("1 all custom 1 dump id c_tensor[1]")
    lammps.run(0)
    assert np.array(lammps.variables["tensor"].value) == pytest.approx(expected_d)
