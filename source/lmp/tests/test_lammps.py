import sys
import subprocess as sp
from pathlib import Path

import pytest
from lammps import PyLammps


pbtxt_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot-1.pbtxt"
pb_file = Path(__file__).parent / "graph.pb"
data_file = Path(__file__).parent.parent.parent.parent / "examples" / "water" / "lmp" / "water.lmp"

sp.check_output("{} -m deepmd convert-from pbtxt -i {} -o {}".format(
    sys.executable,
    pbtxt_file.resolve(),
    pb_file.resolve(),
    ).split())


@pytest.fixture
def lammps() -> PyLammps:
    lammps = PyLammps()
    lammps.plugin("load libdeepmd_lmp.so")
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
    yield lammps


def test_pair_deepmd(lammps):
    lammps.pair_style("deepmd {}".format(pb_file.resolve()))
    lammps.pair_coeff("* *")
    lammps.run(1)
    assert lammps.eval("pe") != 0.


def test_pair_deepmd_model_devi(lammps):
    lammps.pair_style("deepmd {} {} out_file md.out out_freq 1".format(pb_file.resolve(), pb_file.resolve()))
    lammps.pair_coeff("* *")
    lammps.run(1)
    assert lammps.eval("pe") != 0.


def test_pair_deepmd_model_devi_atomic_relative(lammps):
    lammps.pair_style("deepmd {} {} out_file md.out out_freq 1 atomic relative 1.0".format(pb_file.resolve(), pb_file.resolve()))
    lammps.pair_coeff("* *")
    lammps.run(1)
    assert lammps.eval("pe") != 0.


def test_pair_deepmd_model_devi_atomic_relative_v(lammps):
    lammps.pair_style("deepmd {} {} out_file md.out out_freq 1 atomic relative_v 1.0".format(pb_file.resolve(), pb_file.resolve()))
    lammps.pair_coeff("* *")
    lammps.run(1)
    assert lammps.eval("pe") != 0.
