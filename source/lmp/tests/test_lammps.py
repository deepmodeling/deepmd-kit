import sys
import subprocess as sp
from pathlib import Path

import pytest
import numpy as np
from lammps import PyLammps


pbtxt_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot.pbtxt"
pbtxt_file2 = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot-1.pbtxt"
pb_file = Path(__file__).parent / "graph.pb"
pb_file2 = Path(__file__).parent / "graph2.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
md_file = Path(__file__).parent / "md.out"

# this is as the same as python and c++ tests, test_deeppot_a.py
expected_ae = np.array([
    -9.275780747115504710e+01,-1.863501786584258468e+02,-1.863392472863538103e+02,-9.279281325486221021e+01,-1.863671545232153903e+02,-1.863619822847602165e+02,
])
expected_e = np.sum(expected_ae)
expected_f = np.array([
    -3.034045420701179663e-01,8.405844663871177014e-01,7.696947487118485642e-02,7.662001266663505117e-01,-1.880601391333554251e-01,-6.183333871091722944e-01,-5.036172391059643427e-01,-6.529525836149027151e-01,5.432962643022043459e-01,6.382357912332115024e-01,-1.748518296794561167e-01,3.457363524891907125e-01,1.286482986991941552e-03,3.757251165286925043e-01,-5.972588700887541124e-01,-5.987006197104716154e-01,-2.004450304880958100e-01,2.495901655353461868e-01
]).reshape(6, 3)

expected_f2 = np.array([
    [-0.6454949 , 1.72457783, 0.18897958],
    [ 1.68936514,-0.36995299,-1.36044464],
    [-1.09902692,-1.35487928, 1.17416702],
    [ 1.68426111,-0.50835585, 0.98340415],
    [ 0.05771758, 1.12515818,-1.77561531],
    [-1.686822  ,-0.61654789, 0.78950921],
])


sp.check_output("{} -m deepmd convert-from pbtxt -i {} -o {}".format(
    sys.executable,
    pbtxt_file.resolve(),
    pb_file.resolve(),
    ).split())
sp.check_output("{} -m deepmd convert-from pbtxt -i {} -o {}".format(
    sys.executable,
    pbtxt_file2.resolve(),
    pb_file2.resolve(),
    ).split())



@pytest.fixture
def lammps() -> PyLammps:
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
    yield lammps


def test_pair_deepmd(lammps):
    lammps.pair_style("deepmd {}".format(pb_file.resolve()))
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f[ii])
    lammps.run(1)


def test_pair_deepmd_model_devi(lammps):
    lammps.pair_style("deepmd {} {} out_file {} out_freq 1 atomic".format(pb_file.resolve(), pb_file2.resolve(), md_file.resolve()))
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f[ii])
    # load model devi
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))


def test_pair_deepmd_model_devi_atomic_relative(lammps):
    relative = 1.0
    lammps.pair_style("deepmd {} {} out_file {} out_freq 1 atomic relative {}".format(pb_file.resolve(), pb_file2.resolve(), md_file.resolve(), relative))
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f[ii])
    # load model devi
    md = np.loadtxt(md_file.resolve())
    norm = np.linalg.norm(np.mean([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f /= norm + relative
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))


def test_pair_deepmd_model_devi_atomic_relative_v(lammps):
    relative = 1.0
    lammps.pair_style("deepmd {} {} out_file {} out_freq 1 atomic relative_v {}".format(pb_file.resolve(), pb_file2.resolve(), md_file.resolve(), relative))
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f[ii])
    #md = np.loadtxt(md_file.resolve())
    # TODO: how to get the virial?
