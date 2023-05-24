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
    write_lmp_data_full,
)

pbtxt_file = Path(__file__).parent / "lrmodel.pbtxt"
pb_file = Path(__file__).parent / "lrmodel.pb"
dipole_pbtxt_file = Path(__file__).parent / "lrdipole.pbtxt"
dipole_pb_file = Path(__file__).parent / "lrdipole.pb"
data_file = Path(__file__).parent / "data.lmp"

# this is as the same as python and c++ tests, test_deeppot_a.py
expected_e_sr = -39.06006411
expected_f_sr = np.array(
    [
        [0.02437349, 0.02334234, -0.01622272],
        [-0.03321954, -0.03065614, 0.02509050],
        [-0.03660466, 0.06395036, 0.05844523],
        [0.02531325, -0.08482784, -0.05192437],
        [0.08902521, -0.02177398, 0.02264204],
        [-0.06888775, 0.04996526, -0.03803068],
    ]
)

expected_v_sr = np.array(
    [
        [-0.01323848, -0.01246206, 0.01686948, -0.00907439, -0.01248319, 0.00355220, 0.01820345, 0.00151631, -0.02234572],                                                                           
        [-0.03193491, -0.01007088, -0.00618501, -0.00667596, -0.01722477, 0.01334466, -0.00681007, 0.02138824, -0.00598392],                                                                         
        [0.03882108, 0.00138573, -0.05036723, -0.03523660, -0.01812151, 0.04178336, -0.05812888, -0.00574756, 0.07129326],                                                                           
        [-0.00087744, -0.02754943, -0.00508057, 0.00421741, 0.11342152, 0.02018284, 0.00159304, 0.06168723, 0.01173648],                                                                             
        [0.11628843, 0.03910325, 0.00634673, -0.01018783, -0.01296546, -0.00065149, 0.02437827, 0.01124829, 0.00109516],                                                                             
        [0.03060390, -0.07301216, 0.03703715, -0.02564820, 0.05836682, -0.03214814, 0.01938474, -0.04402906, 0.02329266], 
    ]
).reshape(6, 9)

expected_e_lr = -34.20096245

expected_e_kspace = 4.85910166

expected_f_lr = np.array(
    [
        [1.18711962, 0.30528745, 2.39025970],
        [-1.24044330, 1.03239184, 5.66215601],
        [-0.52069124, 4.23392860, -0.69948148],
        [0.76651763, 0.79875349, -2.71153585],
        [1.05397892, -4.34621351, -1.85614889],
        [-1.24648163, -2.02414786, -2.78524948],
    ]
)

expected_WC = np.array(
    [
        [1.40621267, 1.50482158, 0.83444109],
        [0.82897365, 3.01863596, 1.40055418],
    ]
)

box = np.array([0, 20, 0, 20, 0, 20, 0, 0, 0])
coord = np.array(
    [
        [1.25545000, 1.27562200, 0.98873000],
        [0.96101000, 3.25750000, 1.33494000],
        [0.66417000, 1.31153700, 1.74354000],
        [1.29187000, 0.33436000, 0.73085000],
        [1.88885000, 3.51130000, 1.42444000],
        [0.51617000, 4.04330000, 0.90904000],
        [1.25545000, 1.27562200, 0.98873000],
        [0.96101000, 3.25750000, 1.33494000],
    ]
)
mol_list = np.array([1, 2, 1, 1, 2, 2, 1, 2])
type_OH = np.array([1, 1, 2, 2, 2, 2, 3, 3])
charge = np.array([6, 6, 1, 1, 1, 1, -8, -8])
bond_list = (((1, 7), (2, 8)),)
mass_list = np.array([15.99940, 1.00794, 15.99940])
beta = 0.4
mesh = 10


# https://github.com/lammps/lammps/blob/1e1311cf401c5fc2614b5d6d0ff3230642b76597/src/update.cpp#L193
nktv2p = 1.6021765e6

sp.check_output(
    "{} -m deepmd convert-from pbtxt -i {} -o {}".format(
        sys.executable,
        pbtxt_file.resolve(),
        pb_file.resolve(),
    ).split()
)


def setup_module():
    write_lmp_data_full(
        box, coord, mol_list, type_OH, charge, data_file, bond_list, mass_list
    )


def teardown_module():
    os.remove(data_file)


def _lammps(data_file) -> PyLammps:
    lammps = PyLammps()
    lammps.units("metal")
    lammps.boundary("p p p")
    lammps.atom_style("full")
    lammps.neighbor("0.2 bin")
    lammps.neigh_modify("every 1 delay 0 check no exclude type 1 3")
    lammps.read_data(data_file.resolve())
    lammps.timestep(0.0005)
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    yield _lammps(data_file=data_file)


def test_pair_deepmd_sr(lammps):
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e_sr)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f_sr[ii])
    lammps.run(1)


def test_pair_deepmd_sr_virial(lammps):
    lammps.group("real_atom type 1 2")
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("virial real_atom centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii+1}]")
    lammps.dump(
        "1 real_atom custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e_sr)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f_sr[ii])
    for ii in range(9):
        assert np.array(
            lammps.variables[f"virial{ii}"].value[:6]
        ) / nktv2p == pytest.approx(expected_v_sr[:, ii])


def test_pair_deepmd_lr(lammps):
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.bond_style("zero")
    lammps.bond_coeff("*")
    lammps.special_bonds("lj/coul 1 1 1 angle no")
    lammps.kspace_style("pppm/dplr 1e-5")
    lammps.kspace_modify(f"gewald {beta:.2f} diff ik mesh {mesh:d} {mesh:d} {mesh:d}")
    lammps.fix(f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1")
    lammps.fix_modify("0 virial yes")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr)
    assert lammps.eval("elong") == pytest.approx(expected_e_kspace)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f_lr[ii])
    for ii in range(2):
        assert lammps.atoms[6 + ii].position == pytest.approx(expected_WC[ii])
    lammps.run(1)
