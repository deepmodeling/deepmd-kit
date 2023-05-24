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
        [-0.013238483635, -0.012462055486, 0.016869484152, -0.009074392339, -0.012483190259, 0.003552200474, 0.018203454931, 0.001516312552, -0.022345724803],
        [-0.031934913129, -0.010070880681, -0.006185005340, -0.006675959730, -0.017224769573, 0.013344664308, -0.006810074553, 0.021388236966, -0.005983924688],
        [0.038821082617, 0.001385728016, -0.050367230633, -0.035236595571, -0.018121509881, 0.041783356973, -0.058128878561, -0.005747564608, 0.071293260639],
        [-0.000877438368, -0.027549433079, -0.005080572059, 0.004217413959, 0.113421521318, 0.020182843130, 0.001593040749, 0.061687231460, 0.011736478145],
        [0.116288432292, 0.039103247170, 0.006346731441, -0.010187826671, -0.012965462763, -0.000651490102, 0.024378272330, 0.011248286382, 0.001095162847],
        [0.030603900300, -0.073012164194, 0.037037149826, -0.025648197901, 0.058366818747, -0.032148135883, 0.019384742490, -0.044029063852, 0.023292657508],
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
