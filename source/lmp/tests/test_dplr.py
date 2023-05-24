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
expected_e_sr = -28.27186179
expected_f_sr = np.array(
    [
        [0.17101694, -0.33151456, -0.48810198],
        [-0.60326789, 1.05723576, -1.00398772],
        [-0.45535577, -0.52665656, 0.56538784],
        [0.07412830, 0.06141962, 0.08719303],
        [0.25021427, 0.46746804, 0.10825434],
        [0.56326416, -0.72795230, 0.73125450],
    ]
)

expected_v_sr = np.array(
    [
        [
            -1.08360048,
            0.28561092,
            1.12787306,
            -0.27800706,
            -2.22370058,
            -0.60250734,
            1.08392293,
            -0.91343020,
            -1.69211612,
        ],
        [
            -2.67885180,
            0.45057804,
            -0.59798385,
            -0.03229647,
            -1.74227093,
            1.32518278,
            -0.60602871,
            0.71486893,
            -0.58607102,
        ],
        [
            0.80158098,
            -0.64019999,
            -1.07255047,
            1.23622085,
            1.77181377,
            -1.21983559,
            -0.81543862,
            1.05632482,
            1.30333566,
        ],
        [
            0.44888053,
            0.47838603,
            -0.44712259,
            -0.80436553,
            1.19194100,
            1.28550021,
            -0.73434597,
            -0.53774522,
            0.80885901,
        ],
        [
            1.61424595,
            -0.83680788,
            0.73008361,
            1.70236397,
            1.37033017,
            -0.11448440,
            0.10876112,
            -0.44403200,
            0.26187153,
        ],
        [
            1.08938933,
            1.09903360,
            -0.24784609,
            -0.98731504,
            0.68084248,
            -0.45950359,
            0.45558292,
            0.33836575,
            -0.05030076,
        ],
    ]
).reshape(6, 9)

expected_e_lr = -28.03317069

expected_e_kspace = 0.23869110

expected_f_lr = np.array(
    [
        [0.11696108, -0.42944256, -0.40878673],
        [-0.57239656, 1.17989674, -1.00201720],
        [-0.33386296, -0.48158818, 0.40810781],
        [0.05039098, 0.17663026, 0.14472650],
        [0.08849454, 0.38712960, 0.06775784],
        [0.65041292, -0.83262585, 0.79021177],
    ]
)

expected_WC = np.array(
    [
        [1.21548264, 1.18751350, 1.02680021],
        [0.99406312, 3.35460815, 1.31511552],
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
