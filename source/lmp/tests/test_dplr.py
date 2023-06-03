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
expected_e_sr = -40.56538550
expected_f_sr = np.array(
    [
        [0.35019748, 0.27802691, -0.38443156],
        [-0.42115581, -0.20474826, -0.02701100],
        [-0.56357653, 0.34154004, 0.78389512],
        [0.21023870, -0.60684635, -0.39875165],
        [0.78732106, 0.00610023, 0.17197636],
        [-0.36302488, 0.18592742, -0.14567727],
    ]
)

expected_v_sr = np.array(
    [
        [
            -0.053706180723,
            0.014597007250,
            0.067308761596,
            0.004757868347,
            -0.273575531966,
            -0.080634240285,
            0.066766981833,
            -0.082022798771,
            -0.103287504181,
        ],
        [
            -0.281471542200,
            0.111365734859,
            -0.101177745441,
            0.101943364083,
            -0.279013654810,
            0.167459867168,
            -0.104515477897,
            0.146708715161,
            -0.088287749985,
        ],
        [
            0.386519343671,
            -0.005263198500,
            -0.490071800364,
            -0.201548935270,
            -0.057174356686,
            0.252305096540,
            -0.534345834151,
            0.009882886080,
            0.674742623024,
        ],
        [
            0.005150888129,
            -0.224445564542,
            -0.056830156735,
            -0.024741632736,
            0.874091438655,
            0.229907695669,
            -0.012085211023,
            0.477509547430,
            0.122807797587,
        ],
        [
            0.918580719632,
            0.305393876038,
            0.062597464255,
            0.067643302415,
            -0.009009180940,
            0.011159413462,
            0.177132428210,
            0.068689297830,
            0.009370783847,
        ],
        [
            0.256886507259,
            -0.498997455657,
            0.264728675198,
            -0.245403567391,
            0.450588435107,
            -0.248950976847,
            0.153602311537,
            -0.289520792025,
            0.156365366477,
        ],
    ]
).reshape(6, 9)

expected_e_lr = -40.42063823

expected_e_kspace = 0.14474727

expected_f_lr = np.array(
    [
        [0.20445234, 0.27936500, -0.23179282],
        [-0.30801828, -0.15412533, -0.17021364],
        [-0.44078300, 0.34719898, 0.63462716],
        [0.22103191, -0.50831649, -0.36328848],
        [0.60333935, -0.04531002, 0.15891833],
        [-0.28002232, 0.08118786, -0.02825056],
    ]
)

expected_WC = np.array(
    [
        [1.22149689, 1.14543417, 1.01968026],
        [0.98893545, 3.39167201, 1.32303880],
    ]
)

expected_x_min_step1 = np.array(
    [
        [1.26321372, 1.28623039, 0.97992808],
        [0.94931355, 3.25164736, 1.32847644],
        [0.64743204, 1.32472127, 1.76763885],
        [1.30026330, 0.31505758, 0.71705476],
        [1.91176075, 3.50957943, 1.43047464],
        [0.50553664, 4.04638297, 0.90796723],
        [1.22952996, 1.15353755, 1.01171484],
        [0.97744342, 3.38790804, 1.31885315],
    ]
)

expected_e_min_step1 = -40.46708779

expected_e_kspace_min_step1 = 0.16209504

expected_f_min_step1 = np.array(
    [
        [-0.04230911, -0.09057170, -0.00644688],
        [0.03377777, 0.01767955, -0.18004346],
        [-0.10280108, 0.22936656, 0.17841170],
        [0.13239183, 0.02472290, -0.12924794],
        [0.11956397, -0.13352759, 0.08242015],
        [-0.14062339, -0.04766971, 0.05490644],
        [0.00000000, 0.00000000, 0.00000000],
        [0.00000000, 0.00000000, 0.00000000],
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
    for ii in range(2):
        assert lammps.atoms[6 + ii].position == pytest.approx(expected_WC[ii])
    assert lammps.eval("elong") == pytest.approx(expected_e_kspace)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(expected_f_lr[ii])
    lammps.run(1)

def test_min_dplr(lammps):
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.bond_style("zero")
    lammps.bond_coeff("*")
    lammps.special_bonds("lj/coul 1 1 1 angle no")
    lammps.kspace_style("pppm/dplr 1e-5")
    lammps.kspace_modify(f"gewald {beta:.2f} diff ik mesh {mesh:d} {mesh:d} {mesh:d}")
    lammps.fix(f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1")
    lammps.fix_modify("0 virial yes")
    lammps.min_style("cg")
    lammps.minimize("0 1.0e-6 2 2")
    for ii in range(8):
        assert lammps.atoms[ii].position == pytest.approx(expected_x_min_step1[ii])
    assert lammps.eval("pe") == pytest.approx(expected_e_min_step1)
    assert lammps.eval("elong") == pytest.approx(expected_e_kspace_min_step1)
    for ii in range(8):
        assert lammps.atoms[ii].force == pytest.approx(expected_f_min_step1[ii])