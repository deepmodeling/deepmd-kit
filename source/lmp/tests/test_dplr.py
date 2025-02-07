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
    write_lmp_data_full,
)

pbtxt_file = Path(__file__).parent / "lrmodel.pbtxt"
pb_file = Path(__file__).parent / "lrmodel.pb"
dipole_pbtxt_file = Path(__file__).parent / "lrdipole.pbtxt"
dipole_pb_file = Path(__file__).parent / "lrdipole.pb"
data_file = Path(__file__).parent / "data.lmp"
data_file2 = Path(__file__).parent / "data.lmp2"
data_file_si = Path(__file__).parent / "data.si"
data_type_map_file = Path(__file__).parent / "data_type_map.lmp"

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

expected_v_sr = -np.array(
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

expected_evdwl_lr_efield_constant = -40.56538550
expected_e_efield_constant = -0.00813751
expected_e_lr_efield_constant = -40.57352302
expected_f_lr_efield_constant = np.array(
    [
        [0.47635071, 0.15088380, -1.20378471],
        [-0.52759976, -0.01182856, -0.72773815],
        [-0.70317794, 0.29171446, 1.22375302],
        [0.20500683, -0.50210283, -0.04263579],
        [0.82463041, 0.04231172, 0.47856560],
        [-0.27521024, 0.02902140, 0.27184004],
    ]
)

expected_evdwl_lr_efield_constant_step1 = -40.56855711
expected_e_efield_constant_step1 = -0.00924204
expected_e_lr_efield_constant_step1 = -40.57779915
expected_x_lr_efield_constant_step1 = np.array(
    [
        [1.25548591, 1.27563337, 0.98863926],
        [0.96097023, 3.25749911, 1.33488514],
        [0.66332860, 1.31188606, 1.74500430],
        [1.29211530, 0.33375920, 0.73079898],
        [1.88983672, 3.51135063, 1.42501264],
        [0.51584069, 4.04333473, 0.90936527],
        [1.22151773, 1.14543776, 1.01968264],
        [0.98890877, 3.39167719, 1.32307805],
    ]
)
expected_f_lr_efield_constant_step1 = np.array(
    [
        [0.47169787, 0.14277839, -1.20084613],
        [-0.51870331, -0.00742992, -0.72827444],
        [-0.69568128, 0.28776665, 1.21423596],
        [0.20225942, -0.48947283, -0.03600238],
        [0.81231903, 0.04010359, 0.47727588],
        [-0.27189173, 0.02625411, 0.27361111],
    ]
)

expected_evdwl_lr_efield_variable = -40.56538550
expected_e_efield_variable = 0
expected_e_lr_efield_variable = -40.56538550
expected_f_lr_efield_variable = np.array(
    [
        [0.35019748, 0.27802691, -0.38443156],
        [-0.42115581, -0.20474826, -0.02701100],
        [-0.56357653, 0.34154004, 0.78389512],
        [0.21023870, -0.60684635, -0.39875165],
        [0.78732106, 0.00610023, 0.17197636],
        [-0.36302488, 0.18592742, -0.14567727],
    ]
)

expected_evdwl_lr_efield_variable_step1 = -40.56834245
expected_e_efield_variable_step1 = -0.00835811
expected_e_lr_efield_variable_step1 = -40.57670056
expected_x_lr_efield_variable_step1 = np.array(
    [
        [1.25547640, 1.27564296, 0.98870102],
        [0.96097825, 3.25748457, 1.33493796],
        [0.66349564, 1.31194568, 1.74447798],
        [1.29212156, 0.33363387, 0.73037287],
        [1.88979208, 3.51130730, 1.42464578],
        [0.51573562, 4.04352247, 0.90886569],
        [1.22151260, 1.14544923, 1.01967090],
        [0.98891853, 3.39165617, 1.32305887],
    ]
)
expected_f_lr_efield_variable_step1 = np.array(
    [
        [0.472961703581, 0.139180609625, -1.202763462854],
        [-0.522013780229, -0.002650184425, -0.730891801410],
        [-0.697058457903, 0.288298050924, 1.215594205400],
        [0.202354938967, -0.486587189234, -0.035481805009],
        [0.811998296976, 0.040625078957, 0.476748222312],
        [-0.268242701392, 0.021133634152, 0.276794641561],
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
box2 = np.array([0, 20, 0, 3.2575, 0, 20, 0, 0, 0])
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
type_HO = np.array([2, 2, 1, 1, 1, 1, 3, 3])
charge = np.array([6, 6, 1, 1, 1, 1, -8, -8])
bond_list = (((1, 7), (2, 8)),)
mass_list = np.array([15.99940, 1.00794, 15.99940])
beta = 0.4
mesh = 10


sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file.resolve()} -o {pb_file.resolve()}".split()
)


def setup_module() -> None:
    write_lmp_data_full(
        box, coord, mol_list, type_OH, charge, data_file, bond_list, mass_list
    )
    write_lmp_data_full(
        box2, coord, mol_list, type_OH, charge, data_file2, bond_list, mass_list
    )
    write_lmp_data_full(
        box, coord, mol_list, type_HO, charge, data_type_map_file, bond_list, mass_list
    )
    write_lmp_data_full(
        box * constants.dist_metal2si,
        coord * constants.dist_metal2si,
        mol_list,
        type_OH,
        charge * constants.charge_metal2si,
        data_file_si,
        bond_list,
        mass_list * constants.mass_metal2si,
    )


def teardown_module() -> None:
    os.remove(data_file)
    os.remove(data_file2)
    os.remove(data_type_map_file)
    os.remove(data_file_si)


def _lammps(data_file, exclude_type="1 3", units="metal") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("full")
    if units == "metal" or units == "real":
        lammps.neighbor("0.2 bin")
    elif units == "si":
        lammps.neighbor("2.0e-11 bin")
    else:
        raise ValueError("units should be metal, real, or si")
    lammps.neigh_modify("every 1 delay 0 check no exclude type " + exclude_type)
    lammps.read_data(data_file.resolve())
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


@pytest.fixture
def lammps2():
    lmp = _lammps(data_file=data_file2)
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_type_map():
    lmp = _lammps(data_file=data_type_map_file, exclude_type="2 3")
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_si():
    lmp = _lammps(data_file=data_file_si, units="si")
    yield lmp
    lmp.close()


def test_pair_deepmd_sr(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e_sr)
    id_list = lammps.lmp.numpy.extract_atom("id")
    for ii in range(6):
        assert lammps.atoms[np.where(id_list == (ii + 1))[0][0]].force == pytest.approx(
            expected_f_sr[ii]
        )
    lammps.run(1)


def test_pair_deepmd_sr_virial(lammps) -> None:
    lammps.group("real_atom type 1 2")
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("virial real_atom centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 real_atom custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.dump_modify("1 sort id")
    lammps.run(0)
    id_list = lammps.lmp.numpy.extract_atom("id")
    idx_list = [np.where(id_list == i)[0][0] for i in range(1, 7)]
    assert lammps.eval("pe") == pytest.approx(expected_e_sr)
    for ii in range(6):
        assert lammps.atoms[np.where(id_list == (ii + 1))[0][0]].force == pytest.approx(
            expected_f_sr[ii]
        )
    for ii in range(9):
        assert np.array(lammps.variables[f"virial{ii}"].value)[
            idx_list
        ] / constants.nktv2p == pytest.approx(expected_v_sr[:, ii])
    os.remove("dump")


def test_pair_deepmd_lr(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.bond_style("zero")
    lammps.bond_coeff("*")
    lammps.special_bonds("lj/coul 1 1 1 angle no")
    lammps.kspace_style("pppm/dplr 1e-5")
    lammps.kspace_modify(f"gewald {beta:.2f} diff ik mesh {mesh:d} {mesh:d} {mesh:d}")
    lammps.fix(
        f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1 pair_deepmd_index 0"
    )
    lammps.fix_modify("0 virial yes")
    lammps.run(0)
    for ii in range(8):
        if lammps.atoms[ii].id > 6:
            assert lammps.atoms[ii].position == pytest.approx(
                expected_WC[lammps.atoms[ii].id - 7]
            )
    assert lammps.eval("elong") == pytest.approx(expected_e_kspace)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr)
    for ii in range(8):
        if lammps.atoms[ii].id <= 6:
            assert lammps.atoms[ii].force == pytest.approx(
                expected_f_lr[lammps.atoms[ii].id - 1]
            )
    lammps.run(1)


def test_pair_deepmd_lr_run0(lammps2) -> None:
    lammps2.pair_style(f"deepmd {pb_file.resolve()}")
    lammps2.pair_coeff("* *")
    lammps2.bond_style("zero")
    lammps2.bond_coeff("*")
    lammps2.special_bonds("lj/coul 1 1 1 angle no")
    lammps2.kspace_style("pppm/dplr 1e-5")
    lammps2.kspace_modify(f"gewald {beta:.2f} diff ik mesh {mesh:d} {mesh:d} {mesh:d}")
    lammps2.fix(f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1")
    lammps2.fix_modify("0 virial yes")
    lammps2.run(0)
    lammps2.run(0)


def test_pair_deepmd_lr_efield_constant(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.bond_style("zero")
    lammps.bond_coeff("*")
    lammps.special_bonds("lj/coul 1 1 1 angle no")
    lammps.fix(
        f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1 efield 0 0 1"
    )
    lammps.fix_modify("0 energy yes virial yes")
    lammps.run(0)
    id_list = lammps.lmp.numpy.extract_atom("id")
    assert lammps.eval("evdwl") == pytest.approx(expected_evdwl_lr_efield_constant)
    assert lammps.eval("f_0") == pytest.approx(expected_e_efield_constant)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr_efield_constant)
    for ii in range(6):
        assert lammps.atoms[np.where(id_list == (ii + 1))[0][0]].force == pytest.approx(
            expected_f_lr_efield_constant[ii]
        )
    lammps.run(1)
    assert lammps.eval("evdwl") == pytest.approx(
        expected_evdwl_lr_efield_constant_step1
    )
    assert lammps.eval("f_0") == pytest.approx(expected_e_efield_constant_step1)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr_efield_constant_step1)
    for ii in range(8):
        assert lammps.atoms[
            np.where(id_list == (ii + 1))[0][0]
        ].position == pytest.approx(expected_x_lr_efield_constant_step1[ii])
    for ii in range(6):
        assert lammps.atoms[np.where(id_list == (ii + 1))[0][0]].force == pytest.approx(
            expected_f_lr_efield_constant_step1[ii]
        )


def test_pair_deepmd_lr_efield_variable(lammps) -> None:
    lammps.variable("EFIELD_Z equal 2*sin(2*PI*time/0.006)")
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.bond_style("zero")
    lammps.bond_coeff("*")
    lammps.special_bonds("lj/coul 1 1 1 angle no")
    lammps.fix(
        f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1 efield 0 0 v_EFIELD_Z"
    )
    lammps.fix_modify("0 energy yes virial yes")
    lammps.run(0)
    id_list = lammps.lmp.numpy.extract_atom("id")
    assert lammps.eval("evdwl") == pytest.approx(expected_evdwl_lr_efield_variable)
    assert lammps.eval("f_0") == pytest.approx(expected_e_efield_variable)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr_efield_variable)
    for ii in range(6):
        assert lammps.atoms[np.where(id_list == (ii + 1))[0][0]].force == pytest.approx(
            expected_f_lr_efield_variable[ii]
        )
    lammps.run(1)
    assert lammps.eval("evdwl") == pytest.approx(
        expected_evdwl_lr_efield_variable_step1
    )
    assert lammps.eval("f_0") == pytest.approx(expected_e_efield_variable_step1)
    assert lammps.eval("pe") == pytest.approx(expected_e_lr_efield_variable_step1)
    for ii in range(8):
        assert lammps.atoms[
            np.where(id_list == (ii + 1))[0][0]
        ].position == pytest.approx(expected_x_lr_efield_variable_step1[ii])
    for ii in range(6):
        assert lammps.atoms[np.where(id_list == (ii + 1))[0][0]].force == pytest.approx(
            expected_f_lr_efield_variable_step1[ii]
        )


def test_min_dplr(lammps) -> None:
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
        assert lammps.atoms[ii].position == pytest.approx(
            expected_x_min_step1[lammps.atoms[ii].id - 1]
        )
    assert lammps.eval("pe") == pytest.approx(expected_e_min_step1)
    assert lammps.eval("elong") == pytest.approx(expected_e_kspace_min_step1)
    for ii in range(8):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f_min_step1[lammps.atoms[ii].id - 1]
        )


def test_pair_deepmd_lr_type_map(lammps_type_map) -> None:
    lammps_type_map.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_type_map.pair_coeff("* * H O")
    lammps_type_map.bond_style("zero")
    lammps_type_map.bond_coeff("*")
    lammps_type_map.special_bonds("lj/coul 1 1 1 angle no")
    lammps_type_map.kspace_style("pppm/dplr 1e-5")
    lammps_type_map.kspace_modify(
        f"gewald {beta:.2f} diff ik mesh {mesh:d} {mesh:d} {mesh:d}"
    )
    lammps_type_map.fix(
        f"0 all dplr model {pb_file.resolve()} type_associate 2 3 bond_type 1"
    )
    lammps_type_map.fix_modify("0 virial yes")
    lammps_type_map.run(0)
    for ii in range(8):
        if lammps_type_map.atoms[ii].id > 6:
            assert lammps_type_map.atoms[ii].position == pytest.approx(
                expected_WC[lammps_type_map.atoms[ii].id - 7]
            )
    assert lammps_type_map.eval("elong") == pytest.approx(expected_e_kspace)
    assert lammps_type_map.eval("pe") == pytest.approx(expected_e_lr)
    for ii in range(8):
        if lammps_type_map.atoms[ii].id <= 6:
            assert lammps_type_map.atoms[ii].force == pytest.approx(
                expected_f_lr[lammps_type_map.atoms[ii].id - 1]
            )
    lammps_type_map.run(1)


def test_pair_deepmd_lr_si(lammps_si) -> None:
    lammps_si.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_si.pair_coeff("* *")
    lammps_si.bond_style("zero")
    lammps_si.bond_coeff("*")
    lammps_si.special_bonds("lj/coul 1 1 1 angle no")
    lammps_si.kspace_style("pppm/dplr 1e-5")
    lammps_si.kspace_modify(
        f"gewald {beta / constants.dist_metal2si:.6e} diff ik mesh {mesh:d} {mesh:d} {mesh:d}"
    )
    lammps_si.fix(
        f"0 all dplr model {pb_file.resolve()} type_associate 1 3 bond_type 1"
    )
    lammps_si.fix_modify("0 virial yes")
    lammps_si.run(0)
    for ii in range(8):
        if lammps_si.atoms[ii].id > 6:
            assert lammps_si.atoms[ii].position == pytest.approx(
                expected_WC[lammps_si.atoms[ii].id - 7] * constants.dist_metal2si
            )
    assert lammps_si.eval("elong") == pytest.approx(
        expected_e_kspace * constants.ener_metal2si
    )
    assert lammps_si.eval("pe") == pytest.approx(
        expected_e_lr * constants.ener_metal2si
    )
    for ii in range(8):
        if lammps_si.atoms[ii].id <= 6:
            assert lammps_si.atoms[ii].force == pytest.approx(
                expected_f_lr[lammps_si.atoms[ii].id - 1] * constants.force_metal2si
            )
    lammps_si.run(1)
