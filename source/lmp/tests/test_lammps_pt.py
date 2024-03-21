# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import os
import shutil
import subprocess as sp
import sys
import tempfile
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

pbtxt_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot-1.pbtxt"
)
pb_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_sea.pth"
pb_file2 = Path(__file__).parent / "graph2.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
data_file_si = Path(__file__).parent / "data.si"
data_type_map_file = Path(__file__).parent / "data_type_map.lmp"
md_file = Path(__file__).parent / "md.out"

# this is as the same as python and c++ tests, test_deeppot_a.py
expected_ae = np.array(
    [
        -93.016873944029,
        -185.923296645958,
        -185.927096544970,
        -93.019371018039,
        -185.926179995548,
        -185.924351901852,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        0.006277522211,
        -0.001117962774,
        0.000618580445,
        0.009928999655,
        0.003026035654,
        -0.006941982227,
        0.000667853212,
        -0.002449963843,
        0.006506463508,
        -0.007284129115,
        0.000530662205,
        -0.000028806821,
        0.000068097781,
        0.006121331983,
        -0.009019754602,
        -0.009658343745,
        -0.006110103225,
        0.008865499697,
    ]
).reshape(6, 3)

expected_f2 = np.array(
    [
        [-0.6454949, 1.72457783, 0.18897958],
        [1.68936514, -0.36995299, -1.36044464],
        [-1.09902692, -1.35487928, 1.17416702],
        [1.68426111, -0.50835585, 0.98340415],
        [0.05771758, 1.12515818, -1.77561531],
        [-1.686822, -0.61654789, 0.78950921],
    ]
)

expected_v = -np.array(
    [
        -0.000155238009,
        0.000116605516,
        -0.007869862476,
        0.000465578340,
        0.008182547185,
        -0.002398713212,
        -0.008112887338,
        -0.002423738425,
        0.007210716605,
        -0.019203504012,
        0.001724938709,
        0.009909211091,
        0.001153857542,
        -0.001600015103,
        -0.000560024090,
        0.010727836276,
        -0.001034836404,
        -0.007973454377,
        -0.021517399106,
        -0.004064359664,
        0.004866398692,
        -0.003360038617,
        -0.007241406162,
        0.005920941051,
        0.004899151657,
        0.006290788591,
        -0.006478820311,
        0.001921504710,
        0.001313470921,
        -0.000304091236,
        0.001684345981,
        0.004124109256,
        -0.006396084465,
        -0.000701095618,
        -0.006356507032,
        0.009818550859,
        -0.015230664587,
        -0.000110244376,
        0.000690319396,
        0.000045953023,
        -0.005726548770,
        0.008769818495,
        -0.000572380210,
        0.008860603423,
        -0.013819348050,
        -0.021227082558,
        -0.004977781343,
        0.006646239696,
        -0.005987066507,
        -0.002767831232,
        0.003746502525,
        0.007697590397,
        0.003746130152,
        -0.005172634748,
    ]
).reshape(6, 9)
expected_v2 = -np.array(
    [
        [
            -0.70008436,
            -0.06399891,
            0.63678391,
            -0.07642171,
            -0.70580035,
            0.20506145,
            0.64098364,
            0.20305781,
            -0.57906794,
        ],
        [
            -0.6372635,
            0.14315552,
            0.51952246,
            0.04604049,
            -0.06003681,
            -0.02688702,
            0.54489318,
            -0.10951559,
            -0.43730539,
        ],
        [
            -0.25090748,
            -0.37466262,
            0.34085833,
            -0.26690852,
            -0.37676917,
            0.29080825,
            0.31600481,
            0.37558276,
            -0.33251064,
        ],
        [
            -0.80195614,
            -0.10273138,
            0.06935364,
            -0.10429256,
            -0.29693811,
            0.45643496,
            0.07247872,
            0.45604679,
            -0.71048816,
        ],
        [
            -0.03840668,
            -0.07680205,
            0.10940472,
            -0.02374189,
            -0.27610266,
            0.4336071,
            0.02465248,
            0.4290638,
            -0.67496763,
        ],
        [
            -0.61475065,
            -0.21163135,
            0.26652929,
            -0.26134659,
            -0.11560267,
            0.15415902,
            0.34343952,
            0.1589482,
            -0.21370642,
        ],
    ]
).reshape(6, 9)

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
type_HO = np.array([2, 1, 1, 2, 1, 1])


sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file2.resolve()} -o {pb_file2.resolve()}".split()
)


def setup_module():
    write_lmp_data(box, coord, type_OH, data_file)
    write_lmp_data(box, coord, type_HO, data_type_map_file)
    write_lmp_data(
        box * constants.dist_metal2si,
        coord * constants.dist_metal2si,
        type_OH,
        data_file_si,
    )


def teardown_module():
    os.remove(data_file)
    os.remove(data_type_map_file)


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


@pytest.fixture
def lammps_type_map():
    lmp = _lammps(data_file=data_type_map_file)
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_real():
    lmp = _lammps(data_file=data_file, units="real")
    yield lmp
    lmp.close()


@pytest.fixture
def lammps_si():
    lmp = _lammps(data_file=data_file_si, units="si")
    yield lmp
    lmp.close()


def test_pair_deepmd(lammps):
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_virial(lammps):
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii+1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    idx_map = lammps.lmp.numpy.extract_atom("id") - 1
    for ii in range(9):
        assert np.array(
            lammps.variables[f"virial{ii}"].value
        ) / constants.nktv2p == pytest.approx(expected_v[idx_map, ii])


def test_pair_deepmd_model_devi(lammps):
    lammps.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))


def test_pair_deepmd_model_devi_virial(lammps):
    lammps.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic"
    )
    lammps.pair_coeff("* *")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii+1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    idx_map = lammps.lmp.numpy.extract_atom("id") - 1
    for ii in range(9):
        assert np.array(
            lammps.variables[f"virial{ii}"].value
        ) / constants.nktv2p == pytest.approx(expected_v[idx_map, ii])
    # load model devi
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))


def test_pair_deepmd_model_devi_atomic_relative(lammps):
    relative = 1.0
    lammps.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic relative {relative}"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    norm = np.linalg.norm(np.mean([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f /= norm + relative
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))


def test_pair_deepmd_model_devi_atomic_relative_v(lammps):
    relative = 1.0
    lammps.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic relative_v {relative}"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    norm = (
        np.abs(
            np.mean([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0)
        )
        / 6
    )
    expected_md_v /= norm + relative
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))


def test_pair_deepmd_type_map(lammps_type_map):
    lammps_type_map.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_type_map.pair_coeff("* * H O")
    lammps_type_map.run(0)
    assert lammps_type_map.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps_type_map.atoms[ii].force == pytest.approx(
            expected_f[lammps_type_map.atoms[ii].id - 1]
        )
    lammps_type_map.run(1)


def test_pair_deepmd_real(lammps_real):
    lammps_real.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_real.pair_coeff("* *")
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    lammps_real.run(1)


def test_pair_deepmd_virial_real(lammps_real):
    lammps_real.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_real.pair_coeff("* *")
    lammps_real.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps_real.variable(f"virial{jj} atom c_virial[{ii+1}]")
    lammps_real.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    idx_map = lammps_real.lmp.numpy.extract_atom("id") - 1
    for ii in range(9):
        assert np.array(
            lammps_real.variables[f"virial{ii}"].value
        ) / constants.nktv2p_real == pytest.approx(
            expected_v[idx_map, ii] * constants.ener_metal2real
        )


def test_pair_deepmd_model_devi_real(lammps_real):
    lammps_real.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic"
    )
    lammps_real.pair_coeff("* *")
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f * constants.force_metal2real)
    assert md[4] == pytest.approx(np.max(expected_md_f) * constants.force_metal2real)
    assert md[5] == pytest.approx(np.min(expected_md_f) * constants.force_metal2real)
    assert md[6] == pytest.approx(np.mean(expected_md_f) * constants.force_metal2real)
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v) * constants.ener_metal2real)
    assert md[2] == pytest.approx(np.min(expected_md_v) * constants.ener_metal2real)
    assert md[3] == pytest.approx(
        np.sqrt(np.mean(np.square(expected_md_v))) * constants.ener_metal2real
    )


def test_pair_deepmd_model_devi_virial_real(lammps_real):
    lammps_real.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic"
    )
    lammps_real.pair_coeff("* *")
    lammps_real.compute("virial all centroid/stress/atom NULL pair")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps_real.variable(f"virial{jj} atom c_virial[{ii+1}]")
    lammps_real.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    idx_map = lammps_real.lmp.numpy.extract_atom("id") - 1
    for ii in range(9):
        assert np.array(
            lammps_real.variables[f"virial{ii}"].value
        ) / constants.nktv2p_real == pytest.approx(
            expected_v[idx_map, ii] * constants.ener_metal2real
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f * constants.force_metal2real)
    assert md[4] == pytest.approx(np.max(expected_md_f) * constants.force_metal2real)
    assert md[5] == pytest.approx(np.min(expected_md_f) * constants.force_metal2real)
    assert md[6] == pytest.approx(np.mean(expected_md_f) * constants.force_metal2real)
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v) * constants.ener_metal2real)
    assert md[2] == pytest.approx(np.min(expected_md_v) * constants.ener_metal2real)
    assert md[3] == pytest.approx(
        np.sqrt(np.mean(np.square(expected_md_v))) * constants.ener_metal2real
    )


def test_pair_deepmd_model_devi_atomic_relative_real(lammps_real):
    relative = 1.0
    lammps_real.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic relative {relative * constants.force_metal2real}"
    )
    lammps_real.pair_coeff("* *")
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    norm = np.linalg.norm(np.mean([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f /= norm + relative
    assert md[7:] == pytest.approx(expected_md_f * constants.force_metal2real)
    assert md[4] == pytest.approx(np.max(expected_md_f) * constants.force_metal2real)
    assert md[5] == pytest.approx(np.min(expected_md_f) * constants.force_metal2real)
    assert md[6] == pytest.approx(np.mean(expected_md_f) * constants.force_metal2real)
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v) * constants.ener_metal2real)
    assert md[2] == pytest.approx(np.min(expected_md_v) * constants.ener_metal2real)
    assert md[3] == pytest.approx(
        np.sqrt(np.mean(np.square(expected_md_v))) * constants.ener_metal2real
    )


def test_pair_deepmd_model_devi_atomic_relative_v_real(lammps_real):
    relative = 1.0
    lammps_real.pair_style(
        f"deepmd {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic relative_v {relative * constants.ener_metal2real}"
    )
    lammps_real.pair_coeff("* *")
    lammps_real.run(0)
    assert lammps_real.eval("pe") == pytest.approx(
        expected_e * constants.ener_metal2real
    )
    for ii in range(6):
        assert lammps_real.atoms[ii].force == pytest.approx(
            expected_f[lammps_real.atoms[ii].id - 1] * constants.force_metal2real
        )
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    assert md[7:] == pytest.approx(expected_md_f * constants.force_metal2real)
    assert md[4] == pytest.approx(np.max(expected_md_f) * constants.force_metal2real)
    assert md[5] == pytest.approx(np.min(expected_md_f) * constants.force_metal2real)
    assert md[6] == pytest.approx(np.mean(expected_md_f) * constants.force_metal2real)
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    norm = (
        np.abs(
            np.mean([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0)
        )
        / 6
    )
    expected_md_v /= norm + relative
    assert md[1] == pytest.approx(np.max(expected_md_v) * constants.ener_metal2real)
    assert md[2] == pytest.approx(np.min(expected_md_v) * constants.ener_metal2real)
    assert md[3] == pytest.approx(
        np.sqrt(np.mean(np.square(expected_md_v))) * constants.ener_metal2real
    )


def test_pair_deepmd_si(lammps_si):
    lammps_si.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_si.pair_coeff("* *")
    lammps_si.run(0)
    assert lammps_si.eval("pe") == pytest.approx(expected_e * constants.ener_metal2si)
    for ii in range(6):
        assert lammps_si.atoms[ii].force == pytest.approx(
            expected_f[lammps_si.atoms[ii].id - 1] * constants.force_metal2si
        )
    lammps_si.run(1)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
@pytest.mark.parametrize(
    ("balance_args",),
    [(["--balance"],), ([],)],
)
# TODO: [BUG] pt: C++ interface throws errors when the number of ranks is larger than the number of GPUs
# terminate called after throwing an instance of 'c10::Error'
#   what():  CUDA error: invalid device ordinal
# CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
# Exception raised from c10_cuda_check_implementation at ../c10/cuda/CUDAException.cpp:44 (most recent call first):
# frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x6c (0x7f55c1b9fa0c in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libc10.so)
# frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xfa (0x7f55c1b498bc in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libc10.so)
# frame #2: c10::cuda::c10_cuda_check_implementation(int, char const*, char const*, int, bool) + 0x3cc (0x7f55c173201c in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libc10_cuda.so)
# frame #3: c10::cuda::ExchangeDevice(int) + 0x62 (0x7f55c1732542 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libc10_cuda.so)
# frame #4: <unknown function> + 0x2935c (0x7f55c16fe35c in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libc10_cuda.so)
# frame #5: <unknown function> + 0x12fc71d (0x7f5522c1771d in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cuda.so)
# frame #6: <unknown function> + 0x34ccdf5 (0x7f5524de7df5 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cuda.so)
# frame #7: <unknown function> + 0x34ccf84 (0x7f5524de7f84 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cuda.so)
# frame #8: at::_ops::empty_strided::redispatch(c10::DispatchKeySet, c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) + 0x107 (0x7f55779aefb7 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #9: <unknown function> + 0x2d23a0b (0x7f5577da3a0b in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #10: at::_ops::empty_strided::call(c10::ArrayRef<c10::SymInt>, c10::ArrayRef<c10::SymInt>, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>) + 0x1b9 (0x7f55779ff349 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #11: <unknown function> + 0x1c64e49 (0x7f5576ce4e49 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #12: at::native::_to_copy(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1af0 (0x7f55770962d0 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #13: <unknown function> + 0x2f5545f (0x7f5577fd545f in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #14: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x109 (0x7f55775f74b9 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #15: <unknown function> + 0x2d271fa (0x7f5577da71fa in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #16: at::_ops::_to_copy::redispatch(c10::DispatchKeySet, at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x109 (0x7f55775f74b9 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #17: <unknown function> + 0x46f3a45 (0x7f5579773a45 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #18: <unknown function> + 0x46f3f12 (0x7f5579773f12 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #19: at::_ops::_to_copy::call(at::Tensor const&, std::optional<c10::ScalarType>, std::optional<c10::Layout>, std::optional<c10::Device>, std::optional<bool>, bool, std::optional<c10::MemoryFormat>) + 0x1fe (0x7f557769565e in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #20: at::native::to(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0xf7 (0x7f557708dcd7 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #21: <unknown function> + 0x319275d (0x7f557821275d in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #22: at::_ops::to_device::call(at::Tensor const&, c10::Device, c10::ScalarType, bool, bool, std::optional<c10::MemoryFormat>) + 0x1ce (0x7f557785899e in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #23: torch::jit::Unpickler::readInstruction() + 0x1d5a (0x7f557aa190ca in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #24: torch::jit::Unpickler::run() + 0xa8 (0x7f557aa1a418 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #25: torch::jit::Unpickler::parse_ivalue() + 0x32 (0x7f557aa1bf92 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #26: torch::jit::readArchiveAndTensors(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::optional<std::function<c10::StrongTypePtr (c10::QualifiedName const&)> >, std::optional<std::function<c10::intrusive_ptr<c10::ivalue::Object, c10::detail::intrusive_target_default_null_type<c10::ivalue::Object> > (c10::StrongTypePtr const&, c10::IValue)> >, std::optional<c10::Device>, caffe2::serialize::PyTorchStreamReader&, c10::Type::SingletonOrSharedTypePtr<c10::Type> (*)(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&), std::shared_ptr<torch::jit::DeserializationStorageContext>) + 0x569 (0x7f557a9d5629 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #27: <unknown function> + 0x594a178 (0x7f557a9ca178 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #28: <unknown function> + 0x594cfc3 (0x7f557a9ccfc3 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #29: torch::jit::import_ir_module(std::shared_ptr<torch::jit::CompilationUnit>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::optional<c10::Device>, std::unordered_map<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::hash<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::equal_to<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >&, bool, bool) + 0x3df (0x7f557a9d2a1f in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #30: torch::jit::import_ir_module(std::shared_ptr<torch::jit::CompilationUnit>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::optional<c10::Device>, bool) + 0x92 (0x7f557a9d2cd2 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #31: torch::jit::load(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, std::optional<c10::Device>, bool) + 0xc0 (0x7f557a9d2de0 in /__w/deepmd-kit/deepmd-kit/libtorch/lib/libtorch_cpu.so)
# frame #32: deepmd::DeepPotPT::init(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x3d2 (0x7f55bf64f21a in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #33: deepmd::DeepPotPT::DeepPotPT(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xba (0x7f55bf64ed74 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #34: void __gnu_cxx::new_allocator<deepmd::DeepPotPT>::construct<deepmd::DeepPotPT, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(deepmd::DeepPotPT*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xa8 (0x7f55bf64d508 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #35: void std::allocator_traits<std::allocator<deepmd::DeepPotPT> >::construct<deepmd::DeepPotPT, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(std::allocator<deepmd::DeepPotPT>&, deepmd::DeepPotPT*, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x8a (0x7f55bf64cb12 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #36: std::_Sp_counted_ptr_inplace<deepmd::DeepPotPT, std::allocator<deepmd::DeepPotPT>, (__gnu_cxx::_Lock_policy)2>::_Sp_counted_ptr_inplace<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(std::allocator<deepmd::DeepPotPT>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x12a (0x7f55bf64bc52 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #37: std::__shared_count<(__gnu_cxx::_Lock_policy)2>::__shared_count<deepmd::DeepPotPT, std::allocator<deepmd::DeepPotPT>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(deepmd::DeepPotPT*&, std::_Sp_alloc_shared_tag<std::allocator<deepmd::DeepPotPT> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x155 (0x7f55bf649e39 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #38: std::__shared_ptr<deepmd::DeepPotPT, (__gnu_cxx::_Lock_policy)2>::__shared_ptr<std::allocator<deepmd::DeepPotPT>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(std::_Sp_alloc_shared_tag<std::allocator<deepmd::DeepPotPT> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xa2 (0x7f55bf647eac in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #39: std::shared_ptr<deepmd::DeepPotPT>::shared_ptr<std::allocator<deepmd::DeepPotPT>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(std::_Sp_alloc_shared_tag<std::allocator<deepmd::DeepPotPT> >, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x8f (0x7f55bf645eab in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #40: std::shared_ptr<deepmd::DeepPotPT> std::allocate_shared<deepmd::DeepPotPT, std::allocator<deepmd::DeepPotPT>, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(std::allocator<deepmd::DeepPotPT> const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x8a (0x7f55bf643abb in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #41: std::shared_ptr<deepmd::DeepPotPT> std::make_shared<deepmd::DeepPotPT, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&>(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xaf (0x7f55bf641402 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #42: deepmd::DeepPot::init(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x384 (0x7f55bf636a7e in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #43: deepmd::DeepPot::DeepPot(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&, int const&, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x5e (0x7f55bf63667e in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_cc.so)
# frame #44: DP_NewDeepPotWithParam2 + 0x12d (0x7f55c229bf4b in /__w/deepmd-kit/deepmd-kit/dp_test/lib/libdeepmd_c.so)
# frame #45: deepmd::hpp::DeepPot::init(std::string const&, int const&, std::string const&) + 0xeb (0x7f55c1f8730b in /__w/deepmd-kit/deepmd-kit/dp_test/lib/deepmd_lmp/dpplugin.so)
# frame #46: LAMMPS_NS::PairDeepMD::settings(int, char**) + 0x6b8 (0x7f55c1f7e170 in /__w/deepmd-kit/deepmd-kit/dp_test/lib/deepmd_lmp/dpplugin.so)
# frame #47: LAMMPS_NS::Input::execute_command() + 0x741 (0x7f55c2bb79f1 in /__w/_tool/Python/3.11.8/x64/lib/python3.11/site-packages/lammps/liblammps.so)
# frame #48: LAMMPS_NS::Input::one(std::string const&) + 0x89 (0x7f55c2bb8919 in /__w/_tool/Python/3.11.8/x64/lib/python3.11/site-packages/lammps/liblammps.so)
# frame #49: lammps_command + 0x91 (0x7f55c2c09631 in /__w/_tool/Python/3.11.8/x64/lib/python3.11/site-packages/lammps/liblammps.so)
# frame #50: <unknown function> + 0x7e2e (0x7f55ca69be2e in /lib/x86_64-linux-gnu/libffi.so.8)
# frame #51: <unknown function> + 0x4493 (0x7f55ca698493 in /lib/x86_64-linux-gnu/libffi.so.8)
# frame #52: <unknown function> + 0xe6d0 (0x7f55ca0ec6d0 in /__w/_tool/Python/3.11.8/x64/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so)
# frame #53: <unknown function> + 0x14249 (0x7f55ca0f2249 in /__w/_tool/Python/3.11.8/x64/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so)
# <omitting python frames>
@pytest.mark.skipif(
    os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") == ["0"],
    reason="An error will be thrown when there is only one GPU. See the comment above in the source code.",
)
def test_pair_deepmd_mpi(balance_args: list):
    if balance_args == []:
        # TODO: [BUG] pt: fix torch.cat error in the C++ interface when nloc==0
        # when a processor has no atoms, it throws the following errors:
        # terminate called after throwing an instance of 'c10::Error'
        #   what():  torch.cat(): expected a non-empty list of Tensors
        # Exception raised from meta at /home/conda/feedstock_root/build_artifacts/libtorch_1706629241544/work/aten/src/ATen/native/TensorShape.cpp:256 (most recent call first):
        # frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0xb2 (0x1456de6755d2 in /home/jz748/anaconda3/envs/dp3/bin/../lib/././libc10.so)
        # frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, char const*) + 0xfa (0x1456de62ad7c in /home/jz748/anaconda3/envs/dp3/bin/../lib/././libc10.so)
        # frame #2: at::meta::structured_cat::meta(c10::IListRef<at::Tensor> const&, long) + 0x9dc (0x1456485f6fdc in /home/jz748/anaconda3/envs/dp3/bin/../lib/././libtorch_cpu.so)
        # frame #3: <unknown function> + 0x2337b7d (0x145649337b7d in /home/jz748/anaconda3/envs/dp3/bin/../lib/././libtorch_cpu.so)
        # frame #4: <unknown function> + 0x2337c23 (0x145649337c23 in /home/jz748/anaconda3/envs/dp3/bin/../lib/././libtorch_cpu.so)
        # frame #5: at::_ops::cat::call(c10::IListRef<at::Tensor> const&, long) + 0x1af (0x145648a1e97f in /home/jz748/anaconda3/envs/dp3/bin/../lib/././libtorch_cpu.so)
        # frame #6: createNlistTensor(std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > const&) + 0x405 (0x1456de7a0d65 in /home/jz748/anaconda3/envs/dp3/bin/../lib/./libdeepmd_cc.so)
        # frame #7: void deepmd::DeepPotPT::compute<double, std::vector<double, std::allocator<double> > >(std::vector<double, std::allocator<double> >&, std::vector<double, std::allocator<double> >&, std::vector<double, std::allocator<double> >&, std::vector<double, std::allocator<double> >&, std::vector<double, std::allocator<double> >&, std::vector<double, std::allocator<double> > const&, std::vector<int, std::allocator<int> > const&, std::vector<double, std::allocator<double> > const&, int, deepmd::InputNlist const&, int const&, std::vector<double, std::allocator<double> > const&, std::vector<double, std::allocator<double> > const&) + 0x52c (0x1456de7a563c in /home/jz748/anaconda3/envs/dp3/bin/../lib/./libdeepmd_cc.so)
        # frame #8: void deepmd::DeepPotModelDevi::compute<double>(std::vector<double, std::allocator<double> >&, std::vector<std::vector<double, std::allocator<double> >, std::allocator<std::vector<double, std::allocator<double> > > >&, std::vector<std::vector<double, std::allocator<double> >, std::allocator<std::vector<double, std::allocator<double> > > >&, std::vector<std::vector<double, std::allocator<double> >, std::allocator<std::vector<double, std::allocator<double> > > >&, std::vector<std::vector<double, std::allocator<double> >, std::allocator<std::vector<double, std::allocator<double> > > >&, std::vector<double, std::allocator<double> > const&, std::vector<int, std::allocator<int> > const&, std::vector<double, std::allocator<double> > const&, int, deepmd::InputNlist const&, int const&, std::vector<double, std::allocator<double> > const&, std::vector<double, std::allocator<double> > const&) + 0x367 (0x1456de799057 in /home/jz748/anaconda3/envs/dp3/bin/../lib/./libdeepmd_cc.so)
        # frame #9: void DP_DeepPotModelDeviComputeNList_variant<double>(DP_DeepPotModelDevi*, int, int, double const*, int const*, double const*, int, DP_Nlist const*, int, double const*, double const*, double*, double*, double*, double*, double*) + 0x321 (0x1456f74126e1 in /home/jz748/anaconda3/envs/dp3/bin/../lib/libdeepmd_c.so)
        # frame #10: LAMMPS_NS::PairDeepMD::compute(int, int) + 0xf2f (0x1456e6c7d21f in /home/jz748/anaconda3/envs/dp3/lib/deepmd_lmp/dpplugin.so)
        # frame #11: LAMMPS_NS::Verlet::setup(int) + 0x3a2 (0x1456885c2552 in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../liblammps.so)
        # frame #12: LAMMPS_NS::Run::command(int, char**) + 0xa1c (0x14568855969c in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../liblammps.so)
        # frame #13: LAMMPS_NS::Input::execute_command() + 0x76a (0x1456883bb5ba in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../liblammps.so)
        # frame #14: LAMMPS_NS::Input::one(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0x97 (0x1456883bc5c7 in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../liblammps.so)
        # frame #15: lammps_command + 0x90 (0x145688408eb0 in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../liblammps.so)
        # frame #16: <unknown function> + 0x6a4a (0x14571dfffa4a in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../libffi.so.8)
        # frame #17: <unknown function> + 0x5fea (0x14571dffefea in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/../../libffi.so.8)
        # frame #18: <unknown function> + 0x12545 (0x14570d2bf545 in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so)
        # frame #19: <unknown function> + 0x8802 (0x14570d2b5802 in /home/jz748/anaconda3/envs/dp3/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so)
        # frame #20: _PyObject_MakeTpCall + 0x253 (0x556477a31323 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #21: _PyEval_EvalFrameDefault + 0x716 (0x556477a3ee36 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #22: _PyFunction_Vectorcall + 0x181 (0x556477a624c1 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #23: _PyEval_EvalFrameDefault + 0x49f9 (0x556477a43119 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #24: <unknown function> + 0x2a442d (0x556477af542d in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #25: PyEval_EvalCode + 0x9f (0x556477af4abf in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #26: <unknown function> + 0x2c2a1a (0x556477b13a1a in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #27: <unknown function> + 0x2be593 (0x556477b0f593 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #28: <unknown function> + 0x2d3930 (0x556477b24930 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #29: _PyRun_SimpleFileObject + 0x1ae (0x556477b242ce in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #30: _PyRun_AnyFileObject + 0x44 (0x556477b23ff4 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #31: Py_RunMain + 0x374 (0x556477b1e6f4 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #32: Py_BytesMain + 0x37 (0x556477ae4a77 in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        # frame #33: <unknown function> + 0x27b8a (0x14571e136b8a in /lib64/libc.so.6)
        # frame #34: __libc_start_main + 0x8b (0x14571e136c4b in /lib64/libc.so.6)
        # frame #35: <unknown function> + 0x29391d (0x556477ae491d in /home/jz748/anaconda3/envs/dp3/bin/python3.11)
        pytest.skip(
            "An error will be thrown in this test. See the comment above in the source code."
        )
    with tempfile.NamedTemporaryFile() as f:
        sp.check_call(
            [
                "mpirun",
                "-n",
                "2",
                sys.executable,
                Path(__file__).parent / "run_mpi_pair_deepmd.py",
                data_file,
                pb_file,
                pb_file2,
                md_file,
                f.name,
                *balance_args,
            ]
        )
        arr = np.loadtxt(f.name, ndmin=1)
    pe = arr[0]

    relative = 1.0
    assert pe == pytest.approx(expected_e)
    # load model devi
    md = np.loadtxt(md_file.resolve())
    norm = np.linalg.norm(np.mean([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f /= norm + relative
    assert md[7:] == pytest.approx(expected_md_f)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 6
    )
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))
