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
pb_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa.savedmodel"
)
pb_file2 = Path(__file__).parent / "graph2.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
data_file_si = Path(__file__).parent / "data.si"
data_type_map_file = Path(__file__).parent / "data_type_map.lmp"
md_file = Path(__file__).parent / "md.out"

# this is as the same as python and c++ tests, test_deeppot_a.py
expected_ae = np.array(
    [
        -94.24098099691867,
        -187.8049502787117,
        -187.80486052083617,
        -94.24059525229518,
        -187.80366985846246,
        -187.8042377490619,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        -0.0020150115442053,
        -0.0133389255924977,
        -0.0014347177433057,
        -0.0140757358179293,
        0.0031373814221557,
        0.0098594354314677,
        0.004755683505073,
        0.0099471082374397,
        -0.0080868184532793,
        -0.0086166721574536,
        0.0037803939137322,
        -0.0075733131286482,
        0.0037437603038209,
        -0.008452527996008,
        0.0134837461840424,
        0.0162079757106944,
        0.0049265700151781,
        -0.0062483322902769,
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
        0.0133534319524089,
        0.0013445914938337,
        -0.0029370551651952,
        0.0002611806151294,
        0.004662662211533,
        -0.0002717443796319,
        -0.0027779798869954,
        -0.0003277976466339,
        0.0018284972283065,
        0.0085710118978246,
        0.0003865036653608,
        -0.0057964032875089,
        -0.0014358330222619,
        0.0002912625128908,
        0.001212630641674,
        -0.0050582608957046,
        -0.0001087907763249,
        0.0040068757134429,
        0.0116736349373084,
        0.0007055477968445,
        -0.0019544933708784,
        0.0032997459258512,
        0.0037887116116712,
        -0.0043140890650835,
        -0.0034418738401156,
        -0.0029420616852742,
        0.0038219676716965,
        0.0147134944025738,
        0.0005214313829998,
        -0.0006524136175906,
        0.0003656980996363,
        0.0010046161607714,
        -0.0017279359476254,
        0.000111127036911,
        -0.0017063190420654,
        0.0030174567965904,
        0.0104435705455108,
        -0.0008704394438241,
        0.0012354202650812,
        0.0009397615830053,
        0.0029105236407293,
        -0.0044188897903449,
        -0.0011461513500477,
        -0.0045759080125852,
        0.0070310883421107,
        0.0089818851995049,
        0.0038819466696704,
        -0.005443705549253,
        0.0025390283635246,
        0.0012121502955869,
        -0.0016998728971157,
        -0.0032355117893925,
        -0.0015590242752438,
        0.0021980725909838,
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
    # Requires for DPA-2
    lammps.atom_modify("map yes")
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
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
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
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
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
        lammps_real.variable(f"virial{jj} atom c_virial[{ii + 1}]")
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
        lammps_real.variable(f"virial{jj} atom c_virial[{ii + 1}]")
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
@pytest.mark.skip("MPI is not supported")
def test_pair_deepmd_mpi(balance_args: list):
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
