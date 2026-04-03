# SPDX-License-Identifier: LGPL-3.0-or-later
import os
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
    write_lmp_data_spin,
)

pb_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa_spin.pt2"
)
pb_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa_spin_md1.pt2"
)
data_file = Path(__file__).parent / "data.lmp"
md_file = Path(__file__).parent / "md.out"

# Reference values from the seed=1 .pt2 model (same weights as .pth)
expected_ae = np.array(
    [-2.33730603846356, -2.339828637443377, -2.3584765990764933, -2.358478126000974]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        [0.036819000183374, -0.0154603124989284, -0.0277136918031471],
        [-0.0369115932121166, 0.0154614940830129, 0.0277067438704936],
        [-0.0010240778189108, -0.0010425850123752, 0.0015323196618039],
        [0.0011166708476534, 0.0010414034282908, -0.0015253717291505],
    ]
)
expected_fm = np.array(
    [
        [0.007540380021158, -0.0031615447712641, 0.0204706018052022],
        [-0.0074177167392878, 0.0031072528813168, 0.0209277147341756],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
    ]
)

# Reference values from the seed=2 .pt2 model (deeppot_dpa_spin_md1.pt2, PBC)
expected_f2 = np.array(
    [
        [-5.0903657013833805e-03, 2.2714596279667276e-03, 3.5699195164111686e-03],
        [4.9335267498655496e-03, -2.2027414621596151e-03, -3.6334615119367475e-03],
        [1.1774037584944407e-03, 6.9469696563881659e-04, -1.0467746979479903e-03],
        [-1.0205648069766093e-03, -7.6341513144592970e-04, 1.1103166934735694e-03],
    ]
)

expected_fm2 = np.array(
    [
        [1.6161523921420901e-03, -6.1507586761937660e-04, -1.4084754563287318e-02],
        [-1.7131909329371128e-03, 7.1495794003373391e-04, -1.3238736630706918e-02],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
    ]
)

expected_v = -np.array(
    [
        0.0138536891649799,
        -0.0057815832940349,
        -0.0104366273910430,
        -0.0057802135977019,
        0.0024216972469495,
        0.0043747666241247,
        -0.0120159787305366,
        0.0050342035124280,
        0.0090942101965059,
        0.0135151396517160,
        -0.0056617476919350,
        -0.0102276732499471,
        -0.0056606594176084,
        0.0023713573235927,
        0.0042837422619739,
        -0.0084858208754591,
        0.0035548709072868,
        0.0064217022841311,
        0.0007099617850315,
        0.0003917168967788,
        -0.0005467867622337,
        0.0003906286224523,
        0.0003696501943719,
        -0.0005419287758774,
        -0.0005551067425154,
        -0.0005416915274450,
        0.0007957607021995,
        0.0004252005652282,
        0.0003972268438316,
        -0.0005818534050492,
        0.0003958571474987,
        0.0003698139141107,
        -0.0005416992544720,
        -0.0005797982376440,
        -0.0005416536167464,
        0.0007934081146707,
    ]
).reshape(4, 9)

box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
spin = np.array(
    [
        [0, 0, 1.2737],
        [0, 0, 1.2737],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
type_NiO = np.array([1, 1, 2, 2])


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data_spin(box, coord, spin, type_NiO, data_file)


def teardown_module() -> None:
    os.remove(data_file)


def _lammps(data_file, units="metal") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("p p p")
    lammps.atom_style("spin")
    if units == "metal":
        lammps.neighbor("2.0 bin")
    else:
        raise ValueError("units for spin should be metal")
    lammps.neigh_modify("every 10 delay 0 check no")
    lammps.read_data(data_file.resolve())
    if units == "metal":
        lammps.mass("1 58")
        lammps.mass("2 16")
    else:
        raise ValueError("units for spin should be metal")
    if units == "metal":
        lammps.timestep(0.0005)
    else:
        raise ValueError("units for spin should be metal")
    lammps.fix("1 all nve")
    return lammps


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


def test_pair_deepmd(lammps) -> None:
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(4):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_virial(lammps) -> None:
    lammps.pair_style(f"deepspin {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.compute("peatom all pe/atom pair")
    lammps.compute("pressure all pressure NULL pair")
    lammps.compute("virial all centroid/stress/atom NULL pair")
    lammps.variable("eatom atom c_peatom")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"pressure{jj} equal c_pressure[{ii + 1}]")
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        lammps.variable(f"virial{jj} atom c_virial[{ii + 1}]")
    lammps.dump(
        "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(4):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    idx_map = lammps.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    assert np.array(lammps.variables["eatom"].value) == pytest.approx(
        expected_ae[idx_map]
    )
    vol = box[1] * box[3] * box[5]
    for ii in range(6):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        assert np.array(
            lammps.variables[f"pressure{jj}"].value
        ) / constants.nktv2p == pytest.approx(
            -expected_v[idx_map, jj].sum(axis=0) / vol
        )
    for ii in range(9):
        jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
        assert np.array(
            lammps.variables[f"virial{jj}"].value
        ) / constants.nktv2p == pytest.approx(expected_v[idx_map, jj])


def test_pair_deepmd_model_devi(lammps) -> None:
    lammps.pair_style(
        f"deepspin {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(4):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_fm = np.linalg.norm(np.std([expected_fm, expected_fm2], axis=0), axis=1)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    assert md[7] == pytest.approx(np.max(expected_md_fm))
    assert md[8] == pytest.approx(np.min(expected_md_fm))
    assert md[9] == pytest.approx(np.mean(expected_md_fm))


def test_pair_deepmd_model_devi_atomic_relative(lammps) -> None:
    relative = 1.0
    lammps.pair_style(
        f"deepspin {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic relative {relative}"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(4):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    # load model devi
    md = np.loadtxt(md_file.resolve())
    norm = np.linalg.norm(np.mean([expected_f, expected_f2], axis=0), axis=1)
    norm_spin = np.linalg.norm(np.mean([expected_fm, expected_fm2], axis=0), axis=1)
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_f /= norm + relative
    expected_md_fm = np.linalg.norm(np.std([expected_fm, expected_fm2], axis=0), axis=1)
    expected_md_fm /= norm_spin + relative
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    assert md[7] == pytest.approx(np.max(expected_md_fm))
    assert md[8] == pytest.approx(np.min(expected_md_fm))
    assert md[9] == pytest.approx(np.mean(expected_md_fm))
