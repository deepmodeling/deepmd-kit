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

# Reference values from the seed=1 .pt2 model (PBC, Model 0)
expected_e = 4.508261273186509599e-01
expected_ae = np.array(
    [
        -4.384923692739055301e-02,
        -4.127243767708395350e-02,
        2.679072197028402047e-01,
        2.680405822202852617e-01,
    ]
)
expected_f = np.array(
    [
        [-4.2676712486866839e-03, 1.8628771592081692e-03, 8.1929947798959294e-03],
        [4.1470034985421397e-03, -1.8846167362954120e-03, -8.0779540071613375e-03],
        [-1.5774196991860190e-05, 1.5095872055037390e-04, -2.7974166116106585e-04],
        [1.3644194713640517e-04, -1.2921914346313107e-04, 1.6470088842647485e-04],
    ]
)
expected_fm = np.array(
    [
        [-4.5085730123974212e-04, 1.8188156762674026e-04, 1.9591427407519438e-03],
        [3.1582497858262133e-03, -1.3356926943075573e-03, 1.7139696951871720e-03],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
    ]
)

# Reference values from the seed=2 .pt2 model (PBC, Model 1)
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
        -1.4794937634472474e-03,
        6.6688347518566377e-04,
        9.2820857914341511e-04,
        6.9899209610065787e-04,
        -2.3202779455430261e-04,
        -4.3062165105966181e-04,
        4.2186963636220076e-03,
        -1.8596964037585618e-03,
        -3.3537115627438724e-03,
        -2.0962516214727486e-03,
        7.9319592962628796e-04,
        1.3351142815012081e-03,
        8.9145117156866685e-04,
        -3.4764851549428505e-04,
        -6.2285930526845095e-04,
        1.5422972351535255e-03,
        -6.6089529841465919e-04,
        -1.1822529939065273e-03,
        7.9431779279767246e-04,
        -1.2677430787194573e-04,
        6.4796756091853292e-05,
        -1.3924650354116248e-04,
        -3.1301607401264882e-05,
        4.9296926793974502e-05,
        -9.4965688637539702e-05,
        5.6156814646123667e-05,
        -9.1198182079314316e-05,
        2.6039820355670405e-04,
        4.3106382567677546e-05,
        -9.7016827514025993e-05,
        -7.4785284620483794e-05,
        -6.6085841514910694e-05,
        9.2317188787004601e-05,
        1.3480686702278333e-05,
        8.2958814726605517e-05,
        -1.1609755379269147e-04,
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
