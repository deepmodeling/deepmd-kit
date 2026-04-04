# SPDX-License-Identifier: LGPL-3.0-or-later
import os
from pathlib import (
    Path,
)

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

# Reference values from the seed=1 .pt2 model (NoPBC, Model 0)
expected_e = 3.5101080091096860e-01
expected_f = np.array(
    [
        [3.9007324220254663e-03, -1.6340906092268837e-03, 2.4784543132550553e-03],
        [-3.9007324220254660e-03, 1.6340906092268837e-03, -2.4784543132550550e-03],
        [1.0879565176984952e-04, 1.0163804310078055e-04, -1.4887826031663627e-04],
        [-1.0879565176984952e-04, -1.0163804310078055e-04, 1.4887826031663627e-04],
    ]
)
expected_fm = np.array(
    [
        [3.4589594972289518e-03, -1.4490235731634794e-03, 2.3561281037953720e-03],
        [-3.0400436796538990e-04, 1.2735318117469008e-04, -1.4949786028183132e-03],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
    ]
)

# Reference values from the seed=2 .pt2 model (NoPBC, Model 1)
expected_f2 = np.array(
    [
        [-3.0870239868329980e-02, 1.2932127512408504e-02, 2.7561357633479750e-02],
        [3.0870239868329978e-02, -1.2932127512408506e-02, -2.7561357633479742e-02],
        [4.5712656471395960e-04, 4.2705244861435744e-04, -6.2554161487173430e-04],
        [-4.5712656471395960e-04, -4.2705244861435744e-04, 6.2554161487173430e-04],
    ]
)

expected_fm2 = np.array(
    [
        [-7.2838456252868870e-03, 3.0513407349174793e-03, -1.9672009896273334e-02],
        [9.7240358761389140e-03, -4.0735825967608950e-03, -1.7012161861151297e-02],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
    ]
)

box = np.array([0, 100, 0, 100, 0, 100, 0, 0, 0])
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
    lammps.boundary("f f f")
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
    # rel=1e-4: md.out is written with default scientific format (~6 significant digits)
    assert md[4] == pytest.approx(np.max(expected_md_f), rel=1e-4)
    assert md[5] == pytest.approx(np.min(expected_md_f), rel=1e-4)
    assert md[6] == pytest.approx(np.mean(expected_md_f), rel=1e-4)
    assert md[7] == pytest.approx(np.max(expected_md_fm), rel=1e-4)
    assert md[8] == pytest.approx(np.min(expected_md_fm), rel=1e-4)
    assert md[9] == pytest.approx(np.mean(expected_md_fm), rel=1e-4)


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
    # rel=1e-4: md.out is written with default scientific format (~6 significant digits)
    assert md[4] == pytest.approx(np.max(expected_md_f), rel=1e-4)
    assert md[5] == pytest.approx(np.min(expected_md_f), rel=1e-4)
    assert md[6] == pytest.approx(np.mean(expected_md_f), rel=1e-4)
    assert md[7] == pytest.approx(np.max(expected_md_fm), rel=1e-4)
    assert md[8] == pytest.approx(np.min(expected_md_fm), rel=1e-4)
    assert md[9] == pytest.approx(np.mean(expected_md_fm), rel=1e-4)
