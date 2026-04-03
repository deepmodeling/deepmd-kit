# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with .pt2 (AOTInductor) DPA3 model, non-periodic boundary.

Mirrors test_lammps_dpa3_pt2.py but with boundary "f f f" (NoPbc).
Reference values from source/tests/infer/gen_dpa3.py / C++ test (NoPbc).
"""

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
    write_lmp_data,
)

pb_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa3.pt2"
data_file = Path(__file__).parent / "data_dpa3_pt2_nopbc.lmp"
data_file_si = Path(__file__).parent / "data_dpa3_pt2_nopbc.si"
data_type_map_file = Path(__file__).parent / "data_type_map_dpa3_pt2_nopbc.lmp"

# Reference values from gen_dpa3.py / test_deeppot_dpa3_ptexpt.cc (NoPbc)
expected_ae = np.array(
    [
        2.748896667984845887e-01,
        2.803947322373078754e-01,
        2.865499847997139971e-01,
        2.695555136277474895e-01,
        2.739584531066059925e-01,
        2.752217127378932537e-01,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        -4.469562373941994571e-02,
        1.872384237732456838e-02,
        3.382371526226372882e-02,
        4.469562373941994571e-02,
        -1.872384237732456838e-02,
        -3.382371526226372882e-02,
        -8.962417443747255821e-04,
        6.973117535150641388e-05,
        3.708588577163370883e-05,
        6.643516471939500678e-02,
        -2.418189932122343649e-02,
        4.484243027251725439e-02,
        9.031619071676464522e-03,
        5.637239343551967569e-02,
        -8.796029317613156262e-02,
        -7.457054204669674724e-02,
        -3.226022528964775371e-02,
        4.308077701784267244e-02,
    ]
).reshape(6, 3)

expected_v = -np.array(
    [
        -1.634330450074628072e-02,
        6.846519453015231793e-03,
        1.236790610867266604e-02,
        6.846519453015259549e-03,
        -2.868136527614494058e-03,
        -5.181149856335852399e-03,
        1.236790610867266604e-02,
        -5.181149856335859338e-03,
        -9.359496514671244993e-03,
        -1.673145706642453767e-02,
        7.009123906204950405e-03,
        1.266164318540249911e-02,
        7.009123906204922649e-03,
        -2.936254609356120371e-03,
        -5.304201874965906727e-03,
        1.266164318540247136e-02,
        -5.304201874965899788e-03,
        -9.581784032196449807e-03,
        2.483905957089865488e-03,
        -1.710616363479115602e-04,
        -5.347582359011894028e-05,
        -1.996686279554130779e-04,
        1.446275632786597548e-05,
        2.638112328458543858e-06,
        -1.197563523836930226e-04,
        1.205600575305949503e-05,
        -4.593499883389132697e-06,
        -4.089897480719173473e-02,
        -3.495830205935246404e-03,
        1.154978330068986980e-03,
        -3.627142383941225900e-03,
        -1.488475129792680290e-02,
        2.311785022979555293e-02,
        1.347848716528848856e-03,
        2.315545736893441509e-02,
        -3.642400982788428221e-02,
        -1.119743233540158867e-03,
        -7.327254171127076110e-03,
        1.144439607350029517e-02,
        -1.403015516843159061e-03,
        -1.349644754565121341e-02,
        2.117430870829728473e-02,
        2.103115217604090148e-03,
        2.047643373328661420e-02,
        -3.212706064943796069e-02,
        -2.418232649309504101e-02,
        -1.012366394018440752e-02,
        1.334822742508814941e-02,
        -1.588798342485496506e-02,
        -6.330672283764562924e-03,
        8.295385033255518736e-03,
        2.256291842331806241e-02,
        8.946234975702738179e-03,
        -1.170798305154926999e-02,
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


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data(box, coord, type_OH, data_file)
    write_lmp_data(box, coord, type_HO, data_type_map_file)
    write_lmp_data(
        box * constants.dist_metal2si,
        coord * constants.dist_metal2si,
        type_OH,
        data_file_si,
    )


def teardown_module() -> None:
    for f in [data_file, data_type_map_file, data_file_si]:
        if f.exists():
            os.remove(f)


def _lammps(data_file, units="metal") -> PyLammps:
    lammps = PyLammps()
    lammps.units(units)
    lammps.boundary("f f f")
    lammps.atom_style("atomic")
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


def test_pair_deepmd(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()}")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_virial(lammps) -> None:
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
    idx_map = lammps.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    for ii in range(9):
        assert np.array(
            lammps.variables[f"virial{ii}"].value
        ) / constants.nktv2p == pytest.approx(expected_v[idx_map, ii])


def test_pair_deepmd_type_map(lammps_type_map) -> None:
    lammps_type_map.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_type_map.pair_coeff("* * H O")
    lammps_type_map.run(0)
    assert lammps_type_map.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps_type_map.atoms[ii].force == pytest.approx(
            expected_f[lammps_type_map.atoms[ii].id - 1]
        )
    lammps_type_map.run(1)


def test_pair_deepmd_real(lammps_real) -> None:
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


def test_pair_deepmd_virial_real(lammps_real) -> None:
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
    idx_map = lammps_real.lmp.numpy.extract_atom("id")[: coord.shape[0]] - 1
    for ii in range(9):
        assert np.array(
            lammps_real.variables[f"virial{ii}"].value
        ) / constants.nktv2p_real == pytest.approx(
            expected_v[idx_map, ii] * constants.ener_metal2real
        )


def test_pair_deepmd_si(lammps_si) -> None:
    lammps_si.pair_style(f"deepmd {pb_file.resolve()}")
    lammps_si.pair_coeff("* *")
    lammps_si.run(0)
    assert lammps_si.eval("pe") == pytest.approx(expected_e * constants.ener_metal2si)
    for ii in range(6):
        assert lammps_si.atoms[ii].force == pytest.approx(
            expected_f[lammps_si.atoms[ii].id - 1] * constants.force_metal2si
        )
    lammps_si.run(1)
