# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS fparam and aparam input."""

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
    write_lmp_data,
)

pbtxt_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "fparam_aparam.pbtxt"
)
pb_file = Path(__file__).parent / "fparam_aparam.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
md_file = Path(__file__).parent / "md.out"

# from api_cc/tests/test_deeppot_a_fparam_aparam.cc
expected_ae = np.array(
    [
        -1.038271183039953804e-01,
        -7.285433575272914908e-02,
        -9.467600174099155552e-02,
        -1.467050086239614082e-01,
        -7.660561620618722145e-02,
        -7.277295998502930630e-02,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        6.622266817497907132e-02,
        5.278739055693523058e-02,
        2.265727495541422845e-02,
        -2.606047850915838363e-02,
        -4.538811686410718776e-02,
        1.058247569147072187e-02,
        1.679392490937766935e-01,
        -2.257828022687320690e-03,
        -4.490145670355452645e-02,
        -1.148364103573685929e-01,
        -1.169790466695089237e-02,
        6.140402504113953025e-02,
        -8.078778132132799494e-02,
        -5.838878056243369807e-02,
        6.773639989682191109e-02,
        -1.247724708090079161e-02,
        6.494523955924384750e-02,
        -1.174787188812918687e-01,
    ]
).reshape(6, 3)

expected_v = -np.array(
    [
        -1.589185553287162656e-01,
        2.586163333170100279e-03,
        -1.575127933809472624e-04,
        -1.855360380105876630e-02,
        1.949822090859933826e-02,
        -1.006552056166355388e-02,
        3.177029853276916449e-02,
        1.714349636720383010e-03,
        -1.290389175187874483e-03,
        -8.553510339477603253e-02,
        -5.654637257232508415e-03,
        -1.286954833787038420e-02,
        2.464156457499515687e-02,
        -2.398202886026797043e-02,
        -1.957110465239037672e-02,
        2.233492928605742764e-02,
        6.107843207824020099e-03,
        1.707078295947736047e-03,
        -1.653994088976195043e-01,
        3.894358678172111371e-02,
        -2.169595969759342477e-02,
        6.819704294738503786e-03,
        -5.018242039618424008e-03,
        2.640664428663210429e-03,
        -1.985298275686078057e-03,
        -3.638421609610945767e-02,
        2.342932331075030239e-02,
        -8.501331914753691710e-02,
        -2.181253413538992297e-03,
        4.311300069651782287e-03,
        -1.910329328333908129e-03,
        -1.808810159508548836e-03,
        -1.540075281450827612e-03,
        -1.173703213175551763e-02,
        -2.596306629910121507e-03,
        6.705025662372287101e-03,
        -9.038455005073858795e-02,
        3.011717773578577451e-02,
        -5.083054073419784880e-02,
        -2.951210292616929069e-03,
        2.342445652898489383e-02,
        -4.091207474993674431e-02,
        -1.648470649301832236e-02,
        -2.872261885460645689e-02,
        4.763924972552112391e-02,
        -8.300036532764677732e-02,
        1.020429228955421243e-03,
        -1.026734151199098881e-03,
        5.678534096113684732e-02,
        1.273635718045938205e-02,
        -1.530143225195957322e-02,
        -1.061671865629566225e-01,
        -2.486859433265622629e-02,
        2.875323131744185121e-02,
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
type_OH = np.array([1, 1, 1, 1, 1, 1])


sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file.resolve()} -o {pb_file.resolve()}".split()
)


def setup_module() -> None:
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    os.remove(data_file)


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
    elif units == "si":
        lammps.mass("1 %.10e" % (16 * constants.mass_metal2si))
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


def test_pair_deepmd(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()} fparam 0.25852028 aparam 0.25852028")
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(6):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    lammps.run(1)


def test_pair_deepmd_virial(lammps) -> None:
    lammps.pair_style(f"deepmd {pb_file.resolve()} fparam 0.25852028 aparam 0.25852028")
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
