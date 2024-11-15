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

import numpy as np
import pytest
from lammps import (
    PyLammps,
)
from write_lmp_data import (
    write_lmp_data_spin,
)

pbtxt_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deepspin_nlist.pbtxt"
)
pbtxt_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deepspin_nlist-2.pbtxt"
)
pb_file = Path(__file__).parent / "deepspin_nlist.pb"
pb_file2 = Path(__file__).parent / "deepspin_nlist-2.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
data_file_si = Path(__file__).parent / "data.si"
data_type_map_file = Path(__file__).parent / "data_type_map.lmp"
md_file = Path(__file__).parent / "md.out"

expected_ae = np.array(
    [-7.314365618560289, -7.313531316181837, -2.8980532245013997, -2.897373810282277]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        [0.0275132293555514, -0.0112057401883111, -0.0212278132621243],
        [-0.0229926640905535, 0.0114378553363334, 0.019670014885563],
        [0.0086502856137601, 0.0088926283192558, -0.0127014507822769],
        [-0.013170850878758, -0.009124743467278, 0.0142592491588383],
    ]
)
expected_fm = np.array(
    [
        [0.0066245455049449, -0.0023055088004378, 0.0294608578045521],
        [-0.0041979452385972, 0.0025775020220167, 0.0316295420619988],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
    ]
)

expected_f2 = np.array(
    [
        [-0.0009939342103254, 0.0009450997605637, -0.0002710189976979],
        [0.0040364645780618, -0.0008326705633617, -0.000208982833015],
        [0.0007716358981262, 0.0018705501216939, -0.002687696295354],
        [-0.0038141662658625, -0.0019829793188958, 0.0031676981260669],
    ]
)

expected_fm2 = np.array(
    [
        [0.0021649674715341, -0.0008507073771461, 0.0270620372234819],
        [-0.0026523551738949, 0.0013308033074224, 0.0294569107929189],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
    ]
)

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


sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file.resolve()} -o {pb_file.resolve()}".split()
)
sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file2.resolve()} -o {pb_file2.resolve()}".split()
)


def setup_module() -> None:
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
    # for ii in range(9):
    #     jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
    #     lammps.variable(f"pressure{jj} equal c_pressure[{ii+1}]")
    # for ii in range(9):
    #     jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
    #     lammps.variable(f"virial{jj} atom c_virial[{ii+1}]")
    # lammps.dump(
    #     "1 all custom 1 dump id " + " ".join([f"v_virial{ii}" for ii in range(9)])
    # )
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(4):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    idx_map = lammps.lmp.numpy.extract_atom("id") - 1
    assert np.array(lammps.variables["eatom"].value) == pytest.approx(
        expected_ae[idx_map]
    )
    # vol = box[1] * box[3] * box[5]
    # for ii in range(6):
    #     jj = [0, 4, 8, 3, 6, 7, 1, 2, 5][ii]
    #     assert np.array(
    #         lammps.variables[f"pressure{jj}"].value
    #     ) / constants.nktv2p == pytest.approx(
    #         -expected_v[idx_map, jj].sum(axis=0) / vol
    #     )
    # for ii in range(9):
    #     assert np.array(
    #         lammps.variables[f"virial{ii}"].value
    #     ) / constants.nktv2p == pytest.approx(expected_v[idx_map, ii])


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
def test_pair_deepmd_mpi(balance_args: list) -> None:
    with tempfile.NamedTemporaryFile() as f:
        sp.check_call(
            [
                "mpirun",
                "-n",
                "2",
                sys.executable,
                Path(__file__).parent / "run_mpi_pair_deepmd_spin.py",
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
