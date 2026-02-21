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
    write_lmp_data_spin,
)

pbtxt_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deepspin_nlist-2.pbtxt"
)
pb_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa_spin.pth"
)
pb_file2 = Path(__file__).parent / "deepspin_nlist-2.pb"
system_file = Path(__file__).parent.parent.parent / "tests"
data_file = Path(__file__).parent / "data.lmp"
data_file_si = Path(__file__).parent / "data.si"
data_type_map_file = Path(__file__).parent / "data_type_map.lmp"
md_file = Path(__file__).parent / "md.out"

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

expected_v2 = -np.array(
    [
        -0.0068361570854045,
        0.0013367399255969,
        0.0027254156851031,
        0.0013170622611582,
        -0.0006584860372994,
        -0.0011202746253630,
        0.0034372788237214,
        -0.0014078563891643,
        -0.0026234422772135,
        -0.0065741762148638,
        0.0017805968136400,
        0.0034294170220393,
        0.0017765463870965,
        -0.0007431139812367,
        -0.0013178346591454,
        0.0028404497247166,
        -0.0010584927243460,
        -0.0019736773403821,
        -0.0005990686037429,
        -0.0006790752112473,
        0.0009843156545704,
        -0.0006499258389702,
        -0.0005944102287950,
        0.0008730050519947,
        0.0009534901617246,
        0.0008816224750390,
        -0.0012893921034252,
        -0.0004768372014215,
        -0.0006613274582292,
        0.0009616568976184,
        -0.0007221357003557,
        -0.0006885045030101,
        0.0010101324669343,
        0.0010319090966831,
        0.0009955172109546,
        -0.0014553634659113,
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


sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file2.resolve()} -o {pb_file2.resolve()}".split()
)


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


@pytest.mark.skipif(
    os.environ.get("ENABLE_TENSORFLOW", "1") != "1",
    reason="Skip test because TensorFlow support is not enabled.",
)
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


@pytest.mark.skipif(
    os.environ.get("ENABLE_TENSORFLOW", "1") != "1",
    reason="Skip test because TensorFlow support is not enabled.",
)
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
@pytest.mark.skipif(
    os.environ.get("ENABLE_TENSORFLOW", "1") != "1",
    reason="Skip test because TensorFlow support is not enabled.",
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
