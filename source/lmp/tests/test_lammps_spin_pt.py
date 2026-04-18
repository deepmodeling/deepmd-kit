# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import os
import subprocess as sp
import shutil
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

from model_convert import (
    ensure_converted_pb,
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

expected_e = 3.5053686886040974e-01
expected_ae = np.array(
    [
        6.8203336981159180e-02,
        6.4899945717305470e-02,
        1.0867432727951952e-01,
        1.0875925888242556e-01,
    ]
)
expected_f = np.array(
    [
        [3.9965859059739960e-03, -1.5714685255928385e-03, 2.5489246986054267e-03],
        [-3.2538293505478910e-03, 1.5372892024705638e-03, -2.6183962915675120e-03],
        [-5.8572951970559290e-04, 1.2789559354576532e-04, -8.4978782279560860e-05],
        [-1.5702703572051152e-04, -9.3716270423490900e-05, 1.5445037524164582e-04],
    ]
)
expected_fm = np.array(
    [
        [3.5882891345325510e-03, -1.4332293643181068e-03, 2.1309512069449844e-03],
        [-2.2686566316524877e-04, 1.1563585411337640e-04, -1.5585998391414227e-03],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
        [0.0000000000000000e00, 0.0000000000000000e00, 0.0000000000000000e00],
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
        -6.5442162870715910e-04,
        3.3133477241365930e-04,
        3.9883281200020510e-04,
        3.5002780951132103e-04,
        -1.1589573685630256e-04,
        -2.2265549504892014e-04,
        3.4055402098896265e-04,
        -1.9139376762381335e-04,
        -3.3133994148957030e-04,
        3.1589094549377213e-03,
        -1.3741877857714818e-03,
        -2.5227682918535005e-03,
        -1.4131573474734480e-03,
        6.0225037103514220e-04,
        1.0902107277722482e-03,
        1.4702420402848647e-03,
        -6.1977156785934680e-04,
        -1.1144364605899786e-03,
        -2.1325917337790670e-03,
        7.0758382747778870e-05,
        -1.2848658004242357e-05,
        1.0290158078156647e-04,
        -2.7842261103620382e-05,
        3.2953845020209020e-05,
        2.2234591227388966e-04,
        2.2930239123112260e-05,
        -3.1997128170916170e-05,
        -8.1674704416169310e-04,
        -1.1094383565005458e-04,
        1.6599006542109280e-04,
        -1.2281050907954750e-04,
        -5.0200238466137690e-05,
        7.2193112591762430e-05,
        2.7750902949636390e-04,
        8.2718432747583720e-05,
        -1.1997871423457964e-04,
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


ensure_converted_pb(pbtxt_file2, pb_file2)


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
