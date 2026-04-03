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

pb_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa_spin.pt2"
)
pb_file2 = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa_spin_md1.pt2"
)
data_file = Path(__file__).parent / "data.lmp"
md_file = Path(__file__).parent / "md.out"

# Reference values from the seed=1 .pt2 model (same weights as .pth, NoPBC)
expected_ae = np.array(
    [-2.337313880002, -2.339828637443377, -2.358478126000974, -2.358478126000974]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        [0.036908169450058, -0.0154615304452946, -0.0277072310206975],
        [-0.036908169450058, 0.0154615304452946, 0.0277072310206975],
        [-0.001114392839443, -0.0010410775210586, 0.0015249586223957],
        [0.001114392839443, 0.0010410775210586, -0.0015249586223957],
    ]
)
expected_fm = np.array(
    [
        [0.0075469514215227, -0.0031615607306379, 0.0204654183352036],
        [-0.0074172317569893, 0.003107218709009, 0.020927678503653],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
    ]
)

# Reference values from the seed=2 .pt2 model (deeppot_dpa_spin_md1.pt2, NoPBC)
expected_f2 = np.array(
    [
        [-5.0313948956892344e-03, 2.1077465103562995e-03, 3.3650618879773678e-03],
        [5.0313948956892335e-03, -2.1077465103562999e-03, -3.3650618879773678e-03],
        [7.8146213901410743e-04, 7.3005015618423231e-04, -1.0693692428614104e-03],
        [-7.8146213901410743e-04, -7.3005015618423231e-04, 1.0693692428614104e-03],
    ]
)

expected_fm2 = np.array(
    [
        [1.4597145672945039e-03, -6.1150204846121062e-04, -1.3469050004718076e-02],
        [-1.7167887170022188e-03, 7.1919527333876780e-04, -1.3100088729734645e-02],
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
