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
    [-7.313160384523243, -7.312173646552338, -2.8984477845267067, -2.8984477845267067]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        [0.0277100137316238, -0.0116082489956803, -0.0211484273275705],
        [-0.0277100137316238, 0.0116082489956803, 0.0211484273275705],
        [0.0097588349924651, 0.0091168063745397, -0.0133541952528469],
        [-0.0097588349924651, -0.0091168063745397, 0.0133541952528469],
    ]
)
expected_fm = np.array(
    [
        [0.0058990325687816, -0.0024712163463815, 0.0296682261295907],
        [-0.0060028470719556, 0.0025147062058193, 0.0321884178873188],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
    ]
)

expected_f2 = np.array(
    [
        [-0.0020912362538459, 0.0008760584306652, -0.0002029714364812],
        [0.0020912362538459, -0.0008760584306652, 0.0002029714364812],
        [0.0020348523962324, 0.0019009805280592, -0.0027845348580022],
        [-0.0020348523962324, -0.0019009805280592, 0.0027845348580022],
    ]
)

expected_fm2 = np.array(
    [
        [0.0020796789544968, -0.0008712168593162, 0.0269545489546998],
        [-0.0031170434556743, 0.0013057884746744, 0.0295063550138163],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
        [0.0000000000000000, 0.00000000000000000, 0.00000000000000000],
    ]
)

expected_v = -np.array(
    [
        0.0021380771762615,
        -0.0008956809792447,
        -0.0016180043496033,
        -0.0008956809792447,
        0.0003752177075214,
        0.0006778126329419,
        -0.0014520530654550,
        0.0006082925003933,
        0.0010988509684524,
        0.0034592108484302,
        -0.0014491288689370,
        -0.0026177811825959,
        -0.0014491288689370,
        0.0006070674991493,
        0.0010966380629793,
        -0.0027640824464858,
        0.0011579264302846,
        0.0020917380676109,
        -0.0037083572971367,
        -0.0034643864223251,
        0.0050745941960818,
        -0.0034643864223251,
        -0.0032364662629616,
        0.0047407393147607,
        0.0050745941960818,
        0.0047407393147607,
        -0.0069441815314804,
        -0.0037083572971367,
        -0.0034643864223251,
        0.0050745941960818,
        -0.0034643864223251,
        -0.0032364662629616,
        0.0047407393147607,
        0.0050745941960818,
        0.0047407393147607,
        -0.0069441815314804,
        0.0103691205704445,
        -0.0043438207795105,
        -0.0078469020533093,
        -0.0043438207795105,
        0.0018197087049301,
        0.0032872157250350,
        -0.0076002352547860,
        0.0031838823364644,
        0.0057515293820002,
        0.0045390015662654,
        -0.0019014736291112,
        -0.0034349201042009,
        -0.0019014736291112,
        0.0007965632770601,
        0.0014389530166247,
        -0.0038334654556754,
        0.0016059112044046,
        0.0029010008853761,
    ]
).reshape(6, 9)

expected_v2 = -np.array(
    [
        -0.0036598018779382,
        0.0015331602461633,
        0.0027695797995208,
        0.0015331602461633,
        -0.0006422698328522,
        -0.0011602293754749,
        0.0034588126662543,
        -0.0014489620628903,
        -0.0026174798555438,
        -0.0041745421140984,
        0.0017487946694196,
        0.0031591129512096,
        0.0017487946694196,
        -0.0007326031723244,
        -0.0013234121822635,
        0.0025379870399672,
        -0.0010632107870133,
        -0.0019206388410563,
        -0.0007732439105683,
        -0.0007223726006625,
        0.0010581232460408,
        -0.0007223726006625,
        -0.0006748480874610,
        0.0009885098745908,
        0.0010581232460408,
        0.0009885098745908,
        -0.0014479581261611,
        -0.0007732439105683,
        -0.0007223726006625,
        0.0010581232460408,
        -0.0007223726006625,
        -0.0006748480874610,
        0.0009885098745908,
        0.0010581232460408,
        0.0009885098745908,
        -0.0014479581261611,
        0.0041056015792389,
        -0.0017199141750866,
        -0.0031069417356403,
        -0.0017199141750866,
        0.0007205045868606,
        0.0013015566730385,
        -0.0043680668171685,
        0.0018298658288138,
        0.0033055640778573,
        0.0021812275849517,
        -0.0009137575018041,
        -0.0016506587129364,
        -0.0009137575018041,
        0.0003827903048098,
        0.0006914921635274,
        -0.0017789317520491,
        0.0007452281663989,
        0.0013462186231723,
    ]
).reshape(6, 9)

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

sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file.resolve()} -o {pb_file.resolve()}".split()
)
sp.check_output(
    f"{sys.executable} -m deepmd convert-from pbtxt -i {pbtxt_file2.resolve()} -o {pb_file2.resolve()}".split()
)


def setup_module() -> None:
    if os.environ.get("ENABLE_TENSORFLOW", "1") != "1":
        pytest.skip(
            "Skip test because TensorFlow support is not enabled.",
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
    expected_md_v = (
        np.std([np.sum(expected_v[:], axis=0), np.sum(expected_v2[:], axis=0)], axis=0)
        / 4
    )
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))


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


def test_pair_deepmd_model_devi_atomic_relative_v(lammps) -> None:
    relative = 1.0
    lammps.pair_style(
        f"deepspin {pb_file.resolve()} {pb_file2.resolve()} out_file {md_file.resolve()} out_freq 1 atomic relative_v {relative}"
    )
    lammps.pair_coeff("* *")
    lammps.run(0)
    assert lammps.eval("pe") == pytest.approx(expected_e)
    for ii in range(4):
        assert lammps.atoms[ii].force == pytest.approx(
            expected_f[lammps.atoms[ii].id - 1]
        )
    md = np.loadtxt(md_file.resolve())
    expected_md_f = np.linalg.norm(np.std([expected_f, expected_f2], axis=0), axis=1)
    expected_md_fm = np.linalg.norm(np.std([expected_fm, expected_fm2], axis=0), axis=1)
    assert md[4] == pytest.approx(np.max(expected_md_f))
    assert md[5] == pytest.approx(np.min(expected_md_f))
    assert md[6] == pytest.approx(np.mean(expected_md_f))
    assert md[7] == pytest.approx(np.max(expected_md_fm))
    assert md[8] == pytest.approx(np.min(expected_md_fm))
    assert md[9] == pytest.approx(np.mean(expected_md_fm))
    expected_md_v = (
        np.std([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0) / 4
    )
    norm = (
        np.abs(
            np.mean([np.sum(expected_v, axis=0), np.sum(expected_v2, axis=0)], axis=0)
        )
        / 4
    )
    expected_md_v /= norm + relative
    assert md[1] == pytest.approx(np.max(expected_md_v))
    assert md[2] == pytest.approx(np.min(expected_md_v))
    assert md[3] == pytest.approx(np.sqrt(np.mean(np.square(expected_md_v))))


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
                "--nopbc",
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
