# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the Kokkos pair style with compressed DPA4C models that carry a long-range module.

``pair_style deepmd/kk`` evaluates a compressed long-range model through the
device-resident compact canonical lower: it gathers the positions of the
model nodes relative to the origin of the box, passes the cell of a periodic
system, and joins the exchange of the long range on every rank. Run
source/install/test_lammps_kokkos.sh after test_cc_local.sh on a CUDA host:
that job supplies the compressed fixtures of
source/tests/infer/gen_dpa4c_long_range.py and a Kokkos LAMMPS library and
plugin, and sets DPA4C_LONG_RANGE_KOKKOS_REQUIRED=1, so that no missing
prerequisite skips the test silently.

The six atoms of the fixtures, and their box, are translated away from the
origin: the positions relative to the box, and with them every output of the
references, including the per-atom virial, which depends on the origin of a
periodic cell, stay those of the references.
"""

from __future__ import (
    annotations,
)

import importlib.util
import os
import shutil
from pathlib import (
    Path,
)

import constants
import numpy as np
import pytest
from expected_ref import (
    read_expected_ref,
)
from lammps_test_utils import (
    run_mpi_pair_runner,
)
from write_lmp_data import (
    write_lmp_data,
)

_INFER = Path(__file__).resolve().parents[2] / "tests" / "infer"
_RUNNER = Path(__file__).with_name("run_mpi_pair_deepmd_dpa3_pt2.py")
# The translation of the atoms and of the box.
_ORIGIN = np.array([-4.0, 2.5, 7.0])
_COORD = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
# atype [0, 1, 1, 0, 1, 1] (O, H, H, O, H, H) -> LAMMPS types [1, 2, 2, 1, 2, 2]
_TYPE = np.array([1, 2, 2, 1, 2, 2])
# The box of the non-periodic references, and the triclinic cell of the
# periodic ones (gen_dpa4c_long_range.BOX): xlo xhi ylo yhi zlo zhi xy xz yz.
_BOX = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0], dtype=float)
_PERIODIC_BOX = np.array([0, 13, 0, 12.5, 0, 13.5, 0.6, -0.4, 0.8])
# LAMMPS centroid/stress/atom reports the per-atom stress, the negative of the
# per-atom virial, with the components xx, yy, zz, xy, xz, yz, yx, zx, zy; these
# columns of it give the nine components of the reference virial.
_TO_REFERENCE = [0, 6, 7, 3, 1, 8, 4, 5, 2]


@pytest.fixture(scope="module")
def kokkos_job() -> None:
    # The ordinary LAMMPS suite also runs on CUDA, but uses a non-Kokkos wheel.
    # Only the dedicated matching library/plugin job opts into this regression.
    if os.environ.get("DPA4C_LONG_RANGE_KOKKOS_REQUIRED") != "1":
        pytest.skip("Enable the dedicated CUDA Kokkos plugin regression explicitly")
    for module in ("lammps", "mpi4py"):
        if importlib.util.find_spec(module) is None:
            pytest.fail(f"Required Python package missing: {module}")
    if shutil.which("mpirun") is None:
        pytest.fail("mpirun required for the Kokkos regression")


def _translated(box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The box and the positions of the atoms, translated by ``_ORIGIN``."""
    shifted = box.copy()
    shifted[0:6] += np.repeat(_ORIGIN, 2)
    return shifted, _COORD + _ORIGIN


@pytest.mark.parametrize("nprocs", [1, 2])
@pytest.mark.parametrize(
    ("stem", "boundary", "section", "pair_extra"),
    [
        ("deeppot_dpa4c_long_range_compressed", "f f f", "default", ""),
        (
            "deeppot_dpa4c_long_range_compressed",
            "f f f",
            "explicit",
            "charge_spin 1.0 2.0",
        ),
        ("deeppot_dpa4c_long_range_periodic_compressed", "p p p", "periodic", ""),
        ("deeppot_dpa4c_long_range_periodic_compressed", "f f f", "cluster", ""),
    ],
)
def test_pair_deepmd_kokkos_long_range(
    kokkos_job: None,
    tmp_path: Path,
    stem: str,
    boundary: str,
    section: str,
    pair_extra: str,
    nprocs: int,
) -> None:
    model = _INFER / f"{stem}.pt2"
    if not model.is_file():
        pytest.fail(f"Generate the compressed long-range fixture first: {model}")
    reference = read_expected_ref(_INFER / f"{stem}.expected")[section]
    box, coord = _translated(_PERIODIC_BOX if "periodic" in stem else _BOX)
    data = tmp_path / "six.lmp"
    write_lmp_data(box, coord, _TYPE, data)
    runner_args = ["--boundary", boundary, "--kokkos"]
    if pair_extra:
        runner_args += ["--pair-extra", pair_extra]
    result = run_mpi_pair_runner(
        _RUNNER,
        data,
        model,
        nprocs=nprocs,
        processors="1 1 1" if nprocs == 1 else "2 1 1",
        runner_args=runner_args,
        timeout=600,
    )
    assert result["pe"] == pytest.approx(
        float(np.sum(reference["expected_e"])), rel=1e-6
    )
    np.testing.assert_allclose(
        result["forces"], reference["expected_f"].reshape(6, 3), rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        result["virials"][:, _TO_REFERENCE] / constants.nktv2p,
        -reference["expected_v"].reshape(6, 9),
        rtol=0,
        atol=1e-5,
    )
