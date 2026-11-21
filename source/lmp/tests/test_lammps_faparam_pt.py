# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test LAMMPS with default_fparam (PyTorch backend)."""

import importlib
import os
from pathlib import (
    Path,
)

import numpy as np
import pytest
from lammps import lammps as Lammps
from write_lmp_data import (
    write_lmp_data,
)

pth_file = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "fparam_aparam_default.pth"
)
data_file = Path(__file__).parent / "data.lmp"

# expected values from fparam_aparam_default.pth with default_fparam=[0.25852028]
expected_ae = np.array(
    [
        -1.038271223729637e-01,
        -7.285433579124989e-02,
        -9.467600492266426e-02,
        -1.467050207422953e-01,
        -7.660561676973243e-02,
        -7.277296000253175e-02,
    ]
)
expected_e = np.sum(expected_ae)
expected_f = np.array(
    [
        6.622266941151356e-02,
        5.278739714221517e-02,
        2.265728009692279e-02,
        -2.606048291367521e-02,
        -4.538812303131843e-02,
        1.058247419681242e-02,
        1.679392617013225e-01,
        -2.257826240741907e-03,
        -4.490146347357200e-02,
        -1.148364179422036e-01,
        -1.169790528013792e-02,
        6.140403441496690e-02,
        -8.078778123309406e-02,
        -5.838879041789346e-02,
        6.773641084621368e-02,
        -1.247724902386317e-02,
        6.494524782787654e-02,
        -1.174787360813438e-01,
    ]
).reshape(6, 3)

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


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    # Initialize Python autograd before the LAMMPS C++ backend uses it.
    importlib.import_module("torch")
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _lammps(data_file) -> Lammps:
    lmp = Lammps(cmdargs=["-log", "none", "-screen", "none"])
    lmp.commands_list(
        [
            "units metal",
            "boundary p p p",
            "atom_style atomic",
            "neighbor 2.0 bin",
            "neigh_modify every 10 delay 0 check no",
            f"read_data {data_file.resolve()}",
            "mass 1 16",
            "timestep 0.0005",
            "fix 1 all nve",
        ]
    )
    return lmp


def _forces(lmp) -> np.ndarray:
    count = lmp.extract_global("nlocal")
    tags = lmp.numpy.extract_atom("id")[:count]
    return lmp.numpy.extract_atom("f")[:count][np.argsort(tags)].copy()


@pytest.fixture
def lammps():
    lmp = _lammps(data_file=data_file)
    yield lmp
    lmp.close()


def test_pair_deepmd_default_fparam(lammps) -> None:
    """Test that model with default_fparam works without providing fparam."""
    lammps.command(f"pair_style deepmd {pth_file.resolve()} aparam 0.25852028")
    lammps.command("pair_coeff * *")
    lammps.command("run 0")
    assert lammps.get_thermo("pe") == pytest.approx(expected_e)
    np.testing.assert_allclose(_forces(lammps), expected_f, rtol=1.0e-6, atol=1.0e-8)
    lammps.command("run 1")


def test_pair_deepmd_default_fparam_explicit(lammps) -> None:
    """Test that explicit fparam still works with default_fparam model."""
    lammps.command(
        f"pair_style deepmd {pth_file.resolve()} fparam 0.25852028 aparam 0.25852028"
    )
    lammps.command("pair_coeff * *")
    lammps.command("run 0")
    assert lammps.get_thermo("pe") == pytest.approx(expected_e)
    np.testing.assert_allclose(_forces(lammps), expected_f, rtol=1.0e-6, atol=1.0e-8)
    lammps.command("run 1")


@pytest.mark.parametrize("source", ["compute", "fix", "atom_compute"])
def test_parameter_source_lookup(lammps, source) -> None:
    """Read fix/compute parameters through the supported LAMMPS API."""
    value = 0.25852028
    lammps.command(
        "unfix 1"
    )  # Keep coordinates fixed while ave/time samples its input.
    if source == "compute":
        lammps.command(f"variable atom_parameter atom {value}")
        lammps.command("compute parameter all reduce ave v_atom_parameter")
        options = f"fparam_from_compute parameter aparam {value}"
        derivative_source = "c_parameter"
    elif source == "fix":
        lammps.command(f"variable parameter equal {value}")
        lammps.command("fix parameter all ave/time 1 1 1 v_parameter")
        options = f"fparam_from_fix parameter aparam {value}"
        derivative_source = "f_parameter"
    else:
        lammps.command("fix values all property/atom d_parameter")
        lammps.command(f"set atom * d_parameter {value}")
        lammps.command("compute parameter all property/atom d_parameter")
        options = f"fparam {value} aparam_from_compute parameter"
        derivative_source = None
    lammps.command(f"pair_style deepmd {pth_file.resolve()} {options}")
    lammps.command("pair_coeff * *")
    lammps.command("run 1")
    if derivative_source is not None:
        lammps.command(f"compute dedn all deepmd/fparam/dedn {derivative_source}")
        lammps.command("thermo_style custom step pe c_dedn")
    lammps.command("run 0")
    assert lammps.get_thermo("pe") == pytest.approx(expected_e)
    np.testing.assert_allclose(_forces(lammps), expected_f, rtol=1.0e-6, atol=1.0e-8)
    if derivative_source is not None:
        from deepmd.infer import (
            DeepPot,
        )

        potential = DeepPot(str(pth_file))
        eps = 1.0e-6
        energies = [
            potential.eval(
                coord.reshape(1, -1),
                np.diag([13.0, 13.0, 13.0]).reshape(1, -1),
                np.zeros(6, dtype=int),
                fparam=np.array([[value + offset]]),
                aparam=np.full((1, 6), value),
            )[0].item()
            for offset in [-eps, eps]
        ]
        reference = (energies[1] - energies[0]) / (2.0 * eps)
        assert lammps.extract_compute("dedn", 0, 0) == pytest.approx(
            reference, rel=1.0e-4, abs=1.0e-4
        )
