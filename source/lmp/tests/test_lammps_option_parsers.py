# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression tests for malformed DeePMD LAMMPS option lists.

The existing LAMMPS tests exercise complete, valid option lists. These cases
run out of process so a future out-of-bounds parser regression fails one test
without crashing the entire pytest worker.
"""

import os
import subprocess as sp
import sys
from pathlib import (
    Path,
)

import pytest
from lammps_test_utils import (
    require_backend,
)
from model_convert import (
    ensure_converted_pb,
)

pbtxt_file = (
    Path(__file__).parent.parent.parent / "tests" / "infer" / "deepspin_nlist.pbtxt"
)

_LAMMPS_RUNNER = """
import os
import sys

from lammps import lammps

lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])
try:
    if plugin := os.environ.get("DEEPMD_TEST_PLUGIN"):
        lmp.command(f"plugin load {plugin}")
    for command in sys.argv[1:]:
        lmp.command(command)
finally:
    lmp.close()
"""


def setup_module() -> None:
    require_backend("ENABLE_TENSORFLOW", "TensorFlow")


@pytest.fixture(scope="module")
def spin_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a one-spin-type model used to size spin option value lists."""
    model = tmp_path_factory.mktemp("lammps_parser") / "deepspin.pb"
    ensure_converted_pb(pbtxt_file, model)
    return model


def _run_lammps(*commands: str) -> sp.CompletedProcess[str]:
    """Isolate malformed commands that could crash an unfixed LAMMPS parser."""
    return sp.run(
        [sys.executable, "-c", _LAMMPS_RUNNER, *commands],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=os.environ.copy(),
    )


def _assert_lammps_error(result: sp.CompletedProcess[str], message: str) -> None:
    """Require the controlled diagnostic, not merely a crashed subprocess."""
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert f"ERROR: {message} (" in output, output


@pytest.mark.parametrize(
    ("style", "option", "message"),
    [
        ("deepmd", "relative", "Illegal relative, not provided"),
        ("deepmd", "relative_v", "Illegal relative_v, not provided"),
        ("deepspin", "relative", "Illegal relative, not provided"),
        ("deepspin", "relative_v", "Illegal relative_v, not provided"),
        (
            "deepspin",
            "virtual_len",
            "Illegal virtual_len, the dimension should be 1",
        ),
        (
            "deepspin",
            "spin_norm",
            "Illegal spin_norm, the dimension should be 1",
        ),
    ],
)
def test_pair_style_rejects_truncated_option(
    spin_model: Path, style: str, option: str, message: str
) -> None:
    result = _run_lammps(
        "units metal", f"pair_style {style} {spin_model.resolve()} {option}"
    )
    _assert_lammps_error(result, message)


@pytest.mark.parametrize("style", ["deepmd", "deepspin"])
@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("relative", "Illegal relative, not provided"),
        ("relative_v", "Illegal relative_v, not provided"),
    ],
)
def test_pair_style_rejects_keyword_as_relative_value(
    spin_model: Path, style: str, option: str, message: str
) -> None:
    result = _run_lammps(
        "units metal",
        f"pair_style {style} {spin_model.resolve()} {option} atomic",
    )
    _assert_lammps_error(result, message)


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("virtual_len", "Illegal virtual_len, the dimension should be 1"),
        ("spin_norm", "Illegal spin_norm, the dimension should be 1"),
    ],
)
def test_deepspin_rejects_keyword_as_spin_value(
    spin_model: Path, option: str, message: str
) -> None:
    result = _run_lammps(
        "units metal",
        f"pair_style deepspin {spin_model.resolve()} {option} atomic",
    )
    _assert_lammps_error(result, message)


def test_deepspin_accepts_complete_option_values(spin_model: Path) -> None:
    """Keep the valid combined option path covered alongside negative cases."""
    result = _run_lammps(
        "units metal",
        f"pair_style deepspin {spin_model.resolve()} relative 1.0 relative_v 2.0 "
        "virtual_len 0.4 spin_norm 1.2737 atomic",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("fix_options", "message"),
    [
        ("model", "Illegal fix dplr model option, not provided"),
        (
            "model efield 0 0 0",
            "Illegal fix dplr model option, not provided",
        ),
        (
            "model {model} efield 0 0",
            "Illegal fix dplr efield option, three values are required",
        ),
        (
            "model {model} efield 0 bond_type 1",
            "Illegal fix dplr efield option, three values are required",
        ),
        (
            "model {model} type_associate 1",
            "Illegal fix dplr type_associate option, an even number of atom "
            "types is required",
        ),
    ],
)
def test_fix_dplr_rejects_malformed_options(
    spin_model: Path, fix_options: str, message: str
) -> None:
    result = _run_lammps(
        "units metal",
        "atom_style full",
        "region box block 0 1 0 1 0 1",
        "create_box 1 box",
        f"fix 0 all dplr {fix_options.format(model=spin_model.resolve())}",
    )
    _assert_lammps_error(result, message)
