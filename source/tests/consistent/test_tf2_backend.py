# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import subprocess
import sys
from importlib.util import (
    find_spec,
)
from pathlib import (
    Path,
)

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(
    find_spec("tensorflow") is None, reason="TensorFlow is not installed"
)
def test_tf2_consistent_backend_subprocess() -> None:
    """Run TF2 consistent tests in a fresh process.

    Importing the TF1 backend disables eager execution process-wide, while the
    TF2 backend requires eager mode.  The subprocess sets an opt-in flag that
    tells ``source.tests.consistent.common`` to skip TF1 imports and enable the
    TF2 backend tests instead.
    """
    env = os.environ.copy()
    env["DEEPMD_TEST_TF2"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]] if "PYTHONPATH" in env else [str(REPO_ROOT)]
    )
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "source/tests/consistent/test_array_api.py",
            "source/tests/consistent/model/test_ener.py",
            "source/tests/consistent/descriptor",
            "source/tests/consistent/fitting",
            "-k",
            "tf2",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "TF2 consistent subprocess failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
