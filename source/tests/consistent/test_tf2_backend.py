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
    backend_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from deepmd.backend.tensorflow import Backend; "
                "raise SystemExit(0 if Backend.get_backend('tf2')().is_available() "
                "else 1)"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if backend_result.returncode != 0:
        pytest.fail(
            "TF2 backend is not registered or available under DEEPMD_TEST_TF2=1\n"
            f"stdout:\n{backend_result.stdout}\n"
            f"stderr:\n{backend_result.stderr}"
        )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "source/tests/consistent/test_array_api.py",
            "source/tests/consistent/model",
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
    if "passed" not in result.stdout:
        pytest.fail(
            "TF2 consistent subprocess did not execute any passing TF2 tests\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
