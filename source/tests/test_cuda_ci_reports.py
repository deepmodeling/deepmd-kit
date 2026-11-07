# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check CUDA Python timing reports without launching a GPU workflow."""

import os
import shlex
import subprocess
import sys
from pathlib import (
    Path,
)

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SHARDS = {
    "pt_expt-model": ["source/tests/pt_expt/model"],
    "pt_expt-other": ["source/tests/pt_expt", "--ignore=source/tests/pt_expt/model"],
    "other": ["source/tests", "--ignore=source/tests/pt_expt"],
}


@pytest.fixture
def python_job():
    config = yaml.safe_load((ROOT / ".github/workflows/test_cuda.yml").read_text())
    return config["jobs"]["test_python"]


def _test_command(job, shard):
    rows = {row["shard"]: row for row in job["strategy"]["matrix"]["include"]}
    assert set(rows) == set(SHARDS)
    steps = [step for step in job["steps"] if "-m pytest " in step.get("run", "")]
    assert len(steps) == 1  # Reporting must not duplicate the test invocation.
    step = steps[0]
    assert not step.get("continue-on-error", False)
    command = step["run"]
    for key, value in rows[shard].items():
        command = command.replace("${{ matrix." + key + " }}", str(value))
    assert "${{" not in command
    return command


@pytest.mark.parametrize("shard", SHARDS)
def test_python_shards_keep_selection_and_report_all_phases(python_job, shard):
    # pytest's total mode includes setup, call, and teardown in each case time.
    assert shlex.split(_test_command(python_job, shard)) == [
        "python",
        "-m",
        "pytest",
        *SHARDS[shard],
        "--ignore=source/tests/pd",
        f"--junitxml=test-reports/python-{shard}.xml",
        "-o",
        "junit_duration_report=total",
    ]


def test_python_report_upload_runs_after_failure_when_reports_exist(python_job):
    uploads = [
        step
        for step in python_job["steps"]
        if step.get("name") == "Upload Python test reports"
    ]
    assert len(uploads) == 1
    upload = uploads[0]
    assert python_job["steps"][-1] == upload
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert upload["if"] == (
        "${{ always() && hashFiles('test-reports/python-*.xml') != '' }}"
    )
    assert upload["with"]["path"] == "test-reports/python-*.xml"
    assert upload["with"]["if-no-files-found"] == "ignore"
    assert upload["with"]["retention-days"] == 14
    # An artifact-service failure is non-blocking, not an actual test failure.
    assert upload["continue-on-error"] is True
    names = {
        upload["with"]["name"].replace("${{ matrix.shard }}", shard) for shard in SHARDS
    }
    assert len(names) == len(SHARDS)
    assert all(name.startswith("python-cuda-") and "${{" not in name for name in names)


@pytest.mark.skipif(sys.platform != "linux", reason="Requires GNU/Linux shell tools")
@pytest.mark.parametrize("shard", SHARDS)
@pytest.mark.parametrize("status", [0, 17])
def test_python_report_command_preserves_exit_status(
    python_job, tmp_path, shard, status
):
    # Execute the actual workflow command with a lightweight pytest stand-in.
    # Both success and failure write one report from exactly one invocation.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1:3] == ['-m', 'pytest']\n"
        "with open('calls.txt', 'a') as stream: stream.write('pytest\\n')\n"
        "report = Path(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--junitxml=')))\n"
        "report.parent.mkdir(parents=True, exist_ok=True)\n"
        "report.write_text('<testsuites/>')\n"
        "sys.exit(int(os.environ['MOCK_STATUS']))\n"
    )
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _test_command(python_job, shard)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "MOCK_STATUS": str(status),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stderr
    assert (tmp_path / "calls.txt").read_text() == "pytest\n"
    assert {path.name for path in (tmp_path / "test-reports").iterdir()} == {
        f"python-{shard}.xml"
    }
