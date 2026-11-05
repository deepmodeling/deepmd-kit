# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check C++ CI report wiring without launching builds or GitHub workflows."""

import csv
import os
import re
import subprocess
import sys
from pathlib import (
    Path,
)

import pytest
import yaml

ROOT = Path(__file__).parents[2]
COMMON_REPORTS = {
    "source/build_tests/generator-reports/**",
    "source/build_tests/gtest-reports/**",
}


@pytest.fixture(params=[("test_cc.yml", "testcc"), ("test_cuda.yml", "test_cc")])
def workflow(request):
    filename, job_name = request.param
    config = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())
    job = config["jobs"][job_name]
    uploads = [s for s in job["steps"] if s.get("name") == "Upload C++ test reports"]
    assert len(uploads) == 1
    return filename, job, uploads[0]


def test_report_upload_runs_after_failure_only_when_files_exist(workflow):
    filename, job, upload = workflow
    assert upload["uses"].startswith("actions/upload-artifact@")
    assert "always()" in upload["if"]
    assert " != ''" in upload["if"]
    condition_paths = set(re.findall(r"'([^']+)'", upload["if"]))
    paths = set(upload["with"]["path"].splitlines())
    expected = COMMON_REPORTS.copy()
    if filename == "test_cuda.yml":
        expected.add("source/build_lammps_kokkos/test-reports/**")
    assert paths == expected
    assert condition_paths == expected
    assert "hashFiles(" in upload["if"]
    assert upload["with"]["if-no-files-found"] == "ignore"
    assert upload["with"]["retention-days"] == 14
    assert job["steps"][-1] == upload


def test_report_upload_cannot_replace_test_failure_with_success(workflow):
    _, job, upload = workflow
    # An artifact-service failure is non-blocking, but actual test steps keep
    # their normal failure outcome and the existing required-job gate.
    assert upload["continue-on-error"] is True
    test_steps = [
        step
        for step in job["steps"]
        if "source/install/test_cc_local.sh" in step.get("run", "")
        or "source/install/test_lammps_kokkos.sh" in step.get("run", "")
    ]
    assert test_steps
    for step in test_steps:
        assert not step.get("continue-on-error", False)
        assert "|| true" not in step["run"]


def test_matrix_jobs_have_distinct_artifact_names(workflow):
    _, job, upload = workflow
    names = []
    for row in job["strategy"]["matrix"]["include"]:
        name = upload["with"]["name"]
        for key, value in row.items():
            name = name.replace("${{ matrix." + key + " }}", str(value).lower())
        assert "${{" not in name
        names.append(name)
    assert len(set(names)) == len(names)


def test_kokkos_pytest_processes_write_distinct_reports():
    script = (ROOT / "source/install/test_lammps_kokkos.sh").read_text()
    assert (
        'TEST_REPORT_DIR=$(mktemp -d "${BUILD_ROOT}/test-reports/run.XXXXXX")' in script
    )
    assert re.findall(r'--junitxml="([^\"]+)"', script) == [
        "${TEST_REPORT_DIR}/spin.xml",
        "${TEST_REPORT_DIR}/long-range.xml",
    ]
    assert script.startswith("#!/bin/bash\nset -euxo pipefail\n")


@pytest.mark.skipif(sys.platform != "linux", reason="Requires GNU/Linux shell tools")
@pytest.mark.parametrize("status", [0, 17])
def test_kokkos_spin_generator_reports_preserve_status_and_invocations(
    tmp_path, status
):
    script = (ROOT / "source/install/test_lammps_kokkos.sh").read_text()
    command = script[
        script.index("# Record the spin fixture") : script.index("# The wheel's")
    ]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1:] == ['-u', str(Path(os.environ['REPO_ROOT']) / 'source/tests/infer/gen_dpa4c_spin_lmp_kokkos.py')]\n"
        "with open('calls.txt', 'a') as stream: stream.write('generator\\n')\n"
        "sys.exit(int(os.environ['MOCK_STATUS']))\n"
    )
    python.chmod(0o755)
    build = tmp_path / "build"
    for invocation in (1, 2):
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=tmp_path,
            env={
                **os.environ,
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                "REPO_ROOT": str(tmp_path),
                "BUILD_ROOT": str(build),
                "MOCK_STATUS": str(status),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == status, result.stderr
        assert (tmp_path / "calls.txt").read_text() == "generator\n" * invocation
        reports = list((build / "test-reports").glob("run.*/*.tsv"))
        assert len(reports) == invocation
        for report in reports:
            assert report.name == "gen_dpa4c_spin_lmp_kokkos.tsv"
            with report.open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            assert len(rows) == 1
            assert set(rows[0]) == {"generator", "seconds", "exit_status"}
            assert rows[0]["generator"] == report.stem
            assert int(rows[0]["seconds"]) >= 0
            assert int(rows[0]["exit_status"]) == status


@pytest.mark.skipif(sys.platform != "linux", reason="Requires GNU/Linux shell tools")
@pytest.mark.parametrize("failure", ["", "spin.xml", "long-range.xml"])
def test_kokkos_report_commands_preserve_pytest_status(tmp_path, failure):
    # Execute the unmodified test-command tail with a tiny pytest stand-in.
    # This tests exit-status/report wiring, not the CUDA/LAMMPS computation.
    script = (ROOT / "source/install/test_lammps_kokkos.sh").read_text()
    tail = script[script.index("# Keep each invocation and each pytest process") :]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert sys.argv[1:3] == ['-m', 'pytest']\n"
        "report = Path(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--junitxml=')))\n"
        "report.write_text('<testsuites/>')\n"
        "sys.exit(17 if report.name == os.environ['MOCK_FAIL'] else 0)\n"
    )
    python.chmod(0o755)
    build = tmp_path / "build"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", tail],
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "REPO_ROOT": str(tmp_path),
            "BUILD_ROOT": str(build),
            "MPI_ROOT": str(tmp_path),
            "MOCK_FAIL": failure,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (17 if failure else 0), result.stderr
    directories = list((build / "test-reports").glob("run.*"))
    assert len(directories) == 1
    expected = {"spin.xml"}
    if failure != "spin.xml":
        expected.add("long-range.xml")
    assert {path.name for path in directories[0].iterdir()} == expected
