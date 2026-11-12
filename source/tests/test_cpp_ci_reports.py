# SPDX-License-Identifier: LGPL-3.0-or-later
"""Check C++ CI report wiring without launching builds or GitHub workflows."""

import csv
import json
import os
import re
import shutil
import signal
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
    "source/build_tests/pytest-reports/**",
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
        or "source/install/pytest_with_junit.sh" in step.get("run", "")
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


def test_ordinary_pytest_reporting_preserves_commands_and_failure_handlers(workflow):
    filename, job, _ = workflow
    commands = [
        line.strip()
        for step in job["steps"]
        for line in step.get("run", "").splitlines()
        if "source/install/pytest_with_junit.sh" in line
    ]
    if filename == "test_cc.yml":
        assert commands == [
            "source/install/retry_on_sigterm.sh 3 source/install/pytest_with_junit.sh "
            "lammps python -X faulthandler -m pytest -vv --capture=tee-sys "
            "--cov=deepmd source/lmp/tests",
            "source/install/pytest_with_junit.sh ipi pytest --cov=deepmd source/ipi/tests",
        ]
    else:
        assert commands == [
            "source/install/pytest_with_junit.sh lammps python -m pytest -s "
            "source/lmp/tests || (cat log.lammps && exit 1)",
            "source/install/pytest_with_junit.sh ipi python -m pytest source/ipi/tests",
        ]


@pytest.fixture
def ordinary_pytest_command(tmp_path):
    # Copy the real helper to a fake checkout so no reports touch the source tree.
    install_dir = tmp_path / "source/install"
    install_dir.mkdir(parents=True)
    helper = install_dir / "pytest_with_junit.sh"
    shutil.copy2(ROOT / "source/install/pytest_with_junit.sh", helper)
    command = tmp_path / "pytest-command"
    command.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys\n"
        "from pathlib import Path\n"
        "with open('calls.jsonl', 'a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "report = Path(sys.argv[-1].removeprefix('--junitxml='))\n"
        "assert sys.argv[-1].startswith('--junitxml=')\n"
        "statuses = os.environ['MOCK_STATUSES'].split(',')\n"
        "count = len(Path('calls.jsonl').read_text().splitlines())\n"
        "status = statuses[min(count, len(statuses)) - 1]\n"
        "if status == 'SIGTERM': os.kill(os.getpid(), signal.SIGTERM)\n"
        "report.write_text('<testsuites/>')\n"
        "sys.exit(int(status))\n"
    )
    command.chmod(0o755)
    return helper, command


@pytest.mark.skipif(sys.platform != "linux", reason="Requires GNU/Linux shell tools")
@pytest.mark.parametrize("status", [0, 1, 5, 17, 143, "SIGTERM"])
def test_ordinary_pytest_reports_preserve_status_arguments_and_invocations(
    tmp_path, ordinary_pytest_command, status
):
    helper, command = ordinary_pytest_command
    arguments = ["-m", "pytest", "--cov=deepmd", "test directory", "-k", "one or two"]
    for invocation, name in enumerate(["lammps", "ipi", "lammps"], start=1):
        result = subprocess.run(
            [str(helper), name, str(command), *arguments],
            cwd=tmp_path,
            env={**os.environ, "MOCK_STATUSES": str(status)},
            capture_output=True,
            text=True,
            check=False,
        )
        expected_status = -signal.SIGTERM if status == "SIGTERM" else status
        assert result.returncode == expected_status, result.stderr
        calls = [
            json.loads(line)
            for line in (tmp_path / "calls.jsonl").read_text().splitlines()
        ]
        assert len(calls) == invocation
        assert all(args[:-1] == arguments for args in calls)
        report_root = tmp_path / "source/build_tests/pytest-reports"
        reports = list(report_root.glob("run.*/*.xml"))
        # A killed pytest process may never finish writing its XML report.
        assert len(reports) == (0 if status == "SIGTERM" else invocation)
        assert len(list(report_root.glob("run.*"))) == invocation
        assert Path(calls[-1][-1].split("=", 1)[1]).name == f"{name}.xml"


@pytest.mark.skipif(sys.platform != "linux", reason="Requires GNU/Linux shell tools")
@pytest.mark.parametrize(
    ("statuses", "status", "attempts", "report_count"),
    [
        ("143,0", 0, 2, 2),
        ("143", 143, 3, 3),
        ("17", 17, 1, 1),
        ("SIGTERM,0", 0, 2, 1),
        ("SIGTERM", 143, 3, 0),
    ],
)
def test_ordinary_pytest_retry_preserves_each_attempt(
    tmp_path, ordinary_pytest_command, statuses, status, attempts, report_count
):
    helper, command = ordinary_pytest_command
    # Skip only the retry delay, without replacing the actual retry script.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/bash\nexit 0\n")
    sleep.chmod(0o755)
    result = subprocess.run(
        [
            str(ROOT / "source/install/retry_on_sigterm.sh"),
            "3",
            str(helper),
            "lammps",
            str(command),
            "--cov=deepmd",
            "source/lmp/tests",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
            "MOCK_STATUSES": statuses,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stderr
    report_root = tmp_path / "source/build_tests/pytest-reports"
    reports = list(report_root.glob("run.*/lammps.xml"))
    assert len(reports) == report_count
    assert len(list(report_root.glob("run.*"))) == attempts
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == attempts


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


def test_kokkos_uses_a_fixed_spin_capable_lammps_release() -> None:
    config = yaml.safe_load((ROOT / ".github/workflows/test_cuda.yml").read_text())
    steps = config["jobs"]["test_cc"]["steps"]
    overrides = [step for step in steps if "LAMMPS_VERSION" in step.get("env", {})]
    assert len(overrides) == 1
    assert overrides[0]["if"] == "matrix.suite == 'kokkos'"
    assert overrides[0]["env"]["LAMMPS_VERSION"] == "patch_4Jul2026"
    assert "source/install/test_cc_local.sh --build-only" in overrides[0]["run"]
    script = (ROOT / "source/install/test_lammps_kokkos.sh").read_text()
    assert script.count("-D CMAKE_CXX_STANDARD=20") == 2


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
def test_kokkos_report_commands_preserve_pytest_status(
    tmp_path: Path, failure: str
) -> None:
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
        "expected = {\n"
        "    'spin.xml': {'test_dpa4spin_kokkos_magnetic_force.py', 'test_lammps_dpa4c_multiblock_kokkos.py'},\n"
        "    'long-range.xml': {'test_lammps_dpa4c_long_range_kokkos.py'},\n"
        "}\n"
        "assert {Path(a).name for a in sys.argv if a.endswith('.py')} == expected[report.name]\n"
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
