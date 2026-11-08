# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exercise the C++ CI entry point without compiling backend libraries."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import (
    Path,
)

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="The build script requires GNU/Linux utilities"
)

SCRIPT = Path(__file__).parents[1] / "install" / "test_cc_local.sh"
MODELS = {
    "gen_sea.py",
    "gen_dpa1.py",
    "gen_dpa2.py",
    "gen_dpa3.py",
    "gen_fparam_aparam.py",
    "gen_model_devi.py",
    "gen_chg_spin.py",
    "gen_dpa4.py",
    "gen_dpa1_pairexcl.py",
    "gen_spin.py",
    "gen_spin_model_devi.py",
    "gen_corrupt_with_comm.py",
    "gen_dpa4_spin.py",
    "gen_dpa4_zbl.py",
    "gen_dpa4_spin_chgspin.py",
    "gen_dpa4_spin_zbl.py",
    "gen_dpa4c_spin_lmp.py",
    "gen_dpa4c_spin.py",
    "gen_dpa4c_canonical.py",
    "gen_dpa4c_long_range.py",
}
REPORTS = {Path(model).stem for model in MODELS} - {"gen_spin"} | {
    "gen_spin_dpa3",
    "gen_spin_reuse",
}


@pytest.fixture
def build_tree(tmp_path):
    root = tmp_path / "repo"
    script = root / "source/install/test_cc_local.sh"
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    infer = root / "source/tests/infer"
    infer.mkdir(parents=True)
    op = root / "source/build_tests/op/pt/libdeepmd_op_pt.so"
    op.parent.mkdir(parents=True)
    op.write_text("custom op from this build")
    c_api = root / "dp_test/lib/libdeepmd_c.so"
    c_api.parent.mkdir(parents=True)
    c_api.touch()
    fake_package = tmp_path / "packages/deepmd"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").touch()
    (fake_package / "env.py").write_text(
        "import os\nfrom pathlib import Path\n"
        'SHARED_LIB_DIR = Path(os.environ["MOCK_SHARED_LIB_DIR"])\n'
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = f"""#!{sys.executable}
import json
import os
import subprocess
import sys
import time
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
dpa1_ready = Path(os.environ["MOCK_DPA1_READY"])
dpa3_ready = Path(os.environ["MOCK_SPIN_DPA3_READY"])
dpa3_mpi_ready = Path(os.environ["MOCK_SPIN_DPA3_MPI_READY"])
model0_ready = Path(os.environ["MOCK_MODEL0_READY"])
schedule_dir = Path(os.environ["MOCK_SCHEDULE_DIR"])

def schedule_event(event, model):
    with open(os.environ["MOCK_SCHEDULE_LOG"], "a") as stream:
        stream.write(json.dumps({{"event": event, "model": model}}) + "\\n")
    (schedule_dir / (model + "." + event)).touch()

def wait_for_event(event, model):
    deadline = time.monotonic() + 10
    while not (schedule_dir / (model + "." + event)).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Generator pipeline did not release " + model + "." + event)
        time.sleep(0.01)

record = {{
    "name": name,
    "args": args,
    "ld": os.environ.get("LD_LIBRARY_PATH", ""),
    "gtest_output": os.environ.get("GTEST_OUTPUT", ""),
    "dpa1_ready": dpa1_ready.exists(),
    "dpa3_ready": dpa3_ready.exists() and dpa3_mpi_ready.exists(),
    "model0_ready": model0_ready.exists(),
}}
with open(os.environ["MOCK_LOG"], "a") as stream:
    stream.write(json.dumps(record) + "\\n")
if os.environ.get("MOCK_FAIL") and any(os.environ["MOCK_FAIL"] == Path(a).name for a in [name, *args]):
    sys.exit(17)
if name == "python" and args[0] == "-c":
    sys.exit(subprocess.call([sys.executable, "-S", *args]))
if name == "python" and args[0] == "-u":
    model = Path(args[1]).name
    schedule_event("start", model)
    if os.environ.get("MOCK_CHECK_SCHEDULE"):
        # Hold DPA2 open until both replacement jobs complete. This proves
        # actual overlap without relying on timing or machine speed.
        if model == "gen_sea.py":
            wait_for_event("start", "gen_dpa1.py")
            wait_for_event("start", "gen_dpa2.py")
        elif model == "gen_dpa1.py":
            wait_for_event("start", "gen_dpa4.py")
        elif model == "gen_dpa2.py":
            wait_for_event("end", "gen_dpa4.py")
            wait_for_event("end", "gen_dpa3.py")
        elif model == "gen_dpa4.py":
            wait_for_event("start", "gen_dpa3.py")
    if Path(args[1]).name == "gen_dpa1.py":
        # Make a missing wait on this producer visible to the reuse consumer.
        time.sleep(0.05)
        dpa1_ready.touch()
    if "--reuse-dpa1" in args and not dpa1_ready.exists():
        sys.exit(23)
    if "--dpa3-only" in args:
        time.sleep(0.05)
        dpa3_ready.touch()
        dpa3_mpi_ready.touch()
    if Path(args[1]).name == "gen_spin_model_devi.py":
        time.sleep(0.05)
        model0_ready.touch()
    if "--reuse-model-devi" in args and not (dpa3_ready.exists() and dpa3_mpi_ready.exists() and model0_ready.exists()):
        sys.exit(23)
    schedule_event("end", model)
if name == "ctest":
    Path(args[args.index("--output-junit") + 1]).write_text("<testsuites/>")
"""
    for name in ("cmake", "python", "ctest"):
        executable = bin_dir / name
        executable.write_text(stub)
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(fake_package.parent),
        "DP_VARIANT": "cuda",
        "ENABLE_TENSORFLOW": "TRUE",
        "ENABLE_PYTORCH": "TRUE",
        "ENABLE_PADDLE": "FALSE",
        "LD_LIBRARY_PATH": "/existing/lib",
        # Do not inherit sanitizer flags from the caller.
        "CXXFLAGS": "",
        "GTEST_OUTPUT": "",
        "MOCK_LOG": str(tmp_path / "commands.jsonl"),
        "MOCK_DPA1_READY": str(infer / "deeppot_dpa1_graph.pt2"),
        "MOCK_SPIN_DPA3_READY": str(infer / "deeppot_dpa3_spin.pt2"),
        "MOCK_SPIN_DPA3_MPI_READY": str(infer / "deeppot_dpa3_spin_mpi.pt2"),
        "MOCK_MODEL0_READY": str(infer / "deeppot_dpa_spin_md0.pt2"),
        "MOCK_SHARED_LIB_DIR": str(tmp_path / "installed_ops"),
        "MOCK_FAIL": "",
        "MOCK_SCHEDULE_DIR": str(tmp_path / "schedule"),
        "MOCK_SCHEDULE_LOG": str(tmp_path / "schedule.jsonl"),
        "MOCK_CHECK_SCHEDULE": "",
    }
    Path(env["MOCK_SCHEDULE_DIR"]).mkdir()
    return root, script, env


def run_script(build_tree, *args, **overrides):
    root, script, env = build_tree
    env = {**env, **overrides}
    result = subprocess.run(
        ["bash", str(script), *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = Path(env["MOCK_LOG"])
    commands = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, commands


def generated_models(commands):
    return [
        Path(c["args"][1]).name
        for c in commands
        if c["name"] == "python" and c["args"][0] == "-u"
    ]


def generator_reports(root):
    reports = {}
    for path in (root / "source/build_tests/generator-reports").glob("run.*/*.tsv"):
        with path.open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == 1
        row = rows[0]
        assert set(row) == {"generator", "seconds", "exit_status"}
        assert row["generator"] == path.stem
        assert int(row["seconds"]) >= 0
        reports[path] = row
    return reports


def check_ctest_report(root, command, gtest_output=None):
    assert command["name"] == "ctest"
    assert command["args"][:2] == ["--output-on-failure", "--output-junit"]
    assert len(command["args"]) == 3
    junit = Path(command["args"][2]).resolve()
    assert junit.name == "ctest.xml"
    assert junit.parent.parent == root / "source/build_tests/gtest-reports"
    assert junit.parent.name.startswith("run.")
    assert junit.is_file()
    if gtest_output is None:
        kind, path = command["gtest_output"].split(":", 1)
        assert kind == "json"
        assert path.endswith("/")
        assert Path(path).resolve() == junit.parent
    else:
        assert command["gtest_output"] == gtest_output
    return junit


def test_default_still_builds_all_models_and_runs_ctest(build_tree):
    root, _, env = build_tree
    stale = root / "source/tests/infer/stale.pt2"
    stale.touch()
    result, commands = run_script(build_tree)
    assert result.returncode == 0, result.stderr
    assert "BUILD_TESTING:BOOL=TRUE" in commands[0]["args"]
    assert [c["name"] for c in commands[:4]] == ["cmake", "cmake", "cmake", "python"]
    assert commands[1]["args"][:2] == ["--build", "."]
    assert commands[2]["args"] == ["--install", "."]
    check_ctest_report(root, commands[-1])
    assert set(generated_models(commands)) == MODELS
    assert len(generated_models(commands)) == len(MODELS) + 1
    assert generated_models(commands).count("gen_spin.py") == 2
    reports = generator_reports(root)
    assert {path.stem for path in reports} == REPORTS
    assert len(reports) == len(REPORTS)
    assert {row["exit_status"] for row in reports.values()} == {"0"}
    pair_exclude = next(c for c in commands if "--reuse-dpa1" in c["args"])
    assert Path(pair_exclude["args"][1]).name == "gen_dpa1_pairexcl.py"
    assert pair_exclude["dpa1_ready"]
    spin_reuse = next(c for c in commands if "--reuse-model-devi" in c["args"])
    assert spin_reuse["dpa3_ready"] and spin_reuse["model0_ready"]
    dpa3 = next(c for c in commands if "--dpa3-only" in c["args"])
    model_devi = next(
        c for c in commands if "gen_spin_model_devi.py" in generated_models([c])
    )
    assert commands.index(dpa3) < commands.index(spin_reuse)
    assert commands.index(model_devi) < commands.index(spin_reuse)
    assert all("--compressed-only" not in c["args"] for c in commands)
    assert not stale.exists()
    assert (
        Path(env["MOCK_SHARED_LIB_DIR"]) / "libdeepmd_op_pt.so"
    ).read_text() == "custom op from this build"


@pytest.mark.parametrize("variant", ["cpu", "cuda"])
def test_long_generators_refill_completed_slots_before_dpa2_finishes(
    build_tree, variant
):
    root, _, env = build_tree
    result, commands = run_script(
        build_tree, DP_VARIANT=variant, MOCK_CHECK_SCHEDULE="1"
    )
    assert result.returncode == 0, result.stderr
    events = [
        json.loads(line)
        for line in Path(env["MOCK_SCHEDULE_LOG"]).read_text().splitlines()
    ]

    def position(event, model):
        return events.index({"event": event, "model": model})

    assert position("end", "gen_sea.py") < position("start", "gen_dpa4.py")
    assert position("end", "gen_dpa1.py") < position("start", "gen_dpa3.py")
    for model in ("gen_dpa3.py", "gen_dpa4.py"):
        assert position("end", model) < position("end", "gen_dpa2.py")
    first_phase = {
        "gen_sea.py",
        "gen_dpa1.py",
        "gen_dpa2.py",
        "gen_dpa3.py",
        "gen_dpa4.py",
    }
    active = set()
    peak = 0
    for entry in events:
        if entry["model"] not in first_phase:
            continue
        if entry["event"] == "start":
            active.add(entry["model"])
        else:
            active.remove(entry["model"])
        peak = max(peak, len(active))
    assert not active
    assert peak == 3
    for model in ("gen_fparam_aparam.py", "gen_model_devi.py", "gen_chg_spin.py"):
        assert position("end", "gen_dpa2.py") < position("start", model)
    assert position("end", "gen_dpa1.py") < position("start", "gen_dpa1_pairexcl.py")
    assert position("end", "gen_dpa1_pairexcl.py") < position("start", "gen_spin.py")
    for model in first_phase:
        assert generated_models(commands).count(model) == 1
    check_ctest_report(root, commands[-1])


@pytest.mark.parametrize("pytorch", ["TRUE", "FALSE"])
def test_build_only_installs_without_generating_models_or_running_ctest(
    build_tree, pytorch
):
    root, _, env = build_tree
    stale = root / "source/tests/infer/keep.pt2"
    stale.touch()
    result, commands = run_script(build_tree, "--build-only", ENABLE_PYTORCH=pytorch)
    assert result.returncode == 0, result.stderr
    assert "BUILD_TESTING:BOOL=FALSE" in commands[0]["args"]
    assert not generated_models(commands)
    assert all(c["name"] != "ctest" for c in commands)
    assert stale.exists()
    assert not (root / "source/build_tests/generator-reports").exists()
    assert not (root / "source/build_tests/gtest-reports").exists()
    assert (Path(env["MOCK_SHARED_LIB_DIR"]) / "libdeepmd_op_pt.so").exists() == (
        pytorch == "TRUE"
    )


def test_kokkos_models_reuse_build_and_preserve_other_fixtures(build_tree):
    root, _, _ = build_tree
    infer = root / "source/tests/infer"
    keep = infer / "other_model.pt2"
    keep.touch()
    stale = [
        infer / f"deeppot_dpa4c_long_range_compressed.{ext}"
        for ext in ("pt2", "pte", "expected")
    ]
    for path in stale:
        path.touch()
    result, commands = run_script(build_tree, "--models-only=kokkos")
    assert result.returncode == 0, result.stderr
    assert [c["name"] for c in commands] == ["python", "python"]
    assert commands[0]["args"][0] == "-c"
    assert generated_models(commands) == ["gen_dpa4c_long_range.py"]
    assert commands[1]["args"][0] == "-u"
    assert commands[1]["args"][2:] == ["--compressed-only"]
    reports = generator_reports(root)
    assert len(reports) == 1
    assert {path.stem for path in reports} == {"gen_dpa4c_long_range"}
    assert {row["exit_status"] for row in reports.values()} == {"0"}
    assert not (root / "source/build_tests/gtest-reports").exists()
    assert (
        Path(commands[1]["ld"].split(os.pathsep)[0]).resolve() == root / "dp_test/lib"
    )
    assert keep.exists()
    assert not any(path.exists() for path in stale)


@pytest.mark.parametrize(
    "override", [{"DP_VARIANT": "cpu"}, {"ENABLE_PYTORCH": "FALSE"}]
)
def test_kokkos_models_reject_missing_cuda_or_pytorch(build_tree, override):
    result, commands = run_script(build_tree, "--models-only=kokkos", **override)
    assert result.returncode != 0
    assert not commands


@pytest.mark.parametrize(
    "missing",
    ["dp_test/lib/libdeepmd_c.so", "source/build_tests/op/pt/libdeepmd_op_pt.so"],
)
def test_kokkos_models_require_the_existing_build(build_tree, missing):
    root, _, _ = build_tree
    (root / missing).unlink()
    result, commands = run_script(build_tree, "--models-only=kokkos")
    assert result.returncode != 0
    assert not commands


@pytest.mark.parametrize(
    "model",
    [
        "gen_sea.py",
        "gen_dpa1.py",
        "gen_dpa2.py",
        "gen_dpa3.py",
        "gen_dpa4.py",
        "gen_dpa1_pairexcl.py",
        "gen_dpa4c_long_range.py",
    ],
)
def test_default_propagates_generator_failures(build_tree, model):
    root, _, _ = build_tree
    result, commands = run_script(build_tree, MOCK_FAIL=model)
    assert result.returncode == 17
    assert all(c["name"] != "ctest" for c in commands)
    failed = [
        row for path, row in generator_reports(root).items() if path.stem == model[:-3]
    ]
    assert len(failed) == 1
    assert failed[0]["exit_status"] == "17"
    if model == "gen_dpa1.py":
        assert all("--reuse-dpa1" not in c["args"] for c in commands)
    if model in {"gen_dpa2.py", "gen_dpa3.py", "gen_dpa4.py"}:
        assert "gen_fparam_aparam.py" not in generated_models(commands)
        assert "gen_corrupt_with_comm.py" not in generated_models(commands)


def test_kokkos_models_propagate_generator_failure(build_tree):
    root, _, _ = build_tree
    result, _ = run_script(
        build_tree, "--models-only=kokkos", MOCK_FAIL="gen_dpa4c_long_range.py"
    )
    assert result.returncode == 17
    reports = generator_reports(root)
    assert len(reports) == 1
    assert {row["exit_status"] for row in reports.values()} == {"17"}


@pytest.mark.parametrize(
    "failure,report",
    [
        ("--dpa3-only", "gen_spin_dpa3"),
        ("gen_spin_model_devi.py", "gen_spin_model_devi"),
        ("--reuse-model-devi", "gen_spin_reuse"),
    ],
)
def test_spin_reuse_propagates_each_phase_failure(build_tree, failure, report):
    root, _, _ = build_tree
    result, commands = run_script(build_tree, MOCK_FAIL=failure)
    assert result.returncode == 17
    assert all(c["name"] != "ctest" for c in commands)
    reports = {path.stem: row for path, row in generator_reports(root).items()}
    assert reports[report]["exit_status"] == "17"
    if failure != "--reuse-model-devi":
        assert all("--reuse-model-devi" not in c["args"] for c in commands)


def test_each_invocation_keeps_separate_generator_and_ctest_reports(build_tree):
    root, _, _ = build_tree
    first, commands = run_script(build_tree)
    assert first.returncode == 0, first.stderr
    first_junit = check_ctest_report(root, commands[-1])
    first_reports = generator_reports(root)
    second, commands = run_script(build_tree)
    assert second.returncode == 0, second.stderr
    second_junit = check_ctest_report(root, commands[-1])
    assert first_junit != second_junit
    assert first_junit.is_file()
    reports = generator_reports(root)
    assert len(reports) == 2 * len(REPORTS)
    assert len({path.parent for path in reports}) == 2
    assert {path: reports[path] for path in first_reports} == first_reports


@pytest.mark.parametrize(
    "gtest_output", ["json:/custom/reports/", "xml:/custom/result.xml"]
)
def test_ctest_preserves_explicit_gtest_output(build_tree, gtest_output):
    root, _, _ = build_tree
    result, commands = run_script(
        build_tree, ENABLE_PYTORCH="FALSE", GTEST_OUTPUT=gtest_output
    )
    assert result.returncode == 0, result.stderr
    check_ctest_report(root, commands[-1], gtest_output)
    assert not generator_reports(root)


@pytest.mark.parametrize(
    "args", [("--unknown",), ("--build-only", "--models-only=kokkos")]
)
def test_invalid_arguments_fail_before_build(build_tree, args):
    result, commands = run_script(build_tree, *args)
    assert result.returncode == 2
    assert not commands
