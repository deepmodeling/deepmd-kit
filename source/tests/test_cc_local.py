# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exercise the C++ CI entry point without compiling backend libraries."""

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
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
record = {{"name": name, "args": args, "ld": os.environ.get("LD_LIBRARY_PATH", "")}}
with open(os.environ["MOCK_LOG"], "a") as stream:
    stream.write(json.dumps(record) + "\\n")
if os.environ.get("MOCK_FAIL") and any(os.environ["MOCK_FAIL"] == Path(a).name for a in [name, *args]):
    sys.exit(17)
if name == "python" and args[0] == "-c":
    sys.exit(subprocess.call([sys.executable, "-S", *args]))
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
        "MOCK_LOG": str(tmp_path / "commands.jsonl"),
        "MOCK_SHARED_LIB_DIR": str(tmp_path / "installed_ops"),
        "MOCK_FAIL": "",
    }
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
        Path(c["args"][0]).name
        for c in commands
        if c["name"] == "python" and c["args"][0] != "-c"
    ]


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
    assert commands[-1]["name"] == "ctest"
    assert commands[-1]["args"] == ["--output-on-failure"]
    assert set(generated_models(commands)) == MODELS
    assert len(generated_models(commands)) == len(MODELS)
    assert not stale.exists()
    assert (
        Path(env["MOCK_SHARED_LIB_DIR"]) / "libdeepmd_op_pt.so"
    ).read_text() == "custom op from this build"


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


@pytest.mark.parametrize("model", ["gen_sea.py", "gen_dpa4c_long_range.py"])
def test_default_propagates_generator_failures(build_tree, model):
    result, commands = run_script(build_tree, MOCK_FAIL=model)
    assert result.returncode != 0
    assert all(c["name"] != "ctest" for c in commands)


def test_kokkos_models_propagate_generator_failure(build_tree):
    result, _ = run_script(
        build_tree, "--models-only=kokkos", MOCK_FAIL="gen_dpa4c_long_range.py"
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    "args", [("--unknown",), ("--build-only", "--models-only=kokkos")]
)
def test_invalid_arguments_fail_before_build(build_tree, args):
    result, commands = run_script(build_tree, *args)
    assert result.returncode == 2
    assert not commands
