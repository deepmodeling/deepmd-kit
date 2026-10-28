# SPDX-License-Identifier: LGPL-3.0-or-later
"""Kokkos model deviation on the official six-atom, small DPA4C fixture.

Requires a CUDA Kokkos LAMMPS (``DEEPMD_TEST_LAMMPS`` or ``lmp`` on PATH).
``DEEPMD_TEST_PLUGIN`` optionally names its DeePMD plugin. Archives are
generated in pytest's temporary directory, or explicitly supplied through
``DEEPMD_KOKKOS_TEST_MODELS`` after running ``dpa4c_model_devi_fixture.py export``.
CPU/non-Kokkos installations skip before compiling any models. PyTorch and
LAMMPS always execute in separate subprocesses.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import zipfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
from write_lmp_data import (
    write_lmp_data,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "dpa4c_model_devi_fixture.py"
# Same PBC six-atom water geometry as test_lammps_dpa4_graph_pt2.py.
COORD = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
BOX = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
TYPES = np.array([1, 2, 2, 1, 2, 2])


def _checked_run(argv, *, env, cwd, timeout=180, input=None):
    result = subprocess.run(
        argv,
        input=input,
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip("PyTorch support is disabled")
    configured = os.environ.get("DEEPMD_TEST_LAMMPS")
    executable = shutil.which(configured or "lmp")
    if executable is None:
        if configured:
            pytest.fail(f"DEEPMD_TEST_LAMMPS is not executable: {configured}")
        pytest.skip("LAMMPS executable is not available")
    directory = tmp_path_factory.mktemp("dpa4c_kokkos_capabilities")
    env = os.environ.copy()
    env.update(
        DP_CUDA_INFER="2",
        OMP_NUM_THREADS="1",
        DP_INTRA_OP_PARALLELISM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="1",
    )
    # Missing hardware/optional packages is a skip; broken imports are errors.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys\n"
            "if importlib.util.find_spec('torch') is None: sys.exit(77)\n"
            "import torch\n"
            "if not torch.cuda.is_available(): sys.exit(77)\n"
            "import deepmd.pt.cxx_op\n"
            "from deepmd.pt_expt.utils import env\n"
            "from deepmd.pt_expt.kernels.dpa4c.graph_compress import op_available\n"
            "sys.exit(0 if env.DEVICE.type == 'cuda' and op_available() else 77)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if probe.returncode == 77:
        pytest.skip("CUDA and the compiled DPA4C operator are required")
    assert probe.returncode == 0, probe.stdout + probe.stderr
    help_result = _checked_run([executable, "-h"], env=env, cwd=directory)
    if "KOKKOS" not in help_result.stdout:
        pytest.skip("LAMMPS was built without Kokkos")
    command = [
        executable,
        "-k",
        "on",
        "g",
        "1",
        "-sf",
        "kk",
        "-pk",
        "kokkos",
        "neigh",
        "full",
        "newton",
        "off",
        "gpu/aware",
        "off",
        "-log",
        "none",
        "-in",
        "/dev/stdin",
    ]
    plugin = env.get("DEEPMD_TEST_PLUGIN")
    prefix = f'plugin load "{plugin}"\n' if plugin else ""
    styles = subprocess.run(
        command,
        input=prefix + "info styles pair\n",
        text=True,
        capture_output=True,
        env=env,
        cwd=directory,
        timeout=120,
        check=False,
    )
    output = styles.stdout + styles.stderr
    if styles.returncode != 0 and "without GPU support" in output:
        pytest.skip("LAMMPS Kokkos was built without GPU support")
    assert styles.returncode == 0, output
    if "deepmd/kk" not in output:
        pytest.skip("The loaded DeePMD plugin has no deepmd/kk pair style")
    return command, env, prefix


@pytest.fixture(scope="module")
def models(runtime, tmp_path_factory):
    _, env, _ = runtime
    configured = os.environ.get("DEEPMD_KOKKOS_TEST_MODELS")
    if configured:
        directory = Path(configured).resolve()
    else:
        directory = tmp_path_factory.mktemp("dpa4c_canonical_models")
        _checked_run(
            [sys.executable, str(FIXTURE), "export", str(directory)],
            env=env,
            cwd=directory,
            timeout=1200,
        )
    result = {}
    for name in ("m0", "m1", "other_state"):
        path = directory / f"{name}.pt2"
        assert path.is_file(), f"Required generated fixture is missing: {path}"
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("model/extra/metadata.json"))
        assert metadata["lower_input_kind"] == "dpa4c_canonical"
        assert metadata["graph_edge_dtype"] == "float32"
        result[name] = path
    return result


def _archive_variant(
    source,
    destination,
    *,
    no_fold=False,
    atomic_virial=None,
    unknown_condition=False,
    omit_atomic_output=False,
):
    # Like gen_corrupt_with_comm.py: derive a temporary archive without
    # rewriting or uploading a compiled model. The original stays untouched.
    with (
        zipfile.ZipFile(source) as original,
        zipfile.ZipFile(destination, "w") as target,
    ):
        for info in original.infolist():
            if no_fold and info.filename == "model/extra/charge_state.pt2":
                continue
            data = original.read(info.filename)
            if info.filename == "model/extra/metadata.json":
                metadata = json.loads(data)
                if no_fold:
                    metadata.pop("charge_state_constants")
                    assert metadata["dim_chg_spin"] == 0
                    assert metadata["default_chg_spin"]
                if atomic_virial is not None:
                    metadata["do_atomic_virial"] = atomic_virial
                if unknown_condition:
                    assert no_fold
                    for key in (
                        "has_chg_spin_ebd",
                        "has_default_chg_spin",
                        "default_chg_spin",
                        "chg_spin_table_ranges",
                    ):
                        metadata.pop(key, None)
                if omit_atomic_output:
                    assert metadata["do_atomic_virial"] is True
                    metadata["output_keys"].remove("atom_virial")
                data = json.dumps(metadata).encode()
            target.writestr(info, data)
    return destination


def _run_lammps(
    directory,
    runtime,
    selected,
    *,
    atomic=False,
    out_freq=5,
    rebuild=5,
    nprocs=1,
    probe=None,
    expect_error=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / "water.lmp"
    write_lmp_data(BOX, COORD, TYPES, data)
    command, base_env, prefix = runtime
    env = base_env.copy()
    if probe:
        env["KOKKOS_TOOLS_LIBS"] = str(probe)
        env["DEEPMD_KOKKOS_PROBE_OUTPUT"] = str(directory / "profile.tsv")
    options = " atomic" if atomic else ""
    names = " ".join(f'"{path}"' for path in selected)
    script = (
        prefix
        + f"""units metal
boundary p p p
atom_style atomic
atom_modify map array sort 0 0.0
neighbor 2.0 bin
neigh_modify every {rebuild} delay 0 check no
read_data "{data}"
mass 1 16
mass 2 2
pair_style deepmd/kk {names} out_file deviation.out out_freq {out_freq}{options}
pair_coeff * *
timestep 0.0005
velocity all set 0.1 0.02 -0.03
fix integration all nve
thermo 1
thermo_style custom step pe
thermo_modify format float %.16e
dump trajectory all custom 1 trajectory.dump id type x y z fx fy fz
dump_modify trajectory sort id format float %.16e
run 10
"""
    )
    (directory / "input.lammps").write_text(script)
    argv = list(command)
    if nprocs != 1:
        launcher = shutil.which("mpirun")
        if launcher is None:
            pytest.skip("mpirun is unavailable")
        argv = [launcher, "-np", str(nprocs), *argv]
    result = subprocess.run(
        argv,
        input=script,
        text=True,
        capture_output=True,
        cwd=directory,
        env=env,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    (directory / "lammps.out").write_text(output)
    if expect_error:
        assert result.returncode != 0, output
        assert "ERROR" in output and expect_error in output, output
        assert "Loop time" not in output, "invalid ensemble started dynamics"
        return None
    assert result.returncode == 0, output
    assert "Loop time" in output
    frames = {}
    lines = (directory / "trajectory.dump").read_text().splitlines()
    cursor = 0
    while cursor < len(lines):
        assert lines[cursor] == "ITEM: TIMESTEP"
        step = int(lines[cursor + 1])
        natoms = int(lines[cursor + 3])
        assert natoms == 6
        assert lines[cursor + 8] == "ITEM: ATOMS id type x y z fx fy fz"
        table = np.array(
            [
                list(map(float, line.split()))
                for line in lines[cursor + 9 : cursor + 9 + natoms]
            ]
        )
        np.testing.assert_array_equal(table[:, 0], np.arange(1, 7))
        frames[step] = table
        cursor += 9 + natoms
    assert list(frames) == list(range(11))
    energies = {}
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+([-+\d.eE]+)\s*", line)
        if match:
            energies[int(match[1])] = float(match[2])
    assert set(energies) == set(range(11)), output
    deviation_path = directory / "deviation.out"
    rows = (
        [
            line
            for line in deviation_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if deviation_path.exists()
        else []
    )
    deviations = (
        np.loadtxt(rows, ndmin=2) if rows else np.empty((0, 13 if atomic else 7))
    )
    return frames, energies, deviations


def _oracle(directory, runtime, selected, frames):
    request = directory / "oracle_request.json"
    result = directory / "oracle_result.json"
    request.write_text(
        json.dumps(
            {
                "models": [str(path) for path in selected],
                "coords": [table[:, 2:5].tolist() for table in frames.values()],
            }
        )
    )
    _checked_run(
        [sys.executable, str(FIXTURE), "oracle", str(request), str(result)],
        env=runtime[1],
        cwd=directory,
    )
    outputs = json.loads(result.read_text())
    return {key: np.array([model[key] for model in outputs]) for key in outputs[0]}


def _assert_oracle(frames, energies, deviations, oracle, *, atomic, out_freq):
    force = oracle["force"]
    assert np.all(np.isfinite(force))
    assert np.max(np.abs(force)) > 1e-7, "fixture must have nonzero forces"
    np.testing.assert_allclose(
        [table[:, 5:8] for table in frames.values()],
        force[0],
        rtol=3e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        list(energies.values()), oracle["energy"][0], rtol=3e-4, atol=3e-5
    )
    steps = list(range(0, 11, out_freq)) if out_freq else []
    assert deviations.shape == (len(steps), 13 if atomic else 7)
    if not steps:
        return
    np.testing.assert_array_equal(deviations[:, 0], steps)
    std_force = np.linalg.norm(np.std(force[:, steps], axis=0, ddof=0), axis=2)
    std_virial = np.std(oracle["virial"][:, steps], axis=0, ddof=0) / 6
    expected = np.column_stack(
        (
            std_virial.max(axis=1),
            std_virial.min(axis=1),
            np.sqrt(np.mean(std_virial**2, axis=1)),
            std_force.max(axis=1),
            std_force.min(axis=1),
            std_force.mean(axis=1),
        )
    )
    np.testing.assert_allclose(deviations[:, 1:7], expected, rtol=5e-4, atol=3e-5)
    if atomic:
        np.testing.assert_allclose(deviations[:, 7:], std_force, rtol=5e-4, atol=3e-5)


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("order", [("m0", "m1"), ("m1", "m0"), ("m0", "m0")])
def test_deviation_matches_python_oracle(runtime, models, tmp_path, atomic, order):
    selected = [models[name] for name in order]
    frames, energies, deviations = _run_lammps(
        tmp_path, runtime, selected, atomic=atomic
    )
    oracle = _oracle(tmp_path, runtime, selected, frames)
    _assert_oracle(frames, energies, deviations, oracle, atomic=atomic, out_freq=5)
    if order[0] == order[1]:
        np.testing.assert_array_equal(deviations[:, 1:], 0.0)
    else:
        assert np.max(np.abs(oracle["force"][0] - oracle["force"][1])) > 1e-5
        assert np.max(deviations[:, 4]) > 1e-5


@pytest.mark.parametrize("rebuild", [1, 5])
def test_sampling_disabled_preserves_driver(runtime, models, tmp_path, rebuild):
    selected = [models["m0"], models["m1"]]
    frames, energies, deviations = _run_lammps(
        tmp_path, runtime, selected, out_freq=0, rebuild=rebuild
    )
    oracle = _oracle(tmp_path, runtime, selected, frames)
    _assert_oracle(frames, energies, deviations, oracle, atomic=False, out_freq=0)


@pytest.fixture(scope="module")
def profiling_library(runtime, tmp_path_factory):
    directory = tmp_path_factory.mktemp("dpa4c_kokkos_tools")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("C++ compiler is required for the Kokkos Tools observer")
    library = directory / "model_devi_probe.so"
    _checked_run(
        [
            *compiler,
            "-std=c++11",
            "-shared",
            "-fPIC",
            str(HERE / "kokkos_model_devi_probe.cpp"),
            "-o",
            str(library),
        ],
        env=runtime[1],
        cwd=directory,
    )
    return library


@pytest.mark.parametrize("out_freq", [0, 5])
def test_reference_reuses_driver_graph(
    runtime, models, profiling_library, tmp_path, out_freq
):
    _run_lammps(
        tmp_path,
        runtime,
        [models["m0"], models["m1"]],
        out_freq=out_freq,
        probe=profiling_library,
    )
    counts = {}
    for line in (tmp_path / "profile.tsv").read_text().splitlines():
        name, count = line.split("\t")
        counts[name] = int(count)
    assert counts["deepmd/kk:driver_inference"] == 11
    assert counts["compact_canonical:count"] == 11
    assert counts.get("deepmd/kk:reference_inference", 0) == (3 if out_freq else 0)
    assert counts.get("deepmd/kk:model_deviation", 0) == (3 if out_freq else 0)


def test_multiple_mpi_ranks_rejected(runtime, models, tmp_path):
    _run_lammps(
        tmp_path,
        runtime,
        [models["m0"], models["m1"]],
        nprocs=2,
        expect_error="requires one MPI rank",
    )


@pytest.mark.parametrize("no_fold", [False, True])
def test_different_frozen_conditions_rejected(runtime, models, tmp_path, no_fold):
    selected = [models["m0"], models["other_state"]]
    if no_fold:
        selected = [
            _archive_variant(path, tmp_path / f"legacy_{i}.pt2", no_fold=True)
            for i, path in enumerate(selected)
        ]
    _run_lammps(tmp_path, runtime, selected, expect_error="charge/spin")


def test_equal_legacy_frozen_conditions(runtime, models, tmp_path):
    selected = [
        _archive_variant(models[name], tmp_path / f"legacy_{name}.pt2", no_fold=True)
        for name in ("m0", "m1")
    ]
    frames, energies, deviations = _run_lammps(tmp_path, runtime, selected, atomic=True)
    # The modern archives contain exactly the same frozen inference constants.
    oracle = _oracle(tmp_path, runtime, [models["m0"], models["m1"]], frames)
    _assert_oracle(frames, energies, deviations, oracle, atomic=True, out_freq=5)


@pytest.mark.parametrize("unsupported_index", [0, 1])
def test_unknown_frozen_condition_rejected(
    runtime, models, tmp_path, unsupported_index
):
    selected = [models["m0"], models["m1"]]
    selected[unsupported_index] = _archive_variant(
        selected[unsupported_index],
        tmp_path / "unknown_frozen_condition.pt2",
        no_fold=True,
        unknown_condition=True,
    )
    _run_lammps(
        tmp_path,
        runtime,
        selected,
        expect_error="default charge/spin state cannot be established",
    )


@pytest.mark.parametrize("unsupported_index", [0, 1])
def test_atomic_virial_required_for_every_model(
    runtime, models, tmp_path, unsupported_index
):
    selected = [models["m0"], models["m1"]]
    selected[unsupported_index] = _archive_variant(
        selected[unsupported_index],
        tmp_path / "no_atomic_virial.pt2",
        atomic_virial=False,
    )
    _run_lammps(tmp_path, runtime, selected, expect_error="atomic virial")


@pytest.mark.parametrize("unsupported_index", [0, 1])
def test_atomic_virial_output_key_required(
    runtime, models, tmp_path, unsupported_index
):
    selected = [models["m0"], models["m1"]]
    selected[unsupported_index] = _archive_variant(
        selected[unsupported_index],
        tmp_path / "missing_atomic_virial_output.pt2",
        omit_atomic_output=True,
    )
    _run_lammps(tmp_path, runtime, selected, expect_error="atomic virial")


@pytest.mark.parametrize("no_fold", [False, True])
@pytest.mark.parametrize("atomic_virial", [False, True])
def test_c_api_metadata_buffer_contract(
    runtime, models, tmp_path, no_fold, atomic_virial
):
    configured = os.environ.get("DEEPMD_TEST_C_API_LIBRARY")
    if not configured:
        pytest.skip("Set DEEPMD_TEST_C_API_LIBRARY to exercise the C API library")
    library = Path(configured).resolve()
    assert library.is_file(), f"C API library is missing: {library}"
    model = _archive_variant(
        models["other_state"],
        tmp_path / "c_api_model.pt2",
        no_fold=no_fold,
        atomic_virial=atomic_virial,
    )
    _checked_run(
        [
            sys.executable,
            str(FIXTURE),
            "capi",
            str(library),
            str(model),
            "[2.0, 3.0]",
            json.dumps(atomic_virial),
        ],
        env=runtime[1],
        cwd=tmp_path,
    )
