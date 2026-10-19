# SPDX-License-Identifier: LGPL-3.0-or-later
"""Actual CUDA /kk convention and MPI regression, separate from host FD tests.

Run source/install/test_lammps_kokkos.sh after test_cc_local.sh on a CUDA host.
That job supplies a compressed canonical fixture and a matching Kokkos LAMMPS
library/plugin, and sets DPA4SPIN_KOKKOS_REQUIRED=1: no missing prerequisite may
silently skip it. This checks adapter equivalence, not GPU energy differences.
"""

from __future__ import (
    annotations,
)

import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import zipfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
from write_lmp_data import (
    write_lmp_data_spin,
)

HBAR_METAL = 6.5821191e-04
_MODEL = (
    Path(__file__).resolve().parents[2]
    / "tests/infer/deeppot_dpa4c_spin_lmp_kokkos.pt2"
)
_BOX = np.array([0.0, 14.0, 0.0, 14.0, 0.0, 14.0, 0.0, 0.0, 0.0])
# Both ranks own atoms and interact across the x=7 domain boundary. The box
# exceeds twice the cutoff, as required by the single-rank folded /kk graph.
_COORD = np.array(
    [
        [5.5, 4.0, 4.0],
        [7.7, 4.4, 4.1],
        [5.8, 4.8, 4.0],
        [4.9, 4.2, 4.6],
        [8.1, 5.0, 4.3],
        [7.9, 3.7, 4.7],
    ]
)
_SPIN = np.array(
    [
        [0.0, 0.0, 0.0],
        [-0.07, 0.09, 0.03],
        [0.02, -0.06, 0.08],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
_TYPE = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)


def _unavailable(reason):
    if os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def kokkos_model():
    # The ordinary LAMMPS suite also runs on CUDA, but uses a non-Kokkos wheel.
    # Only the dedicated matching library/plugin job opts into this regression.
    if os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") != "1":
        pytest.skip("Enable the dedicated CUDA Kokkos plugin regression explicitly")
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        _unavailable(f"CUDA PyTorch unavailable: {exc}")
        raise
    if not torch.cuda.is_available():
        _unavailable("CUDA device required for the actual dpa4spin/kk adapter")
    for module in ("lammps", "mpi4py"):
        if importlib.util.find_spec(module) is None:
            _unavailable(f"Required Python package missing: {module}")
    if shutil.which(os.environ.get("DPA4SPIN_KOKKOS_MPIEXEC", "mpirun")) is None:
        _unavailable("mpirun required for the two-rank /kk regression")
    model = Path(os.environ.get("DPA4SPIN_KOKKOS_MODEL", _MODEL))
    if not model.is_file():
        _unavailable(f"Generate the compressed canonical fixture first: {model}")
    with zipfile.ZipFile(model) as archive:
        meta = json.loads(archive.read("model/extra/metadata.json"))
    assert meta["lower_input_kind"] == "dpa4c_canonical"
    assert meta["canonical_index_dtype"] == "uint32"
    assert meta["graph_edge_dtype"] == "float32"
    assert meta["spin_scheme"] == "native"
    assert meta["type_map"] == ["Ni", "O"]
    assert meta["use_spin"] == [True, False]
    for key in ("dim_fparam", "dim_aparam", "dim_chg_spin"):
        assert meta[key] == 0
    assert not meta.get("has_comm_artifact", False)
    return model


def _run(tmp_path, data, model, *, ranks, kokkos):
    output = tmp_path / f"{'kokkos' if kokkos else 'host'}-{ranks}.json"
    runner = Path(__file__).with_name("run_mpi_dpa4spin_conventions.py")
    command = [
        os.environ.get("DPA4SPIN_KOKKOS_MPIEXEC", "mpirun"),
        "-n",
        str(ranks),
        sys.executable,
        str(runner),
        str(data),
        str(model),
        str(output),
        "--ranks",
        str(ranks),
    ]
    if kokkos:
        command.append("--kokkos")
    # As in dpa4_spin_harness, bound the whole MPI process group: killing only
    # mpirun can leave workers holding the pipes open after a timeout.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=600)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The process group already exited between timeout and cleanup.
                pass
            stdout, stderr = process.communicate()
            pytest.fail(
                f"MPI runner timed out (process group killed):\n{stdout}{stderr}"
            )
        assert process.returncode == 0, stdout + stderr
    return json.loads(output.read_text())


@pytest.fixture(scope="module", params=[0.0, 1e-8, 0.2], ids=["zero", "tiny", "finite"])
def samples(kokkos_model, tmp_path_factory, request):
    directory = tmp_path_factory.mktemp(f"dpa4spin-kk-{request.param}")
    data = directory / "nio.lmp"
    spin = _SPIN.copy()
    spin[0] = request.param * np.array([2.0, -1.0, 2.0]) / 3.0
    write_lmp_data_spin(_BOX, _COORD, spin, _TYPE, data)
    return (
        _run(directory, data, kokkos_model, ranks=1, kokkos=False),
        _run(directory, data, kokkos_model, ranks=1, kokkos=True),
        _run(directory, data, kokkos_model, ranks=2, kokkos=True),
    )


def _assert_snapshot(actual, expected, scale=1.0):
    assert actual["ids"] == expected["ids"] == list(range(1, 7))
    np.testing.assert_allclose(actual["spin"], expected["spin"], atol=0, rtol=0)
    # The canonical kernels use float32; edge/reduction ordering differs between
    # host, folded single-rank and extended multi-rank graphs. A snapshot holds
    # the per-atom virial only when its run requested it.
    fields = [("force", 2e-7), ("magnetic_force", 5e-9)]
    if "virial" in expected:
        fields.append(("virial", 2e-7))
    for name, atol in fields:
        np.testing.assert_allclose(
            actual[name], scale * np.array(expected[name]), rtol=5e-4, atol=atol
        )
    assert actual["energy"] == pytest.approx(
        scale * expected["energy"], rel=5e-6, abs=2e-7
    )


def test_kokkos_conventions_and_mpi(samples):
    host, single, parallel = samples
    reference = host["cartesian"]["snapshots"][0]
    assert np.linalg.norm(np.array(reference["magnetic_force"])[0]) > 1e-7
    for ranks, result in ((1, single), (2, parallel)):
        cartesian = result["cartesian"]["snapshots"][0]
        magnetic = np.array(cartesian["magnetic_force"])
        # In particular, forcing the GPU kernel to precession at S=0 fails here.
        assert np.linalg.norm(magnetic[0]) > 1e-7
        np.testing.assert_array_equal(magnetic[3:], 0.0)
        _assert_snapshot(cartesian, reference)
        for selection in ("default", "precession", "cartesian"):
            sample = result[selection]
            assert sample["convention"] == int(selection == "cartesian")
            assert sample["pair_style"] == "dpa4spin/kk"
            assert sample["atom_style"] == "spin/kk"
            assert sample["run_style_active"] == 1
            assert sample["kokkos_active"] == sample["kokkos_ngpus"] == 1
            assert sample["newton_pair"] == 0
            assert len(sample["local_counts"]) == ranks
            assert sum(sample["local_counts"]) == 6
            assert min(sample["local_counts"]) > 0
            if ranks == 2:
                assert min(sample["ghost_counts"]) > 0
            actual = sample["snapshots"][0]
            _assert_snapshot(actual, host[selection]["snapshots"][0])
            _assert_snapshot(actual, single[selection]["snapshots"][0])
            if selection != "cartesian":
                moments = np.array(actual["spin"])[:, 3]
                finite = moments > 0
                precession = np.array(actual["magnetic_force"])
                # Normalize BEFORE comparing so the tiny-moment case cannot
                # pass with zero force hidden by a fixed absolute tolerance.
                np.testing.assert_allclose(
                    precession[finite] * (HBAR_METAL / moments[finite, None]),
                    magnetic[finite],
                    rtol=5e-4,
                    atol=5e-9,
                )
                np.testing.assert_array_equal(precession[~finite], 0.0)
            for repeated in sample["snapshots"][1:]:
                _assert_snapshot(repeated, actual)
            virial_snapshots = sample["virial_snapshots"]
            first = virial_snapshots[0]
            _assert_snapshot(first, host[selection]["virial_snapshots"][0])
            _assert_snapshot(first, single[selection]["virial_snapshots"][0])
            # Requesting the per-atom virial changes no force or energy.
            _assert_snapshot(first, actual)
            _assert_snapshot(virial_snapshots[1], first)
            _assert_snapshot(sample["scaled"], first, scale=2.75)


def test_kokkos_overlay_accumulates(samples):
    host, single, parallel = samples
    for result in (single, parallel):
        for selection in ("precession", "cartesian"):
            overlay = result[f"overlay_{selection}"]
            assert overlay["pair_style"] == "hybrid/overlay/kk"
            assert overlay["run_style_active"] == 1
            assert overlay["convention"] == int(selection == "cartesian")
            for key in ("snapshots", "virial_snapshots"):
                for snapshot in overlay[key]:
                    _assert_snapshot(snapshot, host[selection][key][0], scale=2.0)
                    _assert_snapshot(snapshot, result[selection][key][0], scale=2.0)
