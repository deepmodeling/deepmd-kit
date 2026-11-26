# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model deviation of the pair styles that serve compact canonical models.

An ensemble of a compressed DPA4C model and a DPA4 model, whose graph forward
alone reads the source CSR and, with its further message-passing blocks, the
halo exchange, runs through ``deepmd`` and ``deepmd/kk`` (energy models) and
``dpa4spin`` and ``dpa4spin/kk`` (native-spin models), on one rank and on
two. The deviation the pair style writes must match the one computed from the
outputs of each model alone, and the first model must drive the dynamics.
Over six steps with a neighbor rebuild every third step and an atom crossing
the boundary of the ranks, a run of one step, and one more after a reset of
the step and a displacement that changes the neighbors of an atom, the
deviation written every second step must match the one written every step:
the further models then see the steps of a rebuild only when the deviation is
due, and a new run or a reset of the step must not hand them a list they hold
from before. On four
ranks, one of which holds no atom, the deviation must match the one computed
from the models alone as well.

Run source/install/test_lammps_kokkos.sh, which sets DPA4SPIN_KOKKOS_REQUIRED=1,
on a CUDA host with the canonical fixtures of gen_dpa4c_canonical.py.
"""

from __future__ import (
    annotations,
)

import json
import os
import signal
import subprocess
import sys
from pathlib import (
    Path,
)

import numpy as np
import pytest
from expected_ref import (
    read_expected_ref,
)
from write_lmp_data import (
    write_lmp_data,
    write_lmp_data_spin,
)

_INFER = Path(__file__).resolve().parents[2] / "tests" / "infer"
# Both models of a kind are conditioned on a charge/spin state, and each
# evaluates its default.
_MODELS = {
    "energy": ("deeppot_dpa4c_canonical", "deeppot_dpa4_canonical"),
    "spin": ("deeppot_dpa4c_canonical_spin", "deeppot_dpa4_canonical_spin"),
}
_STYLES = {
    "deepmd": "energy",
    "deepmd/kk": "energy",
    "dpa4spin": "spin",
    "dpa4spin/kk": "spin",
}


def _write_data(kind: str, path: Path, vacuum: float = 0.0) -> None:
    """Write the lattice of the fixtures with one atom about to change rank.

    ``vacuum`` widens the box along x beyond the lattice, in Angstrom.
    """
    system = read_expected_ref(_INFER / f"{_MODELS[kind][1]}.expected")["system"]
    cell = system["box"].reshape(3, 3)
    box = np.array([0, cell[0, 0] + vacuum, 0, cell[1, 1], 0, cell[2, 2], 0, 0, 0])
    coord = system["coord"].reshape(-1, 3).copy()
    types = system["atype"].astype(int) + 1
    # Atom 10 crosses the x-domain boundary at step 2, before the rebuild of
    # step 3, so the ranks exchange it while no deviation is due.
    coord[9, 0] = cell[0, 0] / 2 - 0.1
    if kind == "spin":
        write_lmp_data_spin(box, coord, system["spin"].reshape(-1, 3), types, path)
    else:
        write_lmp_data(box, coord, types, path)
    with path.open("a") as stream:
        stream.write("\nVelocities\n\n")
        for atom in range(1, len(coord) + 1):
            stream.write(f"{atom} {150.0 if atom == 10 else 0.0} 0 0\n")


def _run(
    style: str, ranks: int, tmp_path: Path, vacuum: float = 0.0, point: bool = False
) -> dict:
    kind = _STYLES[style]
    models = [_INFER / f"{name}.pt2" for name in _MODELS[kind]]
    for model in models:
        assert model.is_file(), f"Generate the canonical fixture first: {model}"
    data = tmp_path / "lattice.lmp"
    _write_data(kind, data, vacuum)
    output = tmp_path / "result.json"
    command = [
        os.environ.get("DPA4SPIN_KOKKOS_MPIEXEC", "mpirun"),
        "-n",
        str(ranks),
        sys.executable,
        str(Path(__file__).with_name("run_mpi_model_devi_canonical.py")),
        str(data),
        str(output),
        "--style",
        style,
        "--models",
        *map(str, models),
        *(["--point"] if point else []),
    ]
    # Bound the whole MPI process group: killing only mpirun can leave the
    # ranks holding the pipes open after a timeout.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=900)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            pytest.fail(f"MPI runner timed out:\n{stdout}{stderr}")
        assert process.returncode == 0, stdout + stderr
    return json.loads(output.read_text())


def _deviation_row(single: list[dict], magnetic: bool) -> np.ndarray:
    """Statistics and per-atom force deviation of the outputs of the models.

    The deviation of a vector is the root mean square of its distance from
    the model average; that of the virial per atom is taken per component.
    """

    def statistics(values: np.ndarray, rms: bool = False) -> list[float]:
        average = np.sqrt(np.mean(values**2)) if rms else np.mean(values)
        return [np.max(values), np.min(values), average]

    def vector_deviation(name: str) -> np.ndarray:
        outputs = np.array([model[name] for model in single])
        return np.linalg.norm(np.std(outputs, axis=0), axis=1)

    force = vector_deviation("force")
    virial = np.array([model["virial"] for model in single]) / len(force)
    row = [*statistics(np.std(virial, axis=0), rms=True), *statistics(force)]
    if magnetic:
        row += statistics(vector_deviation("magnetic_force"))
    return np.array(row + force.tolist())


def _check_single_point(result: dict, style: str) -> None:
    """The deviation of the ensemble matches that of the models alone."""
    single = result["single"]
    expected = _deviation_row(single, _STYLES[style] == "spin")
    assert np.max(expected[3:6]) > 1e-3, "the fixtures must disagree"
    ensemble = result["ensemble"]
    rows = np.array(ensemble["rows"])
    assert rows.shape == (1, 1 + len(expected))
    assert rows[0, 0] == 0
    # The canonical kernels run in float32 and the per-atom virial of the
    # single models sums along another path than the virial of the ensemble.
    np.testing.assert_allclose(rows[0, 1:], expected, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        ensemble["force"], single[0]["force"], rtol=1e-5, atol=1e-6
    )


def _require_kokkos_regression() -> None:
    if os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") != "1":
        pytest.skip("Enable the dedicated CUDA Kokkos plugin regression explicitly")


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("style", list(_STYLES))
def test_model_deviation(style: str, ranks: int, tmp_path: Path) -> None:
    _require_kokkos_regression()
    result = _run(style, ranks, tmp_path)
    _check_single_point(result, style)

    # The steps of the six-step run, of the one-step run that follows, and of
    # the one-step run after the reset of the step to one; every second step
    # writes the rows of the even steps.
    every_step = np.array(result["every_step"]["rows"])
    every_second = np.array(result["every_second_step"]["rows"])
    np.testing.assert_array_equal(every_step[:, 0], [*range(7), 6, 7, 1, 2])
    np.testing.assert_array_equal(every_second[:, 0], [0, 2, 4, 6, 6, 2])
    np.testing.assert_allclose(
        every_second, every_step[[0, 2, 4, 6, 7, 10]], rtol=1e-4, atol=1e-6
    )
    counts = result["every_step"]["local_counts"]
    assert all(min(step) > 0 for step in counts)
    if ranks == 2:
        # Atom 10 changed rank at the rebuild of step 3.
        assert counts[1] == [counts[0][0] - 1, counts[0][1] + 1]


@pytest.mark.parametrize("style", list(_STYLES))
def test_model_deviation_with_an_empty_rank(style: str, tmp_path: Path) -> None:
    """A rank without local or ghost atoms takes part in the deviation."""
    _require_kokkos_regression()
    # Four ranks along x over the lattice and 27 Angstrom of vacuum: the third
    # rank lies farther than the cutoff and the skin from every atom.
    result = _run(style, 4, tmp_path, vacuum=27.0, point=True)
    assert min(result["ensemble"]["atom_counts"]) == 0
    _check_single_point(result, style)
