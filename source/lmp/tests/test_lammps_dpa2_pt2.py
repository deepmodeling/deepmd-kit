# SPDX-License-Identifier: LGPL-3.0-or-later
"""Multi-rank LAMMPS test for DPA2 .pt2 (extends GNN MPI Phase 5 to DPA2).

DPA2's repformer block participates in the per-layer ghost-atom MPI
exchange just like DPA3's repflows; the with-comm AOTInductor artifact
is produced automatically by ``deepmd/pt_expt/utils/serialization.py``
because ``_has_message_passing`` returns True for any DPA2 model.

Unlike DPA3 (which has ``use_loc_mapping``), DPA2's repformer always
takes a ``mapping`` tensor, so a single ``deeppot_dpa2.pt2`` already
carries the dual-artifact layout — no separate ``_mpi.pt2`` needed.

This file targets the gap "DPA2 multi-rank dispatch never tested
end-to-end" recorded in
``memory/gnn_mpi_untested_paths.md::Dispatch wired, no test fixture``.
The reference is a same-archive single-rank run (``mpirun -n 1``
through the same dual-artifact ``.pt2``); no hardcoded reference
values are needed.
"""

from __future__ import (
    annotations,
)

import importlib.util
import os
import shutil
import subprocess as sp
import sys
import tempfile
from pathlib import (
    Path,
)

import numpy as np
import pytest
from write_lmp_data import (
    write_lmp_data,
)

# Reuses the same generic mpirun driver as the DPA3 multi-rank tests —
# the script is descriptor-agnostic (just LAMMPS + pair_style deepmd).
RUNNER_PATH = Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"

pb_file = Path(__file__).parent.parent.parent / "tests" / "infer" / "deeppot_dpa2.pt2"
data_file = Path(__file__).parent / "data_dpa2_pt2.lmp"

box = np.array([0, 13, 0, 13, 0, 13, 0, 0, 0])
coord = np.array(
    [
        [12.83, 2.56, 2.18],
        [12.09, 2.87, 2.74],
        [0.25, 3.32, 1.68],
        [3.36, 3.00, 1.81],
        [3.51, 2.51, 2.60],
        [4.27, 3.22, 1.56],
    ]
)
type_OH = np.array([1, 2, 2, 1, 2, 2])


def setup_module() -> None:
    if os.environ.get("ENABLE_PYTORCH", "1") != "1":
        pytest.skip(
            "Skip test because PyTorch support is not enabled.",
        )
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _run_mpi_subprocess(nprocs: int = 2) -> dict:
    """Invoke the generic mpirun driver and parse the output.

    With ``nprocs == 2`` (default) the runner forces ``processors 2 1 1``
    so ``DeepPotPTExpt`` routes to the with-comm artifact. With
    ``nprocs == 1`` the runner uses ``processors 1 1 1`` and the C++
    side falls back to the regular artifact — useful as a same-archive
    reference for value comparison.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        out_path = f.name
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(RUNNER_PATH),
            str(data_file.resolve()),
            str(pb_file.resolve()),
            out_path,
        ]
        if nprocs == 1:
            argv.extend(["--processors", "1 1 1"])
        sp.check_call(argv)
        with open(out_path) as fh:
            lines = fh.read().strip().splitlines()
        pe = float(lines[0])
        rows = np.array(
            [list(map(float, line.split())) for line in lines[1:]],
            dtype=np.float64,
        )
        forces = rows[:, :3]
        virials = rows[:, 3:]
        return {"pe": pe, "forces": forces, "virials": virials}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


@pytest.mark.skipif(
    shutil.which("mpirun") is None, reason="MPI is not installed on this system"
)
@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None, reason="mpi4py is not installed"
)
def test_pair_deepmd_mpi_dpa2() -> None:
    """Multi-rank DPA2 .pt2 dispatch must match the same-archive
    single-rank reference for energy, forces, and virial.

    Verifies that:
    - ``DeepPotPTExpt::compute`` correctly routes to the with-comm
      artifact for DPA2 (descriptor-agnostic dispatch).
    - The pt_expt ``DescrptBlockRepformers._exchange_ghosts`` override
      drives ``deepmd_export::border_op`` for repformer's per-layer
      ghost exchange (the path equivalent to DPA3's repflows).
    - Different ``model_nnei`` from DPA3 (DPA2 repformer has nsel=15
      vs DPA3's e_sel=30) — exercises the dynamic-nnei with-comm
      trace at a different baked-in value.

    No hardcoded reference; compares against a same-archive single-rank
    run (``mpirun -n 1`` + ``processors 1 1 1`` falls back to the
    regular artifact).
    """
    out_mpi = _run_mpi_subprocess(nprocs=2)
    out_ref = _run_mpi_subprocess(nprocs=1)
    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(out_mpi["forces"], out_ref["forces"], atol=1e-8, rtol=0)
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=1e-8, rtol=0
    )
