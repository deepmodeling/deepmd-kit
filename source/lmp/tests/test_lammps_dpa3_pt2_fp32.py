# SPDX-License-Identifier: LGPL-3.0-or-later
"""Float32 multi-rank LAMMPS test for DPA3 GNN .pt2.

The float64 multi-rank test in ``test_lammps_dpa3_pt2.py`` validates the
comm_dict path against a same-archive single-rank reference (atol 1e-8).
This file does the same thing for the float32 variant of the fixture
(``deeppot_dpa3_mpi_fp32.pt2``) — the model and trace are byte-identical
in every respect except ``descriptor.precision``/``fitting_net.precision``
being set to ``float32``.

Why a separate test file:
    1. The fp32 fixture is not packaged into ``deeppot_dpa3_mpi.pt2``;
       it is a sibling artifact produced by the same gen script.
    2. fp32 needs looser tolerances. The C++ ``border_op`` kernel's
       ``forward_t<float>`` template path (chosen automatically via
       ``g1.dtype()`` dispatch in ``source/op/pt/comm.cc``) loses ~7
       decimal digits of precision relative to the ``forward_t<double>``
       path. Single-precision GEMM in the AOTI-compiled kernel adds
       further drift.

What this file validates that the float64 test does not:
    * ``border_op`` template dispatch on ``g1.dtype() == kFloat`` (vs
      ``kDouble``) actually fires under MPI.
    * ``register_fake`` returns ``torch.empty_like(g1)`` so the FX trace
      preserves float32 dtype through the opaque op.
    * ``register_autograd``'s ``border_op_backward`` invocation also
      runs under float32, returning float32 gradients.
    * MPI exchange uses ``MPI_FLOAT`` (vs ``MPI_DOUBLE``), halving the
      bandwidth per ghost atom — relevant for slow interconnects.

This is a regression-only test for the comm path. It does not pin any
hardcoded numerical values; mpi-2 must agree with same-archive mpi-1
within float32 tolerances.
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

pb_file_mpi_fp32 = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "infer"
    / "deeppot_dpa3_mpi_fp32.pt2"
)
data_file = Path(__file__).parent / "data_dpa3_pt2_fp32.lmp"

# Same 6-atom O-H system as the float64 test. ``processors 2 1 1``
# splits at x=6.5 -> 3 atoms per rank.
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
        pytest.skip("Skip test because PyTorch support is not enabled.")
    write_lmp_data(box, coord, type_OH, data_file)


def teardown_module() -> None:
    if data_file.exists():
        os.remove(data_file)


def _run_mpi_subprocess(
    nprocs: int,
    processors: str | None = None,
) -> dict:
    """Run ``run_mpi_pair_deepmd_dpa3_pt2.py`` against the fp32 archive.

    Returns ``{"pe", "forces", "virials"}`` parsed from the runner's
    output file.
    """
    with tempfile.NamedTemporaryFile(mode="r", suffix=".out", delete=False) as f:
        out_path = f.name
    try:
        argv = [
            "mpirun",
            "-n",
            str(nprocs),
            sys.executable,
            str(Path(__file__).parent / "run_mpi_pair_deepmd_dpa3_pt2.py"),
            str(data_file.resolve()),
            str(pb_file_mpi_fp32.resolve()),
            out_path,
        ]
        if processors is not None:
            argv.extend(["--processors", processors])
        elif nprocs == 1:
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
def test_pair_deepmd_mpi_dpa3_fp32() -> None:
    """Float32 DPA3 multi-rank must match same-archive single-rank.

    Tolerances follow standard float32 expectations:
    * energy: ``rel=1e-5``  (~7 decimal digits, with mantissa noise)
    * force:  ``atol=1e-4`` absolute (force magnitudes are O(1e-1) for
                                       this system, so ``rel=1e-3``)
    * virial: ``atol=5e-4`` per component

    Single-rank uses the regular artifact (nswap=0); multi-rank uses
    the with-comm artifact -- so any divergence beyond float32 noise
    is necessarily in the multi-rank dispatch (border_op template
    dispatch, MPI_FLOAT exchange, register_fake/register_autograd
    dtype handling).
    """
    out_mpi = _run_mpi_subprocess(nprocs=2)
    out_ref = _run_mpi_subprocess(nprocs=1)

    assert out_mpi["pe"] == pytest.approx(out_ref["pe"], rel=1e-5, abs=1e-7)
    np.testing.assert_allclose(
        out_mpi["forces"], out_ref["forces"], atol=1e-4, rtol=1e-3
    )
    np.testing.assert_allclose(
        out_mpi["virials"], out_ref["virials"], atol=5e-4, rtol=1e-3
    )
