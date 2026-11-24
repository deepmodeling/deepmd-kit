# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import shutil
import subprocess
from pathlib import (
    Path,
)

import pytest


@pytest.mark.parametrize(
    ("style", "model_name"),
    [
        ("deepmd", "deeppot_dpa4_graph.pt2"),
        ("deepspin", "deeppot_dpa4_spin_graph.pt2"),
    ],
)
@pytest.mark.parametrize("comm", ["brick", "tiled"])
def test_two_rank_graph_communication(tmp_path, style, model_name, comm):
    model = Path(__file__).resolve().parents[2] / "tests/infer" / model_name
    lmp = shutil.which("lmp")
    mpi = shutil.which("mpirun")
    if not (model.is_file() and lmp and mpi):
        pytest.skip("requires the DPA4 graph fixtures, mpirun and LAMMPS")
    is_spin = style == "deepspin"
    commands = [
        "units metal",
        f"comm_style {comm}",
        "processors 2 1 1",
        f"atom_style {'spin' if is_spin else 'atomic'}",
        "atom_modify map yes",
        "region cell block 0 13 0 13 0 13 units box",
        "create_box 2 cell",
        "create_atoms 1 single 6 6 6 units box",
        "create_atoms 1 single 7 6 6 units box",
        "mass * 12",
    ]
    if is_spin:
        commands.append("set atom * spin/atom 1.2 0 0 1")
    commands.extend([f'pair_style {style} "{model}"', "pair_coeff * *", "run 0"])
    infile = tmp_path / "in.lammps"
    infile.write_text("\n".join(commands) + "\n")
    result = subprocess.run(
        [mpi, "-np", "2", lmp, "-in", str(infile)],
        cwd=tmp_path,
        env={
            **os.environ,
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    if comm == "brick":
        assert result.returncode == 0, result.stdout
        assert "Loop time" in result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert "use comm_style brick" in result.stdout
