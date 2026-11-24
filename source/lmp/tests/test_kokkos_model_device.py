# SPDX-License-Identifier: LGPL-3.0-or-later
"""Exercise a nonzero Kokkos CUDA device from MPI rank zero."""

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("style", "filename"),
    [
        ("deepmd/kk", "deeppot_dpa4c_canonical.pt2"),
        ("dpa4spin/kk", "deeppot_dpa4c_spin_lmp_kokkos.pt2"),
    ],
)
def test_explicit_cuda_device(tmp_path: Path, style: str, filename: str) -> None:
    if os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") != "1":
        pytest.skip("requires the matching Kokkos LAMMPS build")
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")
    executable = shutil.which("lmp")
    infer = Path(__file__).resolve().parents[2] / "tests/infer"
    model = infer / filename
    assert executable, "Kokkos LAMMPS executable is required"
    assert model.is_file(), model
    with zipfile.ZipFile(model) as archive:
        species = json.loads(archive.read("model/extra/metadata.json"))["type_map"][0]
    spin = style == "dpa4spin/kk"
    energies = []
    forces = []
    for device in (0, 1):
        script = tmp_path / "in.lmp"
        dump = tmp_path / f"forces-{device}"
        script.write_text(
            f"units metal\natom_style {'spin' if spin else 'atomic'}\n"
            "atom_modify map yes\nregion box block 0 20 0 20 0 20 units box\n"
            "create_box 1 box\ncreate_atoms 1 single 6 6 6 units box\n"
            "create_atoms 1 single 7.25 6.1 6 units box\nmass * 12\n"
            + ("set atom * spin/atom 1.2 0 0 1\n" if spin else "")
            + f"pair_style {style} {model}\npair_coeff * * {species}\n"
            'run 0\nprint "ENERGY $(pe:%.17g)"\n'
            f"write_dump all custom {dump} id fx fy fz modify sort id format float %.17g\n"
        )
        run = subprocess.run(
            [
                executable,
                "-k",
                "on",
                "g",
                "1",
                "d",
                str(device),
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "newton",
                "on",
                "neigh",
                "half",
                "gpu/aware",
                "off",
                "-log",
                "none",
                "-in",
                str(script),
            ],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stdout
        assert f"to gpu {device}" in run.stdout, run.stdout
        energies.append(
            float(
                next(
                    line.split()[1]
                    for line in run.stdout.splitlines()
                    if line.startswith("ENERGY ")
                )
            )
        )
        forces.append(np.loadtxt(dump, skiprows=9)[:, 1:])
    assert np.max(np.abs(forces[0])) > 0
    np.testing.assert_allclose(energies[1], energies[0], atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(forces[1], forces[0], atol=1e-5, rtol=1e-5)
