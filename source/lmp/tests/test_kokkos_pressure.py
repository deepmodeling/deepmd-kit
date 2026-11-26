# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest


def test_kokkos_pressure(tmp_path):
    if os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") != "1":
        pytest.skip("requires CUDA Kokkos LAMMPS")
    lmp = shutil.which("lmp")
    assert lmp
    model = (
        Path(__file__).resolve().parents[2] / "tests/infer/deeppot_dpa4c_canonical.pt2"
    )
    with zipfile.ZipFile(model) as archive:
        species = json.loads(archive.read("model/extra/metadata.json"))["type_map"][0]

    def pressure(style, newton, per_atom):
        args = [lmp]
        if style.endswith("/kk"):
            args += [
                "-k",
                "on",
                "g",
                "1",
                "-sf",
                "kk",
                "-pk",
                "kokkos",
                "newton",
                newton,
                "neigh",
                "half" if newton == "on" else "full",
                "gpu/aware",
                "off",
            ]
        script = tmp_path / "pressure.in"
        script.write_text(
            f"units metal\natom_style atomic\natom_modify map yes\nnewton {newton}\n"
            "region box block 0 20 0 20 0 20 units box\ncreate_box 1 box\n"
            "create_atoms 1 single 6 6 6 units box\n"
            "create_atoms 1 single 7.25 6.1 6.4 units box\n"
            "create_atoms 1 single 6.4 7.5 6.2 units box\nmass * 12\n"
            f"velocity all set 0 0 0\npair_style {style} {model}\npair_coeff * * {species}\n"
            + (
                "compute pa all centroid/stress/atom NULL virial\ncompute s all reduce sum c_pa[1]\n"
                if per_atom
                else ""
            )
            + "thermo_style custom step pxx pyy pzz pxy pxz pyz"
            + (" c_s" if per_atom else "")
            + "\nthermo_modify format float %.16g\nrun 0\n"
        )
        run = subprocess.run(
            [*args, "-log", "none", "-in", str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        header = next(
            i
            for i, line in enumerate(run.stdout.splitlines())
            if line.split()[:2] == ["Step", "Pxx"]
        )
        return np.asarray(run.stdout.splitlines()[header + 1].split()[1:7], dtype=float)

    reference = pressure("deepmd", "on", False)
    assert np.all(np.abs(reference) > 1e-6), reference
    for newton in ("on", "off"):
        for per_atom in (False, True):
            np.testing.assert_allclose(
                pressure("deepmd/kk", newton, per_atom),
                reference,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"{newton=}, {per_atom=}",
            )
