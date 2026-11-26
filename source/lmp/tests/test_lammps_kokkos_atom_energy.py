# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-atom energy tallies across LAMMPS atom-capacity growth."""

import os
import subprocess
import sys
from pathlib import (
    Path,
)

import numpy as np
import pytest

_INFER = Path(__file__).resolve().parents[2] / "tests" / "infer"


@pytest.mark.skipif(
    os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") != "1",
    reason="Requires the CUDA Kokkos plugin and canonical fixtures",
)
@pytest.mark.parametrize("spin", [False, True], ids=["deepmd", "dpa4spin"])
def test_per_atom_energy_survives_capacity_growth(spin: bool) -> None:
    style = "dpa4spin/kk" if spin else "deepmd/kk"
    model = _INFER / f"deeppot_dpa4_canonical{'_spin' if spin else ''}.pt2"
    # Keep Kokkos initialization/finalization and native allocator errors local
    # to this simulation.
    subprocess.run(
        [sys.executable, __file__, style, str(model)], check=True, timeout=180
    )


def _check_growth(style: str, model: Path) -> None:
    from lammps import (
        LMP_STYLE_ATOM,
        LMP_TYPE_ARRAY,
        LMP_TYPE_VECTOR,
        lammps,
    )

    spin = style == "dpa4spin/kk"
    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-k", "on", "g", "1"])
    try:
        lmp.commands_list(
            [
                "package kokkos neigh full newton off",
                f"units {'real' if spin else 'metal'}",
                f"atom_style {'spin' if spin else 'atomic'}/kk",
                "atom_modify map yes",
                "region box block 0 20 0 20 0 20 units box",
                "create_box 2 box",
                "create_atoms 1 single 4 4 4 units box",
                "create_atoms 2 single 5 4 4 units box",
                "create_atoms 1 single 4 5 4 units box",
                "mass * 1",
                "run_style verlet/kk",
                f"pair_style {style} {model}",
                "pair_coeff * * Ni NULL" if spin else "pair_coeff * *",
                "compute eat all pe/atom pair",
                "compute esum all reduce sum c_eat",
                "compute centroid all centroid/stress/atom NULL pair",
                "compute vsum all reduce sum c_centroid[1]",
                "thermo_style custom step pe c_esum c_vsum",
                "thermo_modify norm no",
            ]
        )

        def tallies() -> np.ndarray:
            lmp.command("run 0")
            energy = lmp.numpy.extract_compute("eat", LMP_STYLE_ATOM, LMP_TYPE_VECTOR)
            virial = lmp.numpy.extract_compute(
                "centroid", LMP_STYLE_ATOM, LMP_TYPE_ARRAY
            )
            ids = lmp.numpy.extract_atom("id")[:3]
            rows = np.column_stack((energy[:3], virial[:3]))[np.argsort(ids)]
            assert np.max(np.abs(rows[:, 0])) > 1e-8
            np.testing.assert_allclose(
                rows[:, 0].sum(), lmp.get_thermo("pe"), rtol=1e-5
            )
            if spin:
                np.testing.assert_array_equal(rows[1], 0.0)
            return rows

        before = tallies()
        capacity = lmp.extract_setting("nmax")
        # Grow the backing arrays while keeping the evaluated system unchanged.
        lmp.command("group transient empty")
        lmp.command(
            f"create_atoms 1 random {capacity} 74123 box units box group transient"
        )
        assert lmp.extract_setting("nmax") > capacity
        lmp.command("delete_atoms group transient compress no")
        assert lmp.get_natoms() == 3
        np.testing.assert_allclose(tallies(), before, rtol=1e-5, atol=1e-7)
    finally:
        lmp.finalize()


if __name__ == "__main__":
    _check_growth(sys.argv[1], Path(sys.argv[2]))
