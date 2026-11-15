# SPDX-License-Identifier: LGPL-3.0-or-later
"""Kokkos halo exchange for an active two-block canonical energy model."""

import json
import os
import zipfile
from pathlib import (
    Path,
)

import constants
import numpy as np
import pytest
from expected_ref import (
    read_expected_ref,
)
from lammps_test_utils import (
    run_mpi_pair_runner,
)
from write_lmp_data import (
    write_lmp_data,
)


def test_multiblock_halo_survives_neighbor_rebuild(tmp_path: Path) -> None:
    """Owned and ghost rows agree with one rank across rebuild and reuse."""
    if os.environ.get("DPA4SPIN_KOKKOS_REQUIRED") != "1":
        pytest.skip("Enable the dedicated CUDA Kokkos plugin regression explicitly")
    infer = Path(__file__).resolve().parents[2] / "tests/infer"
    model = infer / "deeppot_dpa4c_canonical_multiblock.pt2"
    assert model.is_file(), f"Generate the multiblock canonical fixture: {model}"
    with zipfile.ZipFile(model) as archive:
        metadata = json.loads(archive.read("model/extra/metadata.json"))
    assert metadata["lower_input_kind"] == "dpa4c_canonical"
    assert metadata["has_message_passing"]
    assert not metadata["has_comm_artifact"]
    system = read_expected_ref(model.with_suffix(".expected"))["system"]
    cell = system["box"].reshape(3, 3)
    box = np.array([0, cell[0, 0], 0, cell[1, 1], 0, cell[2, 2], 0, 0, 0])
    coord = system["coord"].reshape(-1, 3).copy()
    # Atom 10 crosses the x-domain boundary before the step-2 rebuild,
    # changing the owned prefix and invalidating the previous halo plan.
    coord[9, 0] = cell[0, 0] / 2 - 0.1
    data = tmp_path / "lattice.lmp"
    write_lmp_data(box, coord, system["atype"].astype(int) + 1, data)
    with data.open("a") as stream:
        stream.write("\nVelocities\n\n")
        for atom in range(1, len(coord) + 1):
            stream.write(f"{atom} {150.0 if atom == 10 else 0.0} 0 0\n")
    runner = Path(__file__).with_name("run_mpi_pair_deepmd_dpa3_pt2.py")
    single, parallel = (
        run_mpi_pair_runner(
            runner,
            data,
            model,
            nprocs=ranks,
            processors=f"{ranks} 1 1",
            runner_args=[
                "--kokkos",
                "--nsteps",
                "3",
                "--neigh-every",
                "2",
                "--check-halo-cadence",
            ],
            timeout=600,
        )
        for ranks in (1, 2)
    )
    assert parallel["pe"] == pytest.approx(single["pe"], rel=1e-5, abs=1e-5)
    np.testing.assert_allclose(
        parallel["forces"], single["forces"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        parallel["virials"] / constants.nktv2p,
        single["virials"] / constants.nktv2p,
        rtol=1e-4,
        atol=1e-5,
    )
