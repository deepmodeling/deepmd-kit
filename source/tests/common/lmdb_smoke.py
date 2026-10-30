# SPDX-License-Identifier: LGPL-3.0-or-later
"""Real legacy-LMDB fixtures shared by backend entrypoint smoke tests."""

from pathlib import (
    Path,
)
from typing import (
    Any,
)

import lmdb
import msgpack
import numpy as np

from deepmd.utils.data import (
    DataRequirementItem,
)


def lmdb_training_config(root: Path) -> dict[str, Any]:
    """Create two equal-size frames with different periodicity for startup."""
    path = root / "frames.lmdb"

    def encode(value):
        array = np.asarray(value)
        return {
            "type": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tobytes(),
        }

    with (
        lmdb.open(str(path), map_size=1024 * 1024) as env,
        env.begin(write=True) as txn,
    ):
        txn.put(
            b"__metadata__",
            msgpack.packb(
                {
                    "nframes": 2,
                    "frame_idx_fmt": "012d",
                    "frame_nlocs": [2, 2],
                    "type_map": ["H"],
                    "system_info": {"natoms": [2], "nframes": 2},
                },
                use_bin_type=True,
            ),
        )
        for index, cell in enumerate((8.0 * np.eye(3), np.zeros((3, 3)))):
            txn.put(
                f"{index:012d}".encode(),
                msgpack.packb(
                    {
                        "atom_types": encode(np.array([0, 0], dtype=np.int64)),
                        "coords": encode(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])),
                        "cells": encode(cell),
                        "energies": encode(np.array([1.0])),
                        "forces": encode(np.zeros((2, 3))),
                    },
                    use_bin_type=True,
                ),
            )
    return {
        "model": {
            "type_map": ["H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": "auto",
                "rcut": 2.0,
                "rcut_smth": 1.5,
                "neuron": [2, 4],
                "axis_neuron": 2,
            },
            "fitting_net": {"neuron": [4]},
        },
        "training": {"training_data": {"systems": str(path), "batch_size": 2}},
    }


def check_legacy_lmdb_data(data) -> None:
    """Exercise batch, placeholder, statistics and full-validation surfaces."""
    data.add_data_requirements(
        [
            DataRequirementItem("energy", 1, atomic=False, must=True, high_prec=True),
        ]
    )
    description = data.get_data_dict()
    assert description["coord"]["atomic"]
    assert description["box"]["ndof"] == 9
    assert description["energy"]["high_prec"]
    assert data.get_batch_size() == [2]
    assert len(data.system_dirs) == len(data.data_systems) == 2
    assert {system.pbc for system in data.data_systems} == {False, True}
    for system in data.data_systems:
        assert system.get_natoms() == 2
        assert system.dirs
        for directory in system.dirs:
            batch = system._load_set(directory)
            assert batch["coord"].shape == (1, 6)
            assert bool(batch["find_box"]) == system.pbc
        test = system.get_test()
        assert test["energy"].shape == (1, 1)
        assert bool(test["find_box"]) == system.pbc
    batches = [data.get_batch() for _ in range(2)]
    assert {len(batch["default_mesh"]) for batch in batches} == {1, 7}
    for batch in batches:
        np.testing.assert_array_equal(batch["natoms_vec"], [2, 2, 2])
        np.testing.assert_array_equal(batch["real_natoms_vec"], [[2, 2, 2]])
