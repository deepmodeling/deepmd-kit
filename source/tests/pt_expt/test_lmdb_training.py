# SPDX-License-Identifier: LGPL-3.0-or-later
"""LMDB-format dataset support for the pt_expt backend.

Covers:

- :class:`LmdbDataSystem.get_batch` returns numpy arrays in the shape that
  :func:`normalize_batch` consumes.
- ``get_trainer()`` routes an LMDB ``systems`` path through
  :class:`LmdbDataSystem` and runs a few training steps.
"""

import os
import shutil
import tempfile
import unittest

import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.pt_expt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt_expt.utils.lmdb_dataset import (
    LmdbDataSystem,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)


def _encode_array(arr: np.ndarray) -> dict:
    return {
        "nd": None,
        "type": str(arr.dtype),
        "kind": "",
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def _make_frame(natoms: int, seed: int) -> dict:
    """Synthetic LMDB frame matching the on-disk schema used by LmdbDataReader."""
    rng = np.random.RandomState(seed)
    half = natoms // 2
    return {
        "atom_numbs": [half, natoms - half],
        "atom_names": ["O", "H"],
        "atom_types": _encode_array(
            np.array([0] * half + [1] * (natoms - half), dtype=np.int64)
        ),
        "orig": _encode_array(np.zeros(3, dtype=np.float64)),
        "cells": _encode_array((np.eye(3) * 10.0).astype(np.float64)),
        "coords": _encode_array((rng.rand(natoms, 3) * 10.0).astype(np.float64)),
        "energies": _encode_array(np.array(rng.randn(), dtype=np.float64)),
        "forces": _encode_array(rng.randn(natoms, 3).astype(np.float64)),
    }


def _create_test_lmdb(path: str, nframes: int, natoms: int) -> None:
    """Write a minimal LMDB containing *nframes* frames of *natoms* atoms each."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": fmt,
        "system_info": {
            "formula": f"O{natoms // 2}H{natoms - natoms // 2}",
            "natoms": [natoms // 2, natoms - natoms // 2],
            "nframes": nframes,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for i in range(nframes):
            key = format(i, fmt).encode()
            txn.put(key, msgpack.packb(_make_frame(natoms, i), use_bin_type=True))
    env.close()


class TestLmdbDataSystemGetBatch(unittest.TestCase):
    """LmdbDataSystem.get_batch produces a numpy dict that normalize_batch accepts."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.tmpdir, "test.lmdb")
        _create_test_lmdb(self.lmdb_path, nframes=8, natoms=6)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_get_batch_shape_and_normalize(self) -> None:
        ds = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
        )
        batch = ds.get_batch()
        # Required structural keys.
        for key in ("coord", "atype", "force", "energy", "natoms"):
            self.assertIn(key, batch, f"missing {key}")
        # NumPy arrays (not torch tensors) — pt_expt converts at the trainer
        # boundary.
        self.assertIsInstance(batch["coord"], np.ndarray)
        self.assertIsInstance(batch["atype"], np.ndarray)
        self.assertEqual(batch["coord"].shape, (2, 6, 3))
        self.assertEqual(batch["atype"].shape, (2, 6))
        self.assertEqual(batch["natoms"].shape, (2, 4))  # nloc, nloc, n_O, n_H

        # normalize_batch must accept the dict and produce input/label splits
        # without raising.
        norm = normalize_batch(batch)
        inputs, labels = split_batch(norm)
        self.assertIn("coord", inputs)
        self.assertIn("atype", inputs)
        self.assertIn("force", labels)
        self.assertIn("natoms", labels)

    def test_get_batch_iterates_past_end(self) -> None:
        """get_batch reseeds the sampler at the end of an epoch."""
        ds = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
        )
        # 8 frames / batch 2 = 4 batches per epoch; pull more than that.
        for _ in range(10):
            batch = ds.get_batch()
            self.assertEqual(batch["coord"].shape, (2, 6, 3))

    def test_add_data_requirements_passthrough(self) -> None:
        from deepmd.utils.data import (
            DataRequirementItem,
        )

        ds = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=1,
            seed=0,
        )
        ds.add_data_requirements(
            [
                DataRequirementItem(
                    "energy", ndof=1, atomic=False, must=False, high_prec=True
                ),
            ]
        )
        batch = ds.get_batch()
        self.assertIn("energy", batch)
        self.assertIn("find_energy", batch)


class TestLmdbTrainingLoop(unittest.TestCase):
    """End-to-end: get_trainer routes an LMDB path and runs training steps."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.lmdb_path = os.path.join(cls.tmpdir, "train.lmdb")
        cls.val_lmdb_path = os.path.join(cls.tmpdir, "val.lmdb")
        _create_test_lmdb(cls.lmdb_path, nframes=8, natoms=6)
        _create_test_lmdb(cls.val_lmdb_path, nframes=4, natoms=6)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _make_lmdb_config(self, numb_steps: int = 3) -> dict:
        return {
            "model": {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [6, 12],
                    "rcut_smth": 0.50,
                    "rcut": 3.00,
                    "neuron": [8, 16],
                    "resnet_dt": False,
                    "axis_neuron": 4,
                    "type_one_side": True,
                    "seed": 1,
                },
                "fitting_net": {
                    "neuron": [16, 16],
                    "resnet_dt": True,
                    "seed": 1,
                },
                "data_stat_nbatch": 1,
            },
            "learning_rate": {
                "type": "exp",
                "decay_steps": 500,
                "start_lr": 0.001,
                "stop_lr": 3.51e-8,
            },
            "loss": {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
            "training": {
                "training_data": {
                    "systems": self.lmdb_path,
                    "batch_size": 1,
                },
                "validation_data": {
                    "systems": self.val_lmdb_path,
                    "batch_size": 1,
                    "numb_btch": 1,
                },
                "numb_steps": numb_steps,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": numb_steps,
                "save_freq": numb_steps,
            },
        }

    def test_get_trainer_routes_lmdb(self) -> None:
        config = self._make_lmdb_config(numb_steps=3)
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            trainer = get_trainer(config)
            self.assertIsInstance(trainer.training_data, LmdbDataSystem)
            trainer.run()
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
