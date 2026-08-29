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
from unittest.mock import (
    patch,
)

import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.utils import lmdb_data as lmdb_data_module
from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.dpmodel.utils.lmdb_data import (
    collate_lmdb_frames,
)
from deepmd.pt_expt.entrypoints.main import (
    _build_data_system,
    _get_neighbor_stat_data,
    get_trainer,
)
from deepmd.pt_expt.loss import (
    EnergyLoss,
)
from deepmd.pt_expt.train.training import (
    Trainer,
)
from deepmd.pt_expt.utils.lmdb_dataset import (
    LmdbDataSystem,
)
from deepmd.pt_expt.utils.stat import (
    make_stat_input,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

from .compile_utils import (
    REQUIRES_SUPPORTED_COMPILE,
)


def _encode_array(arr: np.ndarray) -> dict:
    return {
        "nd": None,
        "type": str(arr.dtype),
        "kind": "",
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


class TestConvertedLmdbValidation(unittest.TestCase):
    """Reject format conversion that resolves to multiple LMDB databases."""

    def test_neighbor_stat_and_training_data_reject_multiple_lmdb(self) -> None:
        params = {
            "systems": "input.extxyz",
            "format": "extxyz",
            "batch_size": 1,
        }
        converted = ["first.lmdb", "second.lmdb"]
        with (
            patch(
                "deepmd.pt_expt.entrypoints.main.process_systems",
                return_value=converted,
            ),
            # is_lmdb is imported inside the validating function, so patch
            # it at the source module rather than at an import site.
            patch(
                "deepmd.dpmodel.utils.lmdb_data.is_lmdb",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "exactly one path"):
                _get_neighbor_stat_data(params, ["O", "H"])
            with self.assertRaisesRegex(ValueError, "exactly one path"):
                _build_data_system(params, ["O", "H"])


def _make_frame(natoms: int, seed: int, *, include_spin: bool = False) -> dict:
    """Synthetic LMDB frame matching the on-disk schema used by LmdbDataReader."""
    rng = np.random.RandomState(seed)
    half = natoms // 2
    frame = {
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
    if include_spin:
        spin = rng.randn(natoms, 3).astype(np.float64)
        spin[half:] = 0.0
        force_mag = rng.randn(natoms, 3).astype(np.float64)
        force_mag[half:] = 0.0
        frame["spin"] = _encode_array(spin)
        frame["force_mag"] = _encode_array(force_mag)
    return frame


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


def _create_charge_spin_lmdb(path: str, charge_spin: np.ndarray) -> None:
    """Write one frame whose stored condition is *charge_spin*."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    natoms = 6
    with env.begin(write=True) as txn:
        txn.put(
            b"__metadata__",
            msgpack.packb(
                {
                    "nframes": 1,
                    "frame_idx_fmt": fmt,
                    "system_info": {
                        "formula": "O3H3",
                        "natoms": [3, 3],
                        "nframes": 1,
                    },
                },
                use_bin_type=True,
            ),
        )
        frame = _make_frame(natoms, 0)
        frame["charge_spin"] = _encode_array(charge_spin.astype(np.float64))
        txn.put(format(0, fmt).encode(), msgpack.packb(frame, use_bin_type=True))
    env.close()


def _create_partially_labeled_lmdb(path: str) -> None:
    """Write frames split between energy-only and force-only labels."""
    nframes = 4
    natoms = 6
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": fmt,
        "system_info": {"natoms": [3, 3]},
        "frame_nlocs": [natoms] * nframes,
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for index in range(nframes):
            frame = _make_frame(natoms, index)
            if index % 2 == 0:
                frame.pop("forces")
            else:
                frame.pop("energies")
            txn.put(
                format(index, fmt).encode(),
                msgpack.packb(frame, use_bin_type=True),
            )
    env.close()


def _create_mixed_nloc_test_lmdb(path: str, *, include_spin: bool = False) -> None:
    """Write an LMDB with five six-atom and five nine-atom frames."""
    frame_nlocs = [6] * 5 + [9] * 5
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    metadata = {
        "nframes": len(frame_nlocs),
        "frame_idx_fmt": fmt,
        "frame_nlocs": frame_nlocs,
        "type_map": ["O", "H"],
        "system_info": {
            "formula": "mixed",
            "natoms": [3, 3],
            "nframes": len(frame_nlocs),
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for index, nloc in enumerate(frame_nlocs):
            key = format(index, fmt).encode()
            txn.put(
                key,
                msgpack.packb(
                    _make_frame(nloc, index, include_spin=include_spin),
                    use_bin_type=True,
                ),
            )
    env.close()


class TestLmdbDataSystemGetBatch(unittest.TestCase):
    """LmdbDataSystem.get_batch produces a numpy dict that normalize_batch accepts."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.tmpdir, "test.lmdb")
        self.mixed_lmdb_path = os.path.join(self.tmpdir, "mixed.lmdb")
        _create_test_lmdb(self.lmdb_path, nframes=8, natoms=6)
        _create_mixed_nloc_test_lmdb(self.mixed_lmdb_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_unaddressable_charge_state_is_rejected(self) -> None:
        """An LMDB condition indexes the same tables, so it is checked too.

        The gathers behind ``charge_spin`` are unguarded, so a stored value
        that names no table row has to fail on the numpy batch rather than be
        truncated onto a neighbouring row inside the forward.
        """
        for state, message in (
            (np.array([[0.5, 1.0]]), "charge must be an integer"),
            (np.array([[0.0, 100.0]]), r"multiplicity must lie in \[0, 100\)"),
        ):
            with self.subTest(charge_spin=state.tolist()):
                path = os.path.join(self.tmpdir, f"cs{abs(hash(state.tobytes()))}.lmdb")
                _create_charge_spin_lmdb(path, state)
                ds = LmdbDataSystem(
                    lmdb_path=path, type_map=["O", "H"], batch_size=1, seed=0
                )
                with self.assertRaisesRegex(ValueError, message):
                    normalize_batch(ds.get_batch())

    def test_addressable_charge_state_survives_the_batch(self) -> None:
        path = os.path.join(self.tmpdir, "cs_ok.lmdb")
        _create_charge_spin_lmdb(path, np.array([[-1.0, 3.0]]))
        ds = LmdbDataSystem(lmdb_path=path, type_map=["O", "H"], batch_size=1, seed=0)
        norm = normalize_batch(ds.get_batch())
        np.testing.assert_allclose(
            np.reshape(norm["charge_spin"], (-1, 2)), [[-1.0, 3.0]]
        )

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

    def test_streaming_batch_matches_frame_collation(self) -> None:
        """Preallocated decoding preserves the legacy per-frame contract."""
        ds = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
            num_workers=0,
        )
        indices = [1, 6]
        expected = collate_lmdb_frames([ds._reader[index] for index in indices])
        actual = ds._reader.decode_batch(indices)

        self.assertEqual(tuple(actual), tuple(expected))
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                self.assertEqual(actual_value, expected_value)

    def test_parallel_prefetch_matches_serial_order(self) -> None:
        """Worker processes preserve sampler order and numerical values."""
        serial = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=7,
            num_workers=0,
        )
        parallel = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=7,
            num_workers=2,
        )
        try:
            for _ in range(6):
                expected = serial.get_batch()
                actual = parallel.get_batch()
                self.assertEqual(actual["fid"], expected["fid"])
                for key, expected_value in expected.items():
                    actual_value = actual[key]
                    if isinstance(expected_value, np.ndarray):
                        np.testing.assert_array_equal(actual_value, expected_value)
                    else:
                        self.assertEqual(actual_value, expected_value)
                pending = parallel._batch_iterator._pending
                self.assertIsNotNone(pending)
                self.assertLessEqual(len(pending.futures), 2)
        finally:
            parallel.close()

    def test_data_requirements_freeze_after_first_read(self) -> None:
        """Batch schemas cannot change after a prefetched read."""
        ds = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
            num_workers=2,
        )
        ds.get_batch()
        with self.assertRaisesRegex(
            RuntimeError,
            "must be registered before reading",
        ):
            ds.add_data_requirements([DataRequirementItem("late_label", ndof=1)])

    def test_missing_mandatory_model_input_is_rejected(self) -> None:
        """A required model input cannot silently become a zero-filled batch."""
        ds = LmdbDataSystem(
            lmdb_path=self.lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
            num_workers=0,
        )
        ds.add_data_requirements(
            [DataRequirementItem("spin", 3, atomic=True, must=True)]
        )
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                r"spin.*frame \d+.*test\.lmdb.*field is absent",
            ):
                ds.get_batch()
        finally:
            ds.close()

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

    def test_distributed_batches_are_sharded_with_equal_epoch_lengths(self) -> None:
        """Ranks cover the global pass without advancing epochs at different steps."""
        systems = [
            LmdbDataSystem(
                lmdb_path=self.lmdb_path,
                type_map=["O", "H"],
                batch_size=3,
                seed=7,
                num_workers=0,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]
        try:
            self.assertEqual([system.nbatches for system in systems], [[3], [3]])
            self.assertEqual([len(system._sampler) for system in systems], [2, 2])
            batches = [
                [system.get_batch()["fid"] for _ in range(len(system._sampler))]
                for system in systems
            ]
        finally:
            for system in systems:
                system.close()

        self.assertFalse(set(batches[0][0]) & set(batches[1][0]))
        observed = {
            frame_id
            for rank_batches in batches
            for batch in rank_batches
            for frame_id in batch
        }
        self.assertEqual(observed, set(range(8)))

    def test_add_data_requirements_passthrough(self) -> None:
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

    def test_default_atom_pref_does_not_partition_training_data(self) -> None:
        """OC20-style unit defaults keep mixed atom_pref frames compatible."""
        path = os.path.join(self.tmpdir, "default_atom_pref.lmdb")
        _create_test_lmdb(path, nframes=8, natoms=6)
        environment = lmdb.open(path, readonly=False, lock=False)
        with environment.begin(write=True) as transaction:
            for index in range(0, 8, 2):
                key = format(index, "012d").encode()
                frame = msgpack.unpackb(transaction.get(key), raw=False)
                frame["atom_pref"] = _encode_array(np.full(6, 2.0))
                transaction.put(key, msgpack.packb(frame, use_bin_type=True))
        environment.close()

        loss = EnergyLoss(
            starter_learning_rate=0.002,
            start_pref_e=20.0,
            limit_pref_e=20.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_pf=20.0,
            limit_pref_pf=20.0,
            start_pref_v=5.0,
            limit_pref_v=5.0,
            loss_func="mae",
            f_use_norm=True,
            use_default_pf=True,
        )
        with patch.object(
            lmdb_data_module,
            "_raw_frame_availability",
            wraps=lmdb_data_module._raw_frame_availability,
        ) as inspect_availability:
            ds = LmdbDataSystem(
                lmdb_path=path,
                type_map=["O", "H"],
                batch_size="mix:30000",
                seed=0,
            )
            self.assertEqual(inspect_availability.call_count, 0)
            ds.add_data_requirements(loss.label_requirement)

        try:
            self.assertLessEqual(inspect_availability.call_count, 8)
            self.assertEqual(len(ds._stat_groups), 1)
            batch = ds._reader.decode_batch([0, 1])
            np.testing.assert_array_equal(batch["atom_pref"][0], np.full(18, 2.0))
            np.testing.assert_array_equal(batch["atom_pref"][1], np.ones(18))
            self.assertEqual(float(batch["find_atom_pref"]), 1.0)
            model_output = {
                "energy": np.zeros_like(batch["energy"]),
                "force": np.zeros_like(batch["force"]),
                "virial": np.zeros_like(batch["virial"]),
                "atom_energy": np.zeros((2, 6, 1), dtype=batch["energy"].dtype),
            }
            loss_value, metrics = loss.call(0.002, 6, model_output, batch)
            self.assertTrue(np.isfinite(loss_value))
            self.assertIn("mae_pf", metrics)
        finally:
            ds.close()

    def test_missing_force_disables_default_prefactor_force_loss(self) -> None:
        """Default atom weights cannot supervise a frame without force."""
        path = os.path.join(self.tmpdir, "partial_force_for_pf.lmdb")
        _create_partially_labeled_lmdb(path)
        loss = EnergyLoss(
            starter_learning_rate=1.0,
            start_pref_e=0.0,
            limit_pref_e=0.0,
            start_pref_f=0.0,
            limit_pref_f=0.0,
            start_pref_pf=1.0,
            limit_pref_pf=1.0,
            use_default_pf=True,
        )
        ds = LmdbDataSystem(
            lmdb_path=path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
            num_workers=0,
        )
        ds.add_data_requirements(loss.label_requirement)
        try:
            observed: dict[float, float] = {}
            for _ in range(2):
                batch = ds.get_batch()
                nframes, nloc = batch["atype"].shape
                model_output = {
                    "energy": np.zeros_like(batch.get("energy", np.zeros(nframes))),
                    "force": np.zeros_like(batch["force"]),
                    "virial": np.zeros((nframes, 9), dtype=batch["force"].dtype),
                    "atom_energy": np.zeros(
                        (nframes, nloc, 1),
                        dtype=batch["force"].dtype,
                    ),
                }
                loss_value, metrics = loss.call(
                    1.0,
                    nloc,
                    model_output,
                    batch,
                )
                find_force = float(batch["find_force"])
                observed[find_force] = float(loss_value)
                if find_force == 0.0:
                    self.assertTrue(np.isnan(metrics["rmse_pf"]))

            self.assertEqual(observed[0.0], 0.0)
            self.assertGreater(observed[1.0], 0.0)
        finally:
            ds.close()

    def test_partial_labels_are_batched_by_availability(self) -> None:
        from deepmd.utils.data import (
            DataRequirementItem,
        )

        partial_path = os.path.join(self.tmpdir, "partial.lmdb")
        _create_partially_labeled_lmdb(partial_path)
        ds = LmdbDataSystem(
            lmdb_path=partial_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
        )
        ds.add_data_requirements(
            [
                DataRequirementItem(
                    "energy", ndof=1, atomic=False, must=False, default=7.0
                ),
                DataRequirementItem(
                    "force", ndof=3, atomic=True, must=False, default=11.0
                ),
            ]
        )

        try:
            batches = [ds.get_batch(), ds.get_batch()]
            observed = {
                (float(batch["find_energy"]), float(batch["find_force"]))
                for batch in batches
            }
            self.assertEqual(observed, {(1.0, 0.0), (0.0, 1.0)})
            self.assertTrue(all(batch["coord"].shape[0] == 2 for batch in batches))

            stat_samples = make_stat_input(ds, nbatches=10)
            stat_availability = {
                (float(sample["find_energy"]), float(sample["find_force"]))
                for sample in stat_samples
            }
            self.assertEqual(stat_availability, observed)
            self.assertTrue(
                all(sample["coord"].shape[0] == 2 for sample in stat_samples)
            )
        finally:
            ds.close()

    def test_stat_input_partitions_mixed_nloc_batches(self) -> None:
        """Statistics expose each atom-count group as one logical system."""
        ds = LmdbDataSystem(
            lmdb_path=self.mixed_lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
        )

        self.assertEqual(ds.get_nsystems(), 1)
        self.assertEqual(ds.get_stat_nsystems(), 2)
        sampled = make_stat_input(ds, nbatches=10)

        self.assertEqual(len(sampled), 2)
        self.assertEqual(
            sorted(sample["atype"].shape for sample in sampled),
            [(5, 6), (5, 9)],
        )

    def test_stat_input_limits_batches_per_nloc_group(self) -> None:
        """Statistics honor the requested batch cap for every nloc group."""
        ds = LmdbDataSystem(
            lmdb_path=self.mixed_lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
        )

        sampled = make_stat_input(ds, nbatches=2)

        self.assertEqual(
            sorted(sample["atype"].shape for sample in sampled),
            [(4, 6), (4, 9)],
        )

    def test_stat_batches_restart_after_non_divisible_pass(self) -> None:
        """Statistical batches restart after including a partial tail batch."""
        ds = LmdbDataSystem(
            lmdb_path=self.mixed_lmdb_path,
            type_map=["O", "H"],
            batch_size=2,
            seed=0,
        )

        for sys_idx, nloc in enumerate((6, 9)):
            self.assertEqual(ds.get_stat_numb_batches(sys_idx), 3)
            shapes = [ds.get_stat_batch(sys_idx)["atype"].shape for _ in range(4)]
            self.assertEqual(
                shapes,
                [(2, nloc), (2, nloc), (1, nloc), (2, nloc)],
            )


class TestLmdbTrainingLoop(unittest.TestCase):
    """End-to-end: get_trainer routes an LMDB path and runs training steps."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.lmdb_path = os.path.join(cls.tmpdir, "train.lmdb")
        cls.mixed_lmdb_path = os.path.join(cls.tmpdir, "mixed.lmdb")
        cls.val_lmdb_path = os.path.join(cls.tmpdir, "val.lmdb")
        _create_test_lmdb(cls.lmdb_path, nframes=8, natoms=6)
        _create_mixed_nloc_test_lmdb(cls.mixed_lmdb_path)
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

    def test_numb_epoch_counts_passes_over_the_lmdb(self) -> None:
        """One epoch is one pass over the frames of the LMDB."""
        config = self._make_lmdb_config()
        del config["training"]["numb_steps"]
        config["training"]["numb_epoch"] = 2.0
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            trainer = get_trainer(config)
        finally:
            os.chdir(cwd)

        # train.lmdb holds eight frames, read one frame per batch.
        self.assertEqual(trainer.num_steps, 2 * 8)

    def test_numb_epoch_counts_mixed_nloc_batches(self) -> None:
        """One epoch of a ``mix:N`` dataset is one pass over its padded batches.

        The batch count of a mixed-nloc pass depends on how the shuffle groups
        atom counts, so an epoch can only be measured on the sampler the
        trainer will actually draw from, not on a nominal batch size.
        """
        config = self._make_lmdb_config()
        del config["training"]["numb_steps"]
        config["training"]["numb_epoch"] = 3.0
        config["training"]["training_data"]["systems"] = self.mixed_lmdb_path
        config["training"]["training_data"]["batch_size"] = "mix:27"
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            trainer = get_trainer(config)
        finally:
            os.chdir(cwd)

        data = trainer.training_data
        self.assertTrue(data._reader.mixed_nloc)
        self.assertEqual(trainer.num_steps, 3 * data.nbatches[0])

    def test_training_closes_parallel_lmdb_pipeline(self) -> None:
        """Trainer shutdown releases spawned decoder processes."""
        config = self._make_lmdb_config(numb_steps=2)
        config["training"]["training_data"]["batch_size"] = 2
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with patch.dict(os.environ, {"DP_LMDB_NUM_WORKERS": "2"}):
                trainer = get_trainer(config)
                trainer.run()
            self.assertTrue(trainer.training_data._batch_iterator.closed)
            self.assertIsNone(trainer.training_data._batch_iterator._pool)
            self.assertTrue(trainer.training_data._reader.closed)
        finally:
            os.chdir(cwd)

    def test_mixed_nloc_statistics_and_training(self) -> None:
        """Trainer computes statistics and trains across fixed-nloc batches."""
        config = self._make_lmdb_config(numb_steps=2)
        config["model"]["data_stat_nbatch"] = 10
        config["training"]["training_data"]["systems"] = self.mixed_lmdb_path
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        with tempfile.TemporaryDirectory(dir=self.tmpdir) as run_dir:
            cwd = os.getcwd()
            os.chdir(run_dir)
            try:
                trainer = get_trainer(config)
                self.assertEqual(trainer.training_data.get_nsystems(), 1)
                self.assertEqual(trainer.training_data.get_stat_nsystems(), 2)
                trainer.run()
            finally:
                os.chdir(cwd)

    def test_mix_batch_size_trains_on_padded_batches(self) -> None:
        """``mix:N`` reaches the trainer and its padded batches train."""
        config = self._make_lmdb_config(numb_steps=2)
        config["model"]["data_stat_nbatch"] = 10
        config["training"]["training_data"]["systems"] = self.mixed_lmdb_path
        # The fixture holds five 6-atom and five 9-atom frames; this budget
        # puts a batch boundary inside the atom-count-sorted run, so at least
        # one batch spans both sizes and is padded.
        config["training"]["training_data"]["batch_size"] = "mix:27"
        config = update_deepmd_input(config, warning=False)
        config = normalize(config)

        with tempfile.TemporaryDirectory(dir=self.tmpdir) as run_dir:
            cwd = os.getcwd()
            os.chdir(run_dir)
            try:
                trainer = get_trainer(config)
                data = trainer.training_data
                self.assertTrue(data._reader.mixed_nloc)
                # Statistics keep their own fixed-nloc view, so padding never
                # reaches the per-type accumulators.
                self.assertEqual(data.get_stat_nsystems(), 2)
                for sys_idx in range(data.get_stat_nsystems()):
                    stat = data.get_stat_batch(sys_idx)
                    self.assertTrue((stat["atype"] >= 0).all())
                padded = next(
                    batch
                    for batch in (data.get_batch() for _ in range(10))
                    if (batch["atype"] < 0).any()
                )
                self.assertEqual(padded["atype"].shape[1], 9)
                self.assertEqual(padded["coord"].shape[1], 9)
                self.assertEqual(padded["force"].shape[1], 9)
                np.testing.assert_array_equal(padded["force"][padded["atype"] < 0], 0.0)
                # Per-frame atom counts stay real, not padded.
                self.assertEqual(sorted(set(padded["natoms"][:, 0].tolist())), [6, 9])
                trainer.run()
            finally:
                os.chdir(cwd)


class TestRaggedTrainingBatches(unittest.TestCase):
    """A model reading a flat node axis is fed one, with nothing padded."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.lmdb_path = os.path.join(self.tmpdir, "mixed.lmdb")
        self.spin_lmdb_path = os.path.join(self.tmpdir, "mixed-spin.lmdb")
        _create_mixed_nloc_test_lmdb(self.lmdb_path)
        _create_mixed_nloc_test_lmdb(self.spin_lmdb_path, include_spin=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _config_for_model(
        self,
        model: dict,
        loss: dict,
        *,
        lmdb_path: str | None = None,
    ) -> dict:
        config = {
            "model": model,
            "learning_rate": {
                "type": "exp",
                "decay_steps": 500,
                "start_lr": 1e-3,
                "stop_lr": 3.5e-8,
            },
            "loss": loss,
            "training": {
                "training_data": {
                    "systems": self.lmdb_path if lmdb_path is None else lmdb_path,
                    "batch_size": "mix:27",
                },
                "numb_steps": 2,
                "seed": 10,
                "disp_file": "lcurve.out",
                "disp_freq": 1,
                "save_freq": 100,
            },
        }
        return normalize(update_deepmd_input(config, warning=False))

    def _config(self, descriptor: dict) -> dict:
        return self._config_for_model(
            {
                "type_map": ["O", "H"],
                "descriptor": descriptor,
                "fitting_net": {
                    "neuron": [8, 8],
                    "precision": "float64",
                    "seed": 1,
                },
                "data_stat_nbatch": 1,
            },
            {
                "type": "ener",
                "start_pref_e": 0.02,
                "limit_pref_e": 1,
                "start_pref_f": 1000,
                "limit_pref_f": 1,
                "start_pref_v": 0,
                "limit_pref_v": 0,
            },
        )

    @staticmethod
    def _dpa1() -> dict:
        return {
            "type": "dpa1",
            "sel": 12,
            "rcut_smth": 0.5,
            "rcut": 3.0,
            "neuron": [8, 16],
            "axis_neuron": 4,
            "attn_layer": 0,
            "precision": "float64",
            "seed": 1,
        }

    @staticmethod
    def _se_e2_a() -> dict:
        return {
            "type": "se_e2_a",
            "sel": [6, 12],
            "rcut_smth": 0.5,
            "rcut": 3.0,
            "neuron": [8, 16],
            "axis_neuron": 4,
            "seed": 1,
        }

    @staticmethod
    def _dpa4_native_spin_model() -> dict:
        return {
            "type": "dpa4",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa4",
                "sel": 20,
                "rcut": 3.0,
                "channels": 8,
                "n_radial": 4,
                "lmax": 1,
                "mmax": 1,
                "n_blocks": 1,
                "precision": "float64",
                "seed": 1,
            },
            "fitting_net": {
                "type": "dpa4_ener",
                "neuron": [8],
                "precision": "float64",
                "seed": 1,
            },
            "spin": {"use_spin": [True, False], "scheme": "native"},
            "data_stat_nbatch": 1,
        }

    def _virtual_spin_model(self) -> dict:
        return {
            "type_map": ["O", "H"],
            "descriptor": self._se_e2_a(),
            "fitting_net": {
                "neuron": [8, 8],
                "precision": "float64",
                "seed": 1,
            },
            "spin": {
                "use_spin": [True, False],
                "virtual_scale": [0.314],
            },
            "data_stat_nbatch": 1,
        }

    @staticmethod
    def _spin_loss() -> dict:
        return {
            "type": "ener_spin",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_fr": 1000,
            "limit_pref_fr": 1,
            "start_pref_fm": 1000,
            "limit_pref_fm": 1,
        }

    def _run(self, descriptor: dict, *, compile: bool = False):
        """Train two steps and return the trainer plus one drawn batch."""
        config = self._config(descriptor)
        config["training"]["enable_compile"] = compile
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            trainer = get_trainer(config)
            batch = trainer.training_data.get_batch()
            trainer.run()
            return trainer, batch
        finally:
            os.chdir(cwd)

    def _run_spin(self, model: dict):
        """Train a spin model for two mixed-size steps and return one batch."""
        config = self._config_for_model(
            model,
            self._spin_loss(),
            lmdb_path=self.spin_lmdb_path,
        )
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            trainer = get_trainer(config)
            batch = trainer.training_data.get_batch()
            trainer.run()
            return trainer, batch
        finally:
            os.chdir(cwd)

    def test_graph_model_trains_on_a_flat_node_axis(self) -> None:
        """DPA1 reads a graph lower, so its batches carry no padded row."""
        trainer, batch = self._run(self._dpa1())
        self.assertTrue(trainer.training_data._reader.ragged_batches)
        self.assertEqual(batch["coord"].ndim, 2)
        self.assertEqual(batch["atype"].ndim, 1)
        self.assertEqual(batch["coord"].shape[0], int(batch["n_node"].sum()))
        self.assertTrue((batch["atype"] >= 0).all(), "nothing is padded")
        # Frame-level fields keep their frame axis.
        self.assertEqual(batch["energy"].shape[0], batch["n_node"].shape[0])

    def test_loss_contract_can_require_rectangular_batches(self) -> None:
        """Generalized-force and Hessian labels keep their rectangular axes."""
        cases = {
            "generalized force": {
                "start_pref_gf": 1.0,
                "limit_pref_gf": 1.0,
                "numb_generalized_coord": 2,
            },
            "Hessian": {
                "start_pref_h": 1.0,
                "limit_pref_h": 1.0,
            },
        }
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            for name, loss_params in cases.items():
                with self.subTest(loss=name):
                    config = self._config(self._dpa1())
                    config["loss"].update(loss_params)
                    trainer = get_trainer(config)
                    self.assertFalse(trainer.training_data._reader.ragged_batches)
        finally:
            os.chdir(cwd)

    def test_non_mixing_rule_keeps_the_rectangular_layout(self) -> None:
        """Model capability alone does not change established LMDB batches."""
        config = self._config(self._dpa1())
        config["training"]["training_data"]["batch_size"] = 2
        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            trainer = get_trainer(config)
            batch = trainer.training_data.get_batch()
        finally:
            os.chdir(cwd)

        self.assertFalse(trainer.training_data._reader.ragged_batches)
        self.assertEqual(batch["coord"].ndim, 3)
        self.assertEqual(batch["atype"].ndim, 2)
        self.assertNotIn("n_node", batch)

    @REQUIRES_SUPPORTED_COMPILE
    def test_compiled_graph_model_trains_on_a_flat_node_axis(self) -> None:
        """The compiled lower reads the flat axis too, so compiling changes nothing.

        Its trace is taken on a rectangular system, which would bake in
        ``N == nframes * nloc`` were the frame, node and edge counts not kept
        as independent symbols.
        """
        trainer, batch = self._run(self._dpa1(), compile=True)
        self.assertEqual(
            type(trainer.wrapper.model["Default"]).__name__, "_CompiledModel"
        )
        self.assertTrue((batch["atype"] >= 0).all())
        self.assertEqual(batch["coord"].shape[0], int(batch["n_node"].sum()))

    def test_native_spin_trains_on_a_flat_node_axis(self) -> None:
        """Native spin carries moments and magnetic labels through ragged training."""
        trainer, batch = self._run_spin(self._dpa4_native_spin_model())

        self.assertTrue(trainer.training_data._reader.ragged_batches)
        self.assertEqual(batch["coord"].ndim, 2)
        self.assertEqual(batch["spin"].shape, batch["coord"].shape)
        self.assertEqual(batch["force_mag"].shape, batch["coord"].shape)
        self.assertEqual(batch["coord"].shape[0], int(batch["n_node"].sum()))
        self.assertTrue((batch["atype"] >= 0).all())

    def test_virtual_spin_trains_on_a_rectangular_node_axis(self) -> None:
        """Virtual-atom spin keeps mixed-size training rectangular and masked."""
        trainer, batch = self._run_spin(self._virtual_spin_model())

        self.assertFalse(trainer.training_data._reader.ragged_batches)
        self.assertEqual(batch["coord"].ndim, 3)
        self.assertEqual(batch["spin"].shape, batch["coord"].shape)
        self.assertEqual(batch["force_mag"].shape, batch["coord"].shape)
        self.assertNotIn("n_node", batch)
        self.assertTrue((batch["atype"] < 0).any())

    def test_dense_model_keeps_padded_batches(self) -> None:
        """se_e2_a reads a rectangular node axis and must still be padded."""
        trainer, batch = self._run(self._se_e2_a())
        self.assertFalse(trainer.training_data._reader.ragged_batches)
        self.assertEqual(batch["coord"].ndim, 3)
        self.assertEqual(batch["atype"].ndim, 2)
        self.assertNotIn("n_node", batch)

    def test_layout_is_settled_before_the_pass_is_counted(self) -> None:
        """The run length must count the packing training actually uses.

        Under ``mix:N`` the layout decides where the sampler cuts a batch, and
        the sampler materializes a pass the moment its length is asked for.
        Settling the layout afterwards would count one packing and train on
        another, and would serve the first epoch from the counted one.

        The pass is cached once built, so both orderings report the same count
        after the fact. What distinguishes them is the layout in force at the
        moment the count is taken, which is what this observes.
        """
        config = self._config(self._dpa1())
        del config["training"]["numb_steps"]
        config["training"]["numb_epoch"] = 1.0

        observed: dict[str, bool] = {}
        original = Trainer._epoch_length

        def record_layout(trainer_self, model_key: str) -> int:
            observed["ragged_when_counted"] = (
                trainer_self.training_data._reader.ragged_batches
            )
            return original(trainer_self, model_key)

        cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with patch.object(Trainer, "_epoch_length", record_layout):
                trainer = get_trainer(config)
        finally:
            os.chdir(cwd)

        # DPA1 reads a flat node axis, so training runs on ragged batches.
        self.assertTrue(trainer.training_data._reader.ragged_batches)
        self.assertTrue(observed["ragged_when_counted"])


if __name__ == "__main__":
    unittest.main()
