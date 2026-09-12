# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Uni-Mol to deepmd data conversion."""

import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np

from deepmd.dpmodel.descriptor.unimol import (
    UNIMOL_ELEMENTS,
)
from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
)
from deepmd.dpmodel.loss.unimol import (
    UniMolLoss,
)
from deepmd.dpmodel.utils.unimol_transform import (
    make_unimol_data_transform,
)
from deepmd.utils.unimol_data import (
    convert_unimol_lmdb,
    read_unimol_lmdb,
)


def _write_unimol_lmdb(path: str, molecules: list[dict]) -> None:
    """Write a file in the upstream layout: one pickle per molecule."""
    import lmdb

    env = lmdb.open(path, subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, mol in enumerate(molecules):
            txn.put(f"{i}".encode(), pickle.dumps(mol))
    env.close()


class TestUniMolDataConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.molecules = [
            {
                "atoms": ["C", "N", "O", "H"],
                "coordinates": [
                    rng.normal(size=(4, 3)).astype(np.float32) for _ in range(3)
                ],
                "smi": "CNO",
            },
            {
                "atoms": ["C", "C", "H"],
                "coordinates": [
                    rng.normal(size=(3, 3)).astype(np.float32) for _ in range(2)
                ],
                "smi": "CC",
            },
            # A single atom cannot be told apart from padding downstream, and an
            # unmapped element would silently become [UNK]; both are skipped.
            {
                "atoms": ["C"],
                "coordinates": [rng.normal(size=(1, 3)).astype(np.float32)],
                "smi": "C",
            },
            {
                "atoms": ["C", "Xx"],
                "coordinates": [rng.normal(size=(2, 3)).astype(np.float32)],
                "smi": "C",
            },
        ]
        self.src = os.path.join(self.tmp, "mol.lmdb")
        _write_unimol_lmdb(self.src, self.molecules)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reader_streams_the_upstream_layout(self) -> None:
        records = list(read_unimol_lmdb(self.src))
        self.assertEqual(len(records), len(self.molecules))
        self.assertEqual(records[0]["atoms"], ["C", "N", "O", "H"])
        self.assertEqual(len(records[0]["coordinates"]), 3)

    def test_conversion_round_trips_through_the_deepmd_reader(self) -> None:
        dst = os.path.join(self.tmp, "converted")
        counts = convert_unimol_lmdb(self.src, dst, map_size=1 << 24)
        # One frame per conformer, and the two unusable records are skipped.
        self.assertEqual(counts["frames"], 5)
        self.assertEqual(counts["molecules"], 2)
        self.assertEqual(counts["skipped"], 2)

        reader = LmdbDataReader(dst, list(UNIMOL_ELEMENTS))
        self.assertEqual(len(reader), 5)
        frame = reader[0]
        coord = np.asarray(frame["coord"]).reshape(-1, 3)
        np.testing.assert_allclose(
            coord, self.molecules[0]["coordinates"][0].astype(np.float64), atol=1e-6
        )
        symbols = [UNIMOL_ELEMENTS[i] for i in np.asarray(frame["atype"]).reshape(-1)]
        self.assertEqual(symbols, self.molecules[0]["atoms"])
        # Molecules are not periodic, so no cell is written at all. A zero cell
        # would not do: the neighbour-list builder takes any cell at face value
        # and inverts it.
        self.assertNotIn("box", frame)

    def test_transform_corrupts_frames_in_the_data_path(self) -> None:
        """The reader hook is what makes self-supervised training possible."""
        type_map = [*UNIMOL_ELEMENTS, "[MASK]"]
        dst = os.path.join(self.tmp, "for_training")
        convert_unimol_lmdb(self.src, dst, type_map=type_map, map_size=1 << 24)
        reader = LmdbDataReader(dst, type_map)
        plain = reader[0]
        reader.set_frame_transform(
            make_unimol_data_transform(type_map, seed=1, epoch=1)
        )
        corrupted = reader[0]

        self.assertEqual(
            sorted(set(corrupted) - set(plain)),
            [
                "find_unimol_coord_target",
                "find_unimol_token_target",
                "unimol_coord_target",
                "unimol_token_target",
            ],
        )
        target = np.asarray(corrupted["unimol_token_target"])
        clean = np.asarray(corrupted["unimol_coord_target"]).reshape(-1, 3)
        noisy = np.asarray(corrupted["coord"]).reshape(-1, 3)
        # Only selected atoms may move, and the clean target is centred.
        moved = np.abs(noisy - clean).max(axis=-1) > 0
        self.assertTrue(bool(np.all(~moved | (target != 0))))
        # Centring is exact only to fp32, because the transform keeps upstream's
        # fp32 coordinates.
        np.testing.assert_allclose(clean.mean(axis=0), np.zeros(3), atol=1e-6)
        # Masked atoms are carried as the pseudo-element. Of the selected
        # atoms, 90% are masked and 5% take a random element; both are moved,
        # so the masked ones are a subset of the moved ones.
        is_mask = np.asarray(corrupted["atype"]) == type_map.index("[MASK]")
        self.assertLessEqual(int(is_mask.sum()), int(moved.sum()))
        self.assertTrue(bool(np.all(~is_mask | moved)))

    def test_the_objective_supplies_its_own_transform(self) -> None:
        """A trainer installs whatever the loss declares, and nothing else.

        Supervised losses return None here, so the data path is untouched for
        them; this objective returns the corruption that produces its labels.
        """
        from deepmd.dpmodel.loss.property import (
            PropertyLoss,
        )

        type_map = [*UNIMOL_ELEMENTS, "[MASK]"]
        self.assertIsNone(
            PropertyLoss(task_dim=1, var_name="property").frame_transform(type_map)
        )

        loss = UniMolLoss(mask_prob=0.2)
        self.assertEqual(
            [r.key for r in loss.label_requirement],
            ["unimol_token_target", "unimol_coord_target"],
        )
        dst = os.path.join(self.tmp, "through_the_loss")
        convert_unimol_lmdb(self.src, dst, type_map=type_map, map_size=1 << 24)
        reader = LmdbDataReader(dst, type_map)
        before = set(reader[0])
        reader.set_frame_transform(loss.frame_transform(type_map))
        after = reader[0]
        for key in ("unimol_token_target", "unimol_coord_target"):
            self.assertIn(key, set(after) - before)
        self.assertEqual(
            len(np.asarray(after["unimol_token_target"])), len(after["atype"])
        )

    def test_transform_requires_the_mask_pseudo_element(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[MASK\]"):
            make_unimol_data_transform(list(UNIMOL_ELEMENTS))

    def test_limits_are_respected(self) -> None:
        dst = os.path.join(self.tmp, "limited")
        counts = convert_unimol_lmdb(
            self.src, dst, max_molecules=1, max_conformers=2, map_size=1 << 24
        )
        self.assertEqual(counts["molecules"], 1)
        self.assertEqual(counts["frames"], 2)


if __name__ == "__main__":
    unittest.main()
