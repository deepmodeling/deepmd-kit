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
        # Molecules are not periodic; the cell is zero.
        np.testing.assert_array_equal(np.asarray(frame["box"]).reshape(-1), np.zeros(9))

    def test_limits_are_respected(self) -> None:
        dst = os.path.join(self.tmp, "limited")
        counts = convert_unimol_lmdb(
            self.src, dst, max_molecules=1, max_conformers=2, map_size=1 << 24
        )
        self.assertEqual(counts["molecules"], 1)
        self.assertEqual(counts["frames"], 2)


if __name__ == "__main__":
    unittest.main()
