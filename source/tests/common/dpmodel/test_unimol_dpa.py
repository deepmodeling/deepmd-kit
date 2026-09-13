# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol pretraining heads on a DPA backbone."""

import unittest

import numpy as np

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.fitting.unimol_dpa_heads import (
    EquivariantCoordHead,
    l1_to_cartesian,
)

SMALL = {"nloc": 6, "nnei": 5, "ntypes": 2}


def _rotation(rng):
    """A random proper rotation."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


class TestEquivariantCoordHead(unittest.TestCase):
    """The head that replaces Uni-Mol's pair-channel coordinate update."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(0)
        nloc, nnei = SMALL["nloc"], SMALL["nnei"]
        # float64 throughout: the property under test is exact, and the default
        # single precision would only show it to about 1e-6.
        self.desc = DescrptDPA4(
            rcut=6.0,
            rcut_smth=5.0,
            sel=nnei,
            ntypes=SMALL["ntypes"],
            seed=0,
            precision="float64",
        )
        self.head = EquivariantCoordHead(
            lmax=self.desc.node_readout_lmax,
            channels=self.desc.channels,
            precision="float64",
            seed=1,
        )
        self.coord = self.rng.normal(size=(1, nloc, 3)) * 1.5
        self.atype = self.rng.integers(0, SMALL["ntypes"], size=(1, nloc))
        self.nlist = np.stack(
            [
                np.stack(
                    [
                        np.array([j for j in range(nloc) if j != i][:nnei])
                        for i in range(nloc)
                    ]
                )
            ]
        )
        self.mapping = np.tile(np.arange(nloc), (1, 1))

    def _predict(self, coord):
        _, latent = self.desc.call_with_latent(
            coord, self.atype, self.nlist, mapping=self.mapping
        )
        return self.head(latent)

    def test_one_vector_per_atom(self) -> None:
        out = self._predict(self.coord)
        self.assertEqual(out.shape, (SMALL["nloc"], 3))
        self.assertTrue(bool(np.all(np.isfinite(out))))

    def test_rotating_the_molecule_rotates_the_prediction(self) -> None:
        """The point of reading l=1 rather than three arbitrary numbers.

        A coordinate update has to be equivariant; if it were not, the model
        could not denoise a rotated copy of a structure it had already learned.
        """
        base = self._predict(self.coord)
        for trial in range(5):
            rot = _rotation(self.rng)
            with self.subTest(rotation=trial):
                turned = self._predict(self.coord @ rot.T)
                np.testing.assert_allclose(turned, base @ rot.T, rtol=1e-9, atol=1e-9)

    def test_translating_the_molecule_changes_nothing(self) -> None:
        base = self._predict(self.coord)
        shifted = self._predict(self.coord + self.rng.normal(size=3))
        np.testing.assert_allclose(shifted, base, rtol=1e-9, atol=1e-9)

    def test_reading_the_rows_as_xyz_would_not_be_equivariant(self) -> None:
        """Guards the decode itself, not just the head around it.

        SeZM packs l=1 in its own basis, so taking the rows as (x, y, z) gives
        something that does not rotate. This pins the mapping that does.
        """
        rows = self.rng.normal(size=(4, 3))
        np.testing.assert_allclose(
            l1_to_cartesian(rows),
            np.stack([-rows[:, 2], rows[:, 0], rows[:, 1]], axis=-1),
        )

    def test_a_backbone_without_l1_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least degree 1"):
            EquivariantCoordHead(lmax=0, channels=8)

    def test_serialize_round_trip(self) -> None:
        revived = EquivariantCoordHead.deserialize(self.head.serialize())
        _, latent = self.desc.call_with_latent(
            self.coord, self.atype, self.nlist, mapping=self.mapping
        )
        np.testing.assert_allclose(revived(latent), self.head(latent))


if __name__ == "__main__":
    unittest.main()
