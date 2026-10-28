# SPDX-License-Identifier: LGPL-3.0-or-later
"""The TensorFlow fittings reject a serialized ``vacuum_ref`` and load older dictionaries."""

import unittest

from deepmd.dpmodel.fitting import (
    DipoleFitting,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFitting,
)
from deepmd.tf.fit import (
    DipoleFittingSeA,
    DOSFitting,
    EnerFitting,
    PolarFittingSeA,
)

NTYPES, DIM_DESCRPT, EMBEDDING_WIDTH = 2, 6, 4


class TestFittingVacuumRef(unittest.TestCase):
    def cases(self) -> list[tuple[dict, type, int]]:
        """The serialized dictionary, TensorFlow class and version of every fitting."""
        return [
            (
                EnergyFittingNet(NTYPES, DIM_DESCRPT, mixed_types=False).serialize(),
                EnerFitting,
                5,
            ),
            (
                DOSFittingNet(
                    NTYPES, DIM_DESCRPT, numb_dos=3, mixed_types=False
                ).serialize(),
                DOSFitting,
                5,
            ),
            (
                DipoleFitting(
                    NTYPES, DIM_DESCRPT, EMBEDDING_WIDTH, mixed_types=False
                ).serialize(),
                DipoleFittingSeA,
                5,
            ),
            (
                PolarFitting(
                    NTYPES, DIM_DESCRPT, EMBEDDING_WIDTH, mixed_types=False
                ).serialize(),
                PolarFittingSeA,
                6,
            ),
        ]

    def test_vacuum_ref_is_rejected(self) -> None:
        for data, tf_class, version in self.cases():
            self.assertEqual(data["@version"], version)
            self.assertFalse(data["vacuum_ref"])
            with self.assertRaises(NotImplementedError):
                tf_class.deserialize({**data, "vacuum_ref": True}, suffix="")

    def test_previous_version_loads(self) -> None:
        for data, tf_class, version in self.cases():
            older = {k: v for k, v in data.items() if k != "vacuum_ref"}
            older["@version"] = version - 1
            self.assertIsInstance(tf_class.deserialize(older, suffix=""), tf_class)


if __name__ == "__main__":
    unittest.main()
