# SPDX-License-Identifier: LGPL-3.0-or-later
"""The batch boundary rejects a charge state no embedding row answers.

Every backend reads its training batches through
:func:`deepmd.dpmodel.utils.batch.normalize_batch`, whatever data system
produced them, and the charge and multiplicity gathers behind the condition
are unguarded. A value that names no table row therefore has to fail here, on
the numpy batch, rather than be truncated onto a neighbouring row inside the
compiled forward.
"""

import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.utils.batch import (
    normalize_batch,
)
from deepmd.utils.data import (
    DeepmdData,
)

#: Conditions that address no table row, with the value each one violates.
UNADDRESSABLE_STATES = (
    ([[0.5, 1.0]], "charge must be an integer"),
    ([[0.0, 1.5]], "multiplicity must be an integer"),
    ([[-101.0, 1.0]], r"charge must lie in \[-100, 100\)"),
    ([[100.0, 1.0]], r"charge must lie in \[-100, 100\)"),
    ([[0.0, -1.0]], r"multiplicity must lie in \[0, 100\)"),
    ([[0.0, 100.0]], r"multiplicity must lie in \[0, 100\)"),
)


class TestNormalizeBatchChargeState(unittest.TestCase):
    """The shared chokepoint every data system feeds."""

    @staticmethod
    def _batch(charge_spin, nframes: int = 1) -> dict:
        return {
            "coord": np.zeros((nframes, 2, 3)),
            "type": np.zeros((nframes, 2), dtype=int),
            "charge_spin": np.asarray(charge_spin, dtype=np.float64),
            "find_charge_spin": 1.0,
        }

    def test_unaddressable_states_are_rejected(self) -> None:
        for state, message in UNADDRESSABLE_STATES:
            with self.subTest(charge_spin=state):
                with self.assertRaisesRegex(ValueError, message):
                    normalize_batch(self._batch(state))

    def test_every_frame_is_checked(self) -> None:
        # A valid first frame must not mask an invalid later one.
        with self.assertRaisesRegex(ValueError, "charge must be an integer"):
            normalize_batch(self._batch([[0.0, 1.0], [0.5, 1.0]], nframes=2))

    def test_addressable_states_pass_through_unchanged(self) -> None:
        batch = normalize_batch(self._batch([[-1.0, 3.0], [2.0, 0.0]], nframes=2))
        np.testing.assert_allclose(batch["charge_spin"], [[-1.0, 3.0], [2.0, 0.0]])

    def test_an_unclaimed_condition_is_left_alone(self) -> None:
        # ``find_charge_spin`` false means the frame carries a placeholder the
        # model never reads, so it is not a state and is not checked.
        batch = dict(self._batch([[0.5, 1.0]]), find_charge_spin=0.0)
        self.assertIsNotNone(normalize_batch(batch)["charge_spin"])

    def test_a_batch_without_a_condition_is_left_alone(self) -> None:
        batch = self._batch([[0.0, 1.0]])
        del batch["charge_spin"], batch["find_charge_spin"]
        self.assertNotIn("charge_spin", normalize_batch(batch))


class TestStandardDatasetChargeState(unittest.TestCase):
    """A condition read from a standard system reaches the same guard."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        set_dir = self.root / "set.000"
        set_dir.mkdir()
        atom_types = np.array([0, 1], dtype=np.int32)
        np.savetxt(self.root / "type.raw", atom_types, fmt="%d")
        np.save(set_dir / "coord.npy", np.zeros((1, atom_types.size * 3)))
        np.save(set_dir / "box.npy", np.eye(3).reshape(1, 9))
        self.set_dir = set_dir

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _batch(self, charge_spin) -> dict:
        np.save(self.set_dir / "charge_spin.npy", np.asarray([charge_spin]))
        data = DeepmdData(str(self.root))
        data.add("charge_spin", 2, atomic=False, must=True, high_prec=False)
        return data.get_batch(1)

    def test_an_unaddressable_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "charge must be an integer"):
            normalize_batch(self._batch([0.5, 1.0]))

    def test_an_addressable_state_survives(self) -> None:
        batch = normalize_batch(self._batch([-1.0, 3.0]))
        np.testing.assert_allclose(
            np.reshape(batch["charge_spin"], (-1, 2)), [[-1.0, 3.0]]
        )


if __name__ == "__main__":
    unittest.main()
