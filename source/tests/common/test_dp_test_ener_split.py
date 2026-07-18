# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for splitting DeepPot.eval optional outputs in `dp test`.

`DeepPot.eval` appends optional outputs after (energy, force, virial) in a fixed
order: atomic (atom_energy, atom_virial), then spin (force_mag, mask_mag), then
hessian. `dp test` used to read the hessian unconditionally from ret[3], so when
atomic or spin outputs were also present the hessian slot was mistaken for an
atomic energy / magnetic force. This checks the hessian (and the other optional
outputs) are read from the correct advancing index.
"""

import unittest

import numpy as np

from deepmd.entrypoints.test import (
    _split_optional_ener_outputs,
)


def _sentinel(value: int, width: int) -> np.ndarray:
    # one frame, filled with `value` so the source slot is identifiable
    return np.full((1, width), value, dtype=np.float64)


# distinct sentinel value per optional output slot
AE, AV, FM, MM, HE = 3, 4, 5, 6, 7


def _ret(*, atomic: bool, spin: bool, hessian: bool) -> tuple:
    ret = [_sentinel(0, 1), _sentinel(1, 6), _sentinel(2, 9)]  # energy, force, virial
    if atomic:
        ret += [_sentinel(AE, 2), _sentinel(AV, 18)]
    if spin:
        ret += [_sentinel(FM, 6), _sentinel(MM, 2)]
    if hessian:
        ret += [_sentinel(HE, 36)]
    return tuple(ret)


class TestSplitOptionalEnerOutputs(unittest.TestCase):
    def _run(self, atomic: bool, spin: bool, hessian: bool):
        return _split_optional_ener_outputs(
            _ret(atomic=atomic, spin=spin, hessian=hessian),
            has_atom_ener=atomic,
            has_spin=spin,
            has_hessian=hessian,
            numb_test=1,
        )

    def test_hessian_only(self) -> None:
        out = self._run(atomic=False, spin=False, hessian=True)
        self.assertEqual(out.hessian[0, 0], HE)
        self.assertIsNone(out.atom_energy)

    def test_atomic_and_hessian(self) -> None:
        # hessian must come from ret[5], not ret[3] (which is atom_energy)
        out = self._run(atomic=True, spin=False, hessian=True)
        self.assertEqual(out.atom_energy[0, 0], AE)
        self.assertEqual(out.atom_virial[0, 0], AV)
        self.assertEqual(out.hessian[0, 0], HE)

    def test_spin_and_hessian(self) -> None:
        out = self._run(atomic=False, spin=True, hessian=True)
        self.assertEqual(out.force_mag[0, 0], FM)
        self.assertEqual(out.mask_mag[0, 0], MM)
        self.assertEqual(out.hessian[0, 0], HE)

    def test_atomic_spin_and_hessian(self) -> None:
        # hessian at ret[7] with all optional outputs present
        out = self._run(atomic=True, spin=True, hessian=True)
        self.assertEqual(out.atom_energy[0, 0], AE)
        self.assertEqual(out.force_mag[0, 0], FM)
        self.assertEqual(out.hessian[0, 0], HE)

    def test_atomic_and_spin_no_hessian(self) -> None:
        out = self._run(atomic=True, spin=True, hessian=False)
        self.assertEqual(out.atom_energy[0, 0], AE)
        self.assertEqual(out.force_mag[0, 0], FM)
        self.assertIsNone(out.hessian)


if __name__ == "__main__":
    unittest.main()
