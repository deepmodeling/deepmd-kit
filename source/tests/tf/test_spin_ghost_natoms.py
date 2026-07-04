# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for the TensorFlow spin ghost-atom type counting.

The spin ``natoms_not_match`` path (triggered only when ``natoms[0] != natoms[1]``,
i.e. ghost atoms are present) needs a *dense*, type-indexed count vector of the
ghost atoms so that the per-type coordinate slices land on the right offsets.

The ghost region fed to the graph is guaranteed to be sorted in ascending type
order by the C++ side (``DeepSpinTF::extend`` reorders ghost atoms into
contiguous per-type blocks), but a type may be entirely absent from a rank's
ghost region.  Counting the ghost types with ``tf.unique_with_counts`` returns
only the *present* types in encounter order, so a missing type shifts every
subsequent count and the last index runs past the end of the vector, corrupting
the ghost spin coordinates or failing at runtime.  ``tf.math.bincount`` with
``minlength=ntypes`` produces the dense vector the slicing assumes.
"""

import unittest

import numpy as np

from deepmd.tf.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.spin import (
    Spin,
)


def _make_spin_descriptor() -> DescrptSeA:
    """Build a minimal ``DescrptSeA`` exposing only what the ghost-count path uses.

    ``natoms_not_match`` / ``natoms_match`` read only ``self.ntypes``,
    ``self.ntypes_spin`` and ``self.spin.use_spin``, so we bypass the full
    constructor and set those attributes directly to keep the test focused on
    the counting logic.  ``use_spin=[True, False]`` gives two real types (type 0
    has spin, type 1 has none) plus one virtual type (type 2 = spin of type 0),
    i.e. ``ntypes == 3``, ``ntypes_spin == 1``.
    """
    dd = DescrptSeA.__new__(DescrptSeA)
    dd.spin = Spin(use_spin=[True, False], spin_norm=[1.0], virtual_len=[0.4])
    dd.ntypes = 3
    dd.ntypes_spin = 1
    return dd


class TestSpinGhostNatoms(unittest.TestCase):
    def _run_natoms_not_match(
        self, coord: np.ndarray, natoms: np.ndarray, atype: np.ndarray
    ) -> np.ndarray:
        dd = _make_spin_descriptor()
        with tf.Graph().as_default():
            t_coord = tf.placeholder(tf.float64, [None, None], name="t_coord")
            t_natoms = tf.placeholder(tf.int32, [2 + dd.ntypes], name="t_natoms")
            t_atype = tf.placeholder(tf.int32, [None, None], name="t_atype")
            diff = dd.natoms_not_match(t_coord, t_natoms, t_atype)
            with tf.Session() as sess:
                return sess.run(
                    diff,
                    feed_dict={
                        t_coord: coord,
                        t_natoms: natoms,
                        t_atype: atype,
                    },
                )

    def test_missing_ghost_type(self) -> None:
        """A non-spin real type absent from the ghost region must not corrupt slices.

        Layout (each atom is one 3-vector):
        - local:  type0, type1, type2      (nloc = 3, dense counts [1, 1, 1])
        - ghost:  type0, type2             (type 1 absent; ascending, dense [1, 0, 1])

        ``diff_coord`` is the real->virtual spin displacement per atom, laid out in
        the same order as ``coord``.  Only the spin/virtual type (2) is nonzero;
        it pairs with its real type (0).
        """
        # fmt: off
        coord = np.array(
            [[
                1.0, 1.0, 1.0,   # local type 0 (real, spin)
                2.0, 2.0, 2.0,   # local type 1 (real, no spin)
                3.0, 3.0, 3.0,   # local type 2 (virtual of 0)
                5.0, 5.0, 5.0,   # ghost type 0 (real, spin)
                6.0, 6.0, 6.0,   # ghost type 2 (virtual of 0)
            ]],
            dtype=np.float64,
        )
        # fmt: on
        natoms = np.array([3, 5, 1, 1, 1], dtype=np.int32)
        atype = np.array([[0, 1, 2, 0, 2]], dtype=np.int32)

        out = self._run_natoms_not_match(coord, natoms, atype)

        # fmt: off
        expected = np.array(
            [[
                0.0, 0.0, 0.0,   # local type 0 -> zero
                0.0, 0.0, 0.0,   # local type 1 -> zero
                2.0, 2.0, 2.0,   # local type 2 -> t2_loc - t0_loc
                0.0, 0.0, 0.0,   # ghost type 0 -> zero
                1.0, 1.0, 1.0,   # ghost type 2 -> t2_ghost - t0_ghost
            ]],
            dtype=np.float64,
        )
        # fmt: on
        np.testing.assert_allclose(out, expected)

    def test_all_ghost_types_present(self) -> None:
        """Control: every type present in the ghost region (behaviour-preserving)."""
        # fmt: off
        coord = np.array(
            [[
                1.0, 1.0, 1.0,   # local type 0
                2.0, 2.0, 2.0,   # local type 1
                3.0, 3.0, 3.0,   # local type 2
                5.0, 5.0, 5.0,   # ghost type 0
                6.5, 6.5, 6.5,   # ghost type 1
                8.0, 8.0, 8.0,   # ghost type 2
            ]],
            dtype=np.float64,
        )
        # fmt: on
        natoms = np.array([3, 6, 1, 1, 1], dtype=np.int32)
        atype = np.array([[0, 1, 2, 0, 1, 2]], dtype=np.int32)

        out = self._run_natoms_not_match(coord, natoms, atype)

        # fmt: off
        expected = np.array(
            [[
                0.0, 0.0, 0.0,   # local type 0
                0.0, 0.0, 0.0,   # local type 1
                2.0, 2.0, 2.0,   # local type 2 -> t2_loc - t0_loc
                0.0, 0.0, 0.0,   # ghost type 0
                0.0, 0.0, 0.0,   # ghost type 1 (no spin)
                3.0, 3.0, 3.0,   # ghost type 2 -> t2_ghost - t0_ghost
            ]],
            dtype=np.float64,
        )
        # fmt: on
        np.testing.assert_allclose(out, expected)


if __name__ == "__main__":
    unittest.main()
