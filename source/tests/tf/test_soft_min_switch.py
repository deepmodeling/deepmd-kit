# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for the ``soft_min_switch`` op when a neighbor row is
fully padded (i.e. an atom has no valid neighbors).

See https://github.com/deepmodeling/deepmd-kit/issues/5651 for context.
Previously ``bb / aa`` divided by zero when every entry of a neighbor row
was ``-1``, producing NaN inside ``spline5_switch``.  Although the current
``spline5_switch`` implementation happens to map NaN to the ``else`` branch
(``vv = 0, dd = 0``) because ``NaN < x`` is always ``false`` in IEEE 754,
this behaviour is fragile and can break under aggressive compiler
optimizations (e.g. ``-ffast-math``).  The fix in ``soft_min_switch.cc``
adds an explicit ``aa == 0`` guard so the zero-initialized output is
preserved without relying on NaN-comparison semantics.

This test verifies that the op produces finite, zero-valued output for an
isolated atom (all neighbors padded) and also checks a mixed scenario
where one atom has neighbors and another does not.
"""

import numpy as np
import tensorflow as tf

import deepmd.tf.op  # noqa: F401
from deepmd.tf.env import (
    op_module,
)
from deepmd.tf.env import tf as dp_tf


class TestSoftMinSwitchEmptyNeighbors(tf.test.TestCase):
    """The switch op must stay finite for isolated atoms (no neighbors)."""

    def _run_isolated(self, dtype: np.dtype) -> None:
        """Single atom with all-padded neighbor list."""
        # A single isolated atom: nloc == nall == 1, one type, one frame.
        nframes = 1
        nloc = 1
        nall = 1
        ntypes = 1
        # ``natoms`` tensor layout: [nloc, nall, ntype_0, ntype_1, ...]
        natoms = np.array([nloc, nall] + [0] * ntypes, dtype=np.int32)
        natoms[2] = nloc  # the single atom is of type 0

        # Neighbor selection: request a single neighbor slot.
        sel_a = [1]
        sel_r = [0]
        nnei = sum(sel_a) + sum(sel_r)

        # All neighbor slots are padded (-1): the atom has no neighbors.
        nlist = np.full((nframes, nloc * nnei), -1, dtype=np.int32)
        # ``rij`` must still be provided with the right shape; values are
        # irrelevant because the padded neighbors are skipped, but they
        # must be finite so the op does not produce NaN from garbage.
        rij = np.zeros((nframes, 3 * nloc * nnei), dtype=dtype)
        # ``type`` tensor has shape [nframes, nall].
        type_tensor = np.zeros((nframes, nall), dtype=np.int32)

        # Build the graph and run the op.
        with self.cached_session() as sess:
            t_type = dp_tf.constant(type_tensor, dtype=dp_tf.int32)
            t_rij = dp_tf.constant(rij, dtype=dtype)
            t_nlist = dp_tf.constant(nlist, dtype=dp_tf.int32)
            t_natoms = dp_tf.constant(natoms, dtype=dp_tf.int32)
            sw_value, sw_deriv = op_module.soft_min_switch(
                t_type,
                t_rij,
                t_nlist,
                t_natoms,
                sel_a=sel_a,
                sel_r=sel_r,
                alpha=0.3,
                rmin=1.0,
                rmax=3.45,
            )
            v_val, d_val = self.evaluate([sw_value, sw_deriv])

        # The switch value and derivatives must be finite (no NaN/Inf).
        self.assertTrue(np.all(np.isfinite(v_val)), f"sw_value not finite: {v_val}")
        self.assertTrue(np.all(np.isfinite(d_val)), f"sw_deriv not finite: {d_val}")
        # With no neighbors the result must remain zero-initialized.
        np.testing.assert_allclose(v_val, 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(d_val, 0.0, rtol=0, atol=0)

    def _run_mixed(self, dtype: np.dtype) -> None:
        """Two atoms: one with a valid neighbor, one isolated."""
        nframes = 1
        nloc = 2
        nall = 2
        ntypes = 1
        natoms = np.array([nloc, nall] + [0] * ntypes, dtype=np.int32)
        natoms[2] = nloc

        sel_a = [1]
        sel_r = [0]
        nnei = sum(sel_a) + sum(sel_r)

        # Atom 0 has a valid neighbor (atom 1); atom 1 is isolated (-1).
        nlist = np.array([[1, -1]], dtype=np.int32)
        # rij for atom 0's neighbor: a small displacement.
        # rij for atom 1's padded slot: zeros (skipped).
        rij = np.zeros((nframes, 3 * nloc * nnei), dtype=dtype)
        rij[0, 0] = 1.0  # atom 0 -> atom 1, x-component
        rij[0, 1] = 0.0  # y
        rij[0, 2] = 0.0  # z
        type_tensor = np.zeros((nframes, nall), dtype=np.int32)

        with self.cached_session() as sess:
            t_type = dp_tf.constant(type_tensor, dtype=dp_tf.int32)
            t_rij = dp_tf.constant(rij, dtype=dtype)
            t_nlist = dp_tf.constant(nlist, dtype=dp_tf.int32)
            t_natoms = dp_tf.constant(natoms, dtype=dp_tf.int32)
            sw_value, sw_deriv = op_module.soft_min_switch(
                t_type,
                t_rij,
                t_nlist,
                t_natoms,
                sel_a=sel_a,
                sel_r=sel_r,
                alpha=0.3,
                rmin=1.0,
                rmax=3.45,
            )
            v_val, d_val = self.evaluate([sw_value, sw_deriv])

        # Both atoms must produce finite results.
        self.assertTrue(np.all(np.isfinite(v_val)), f"sw_value not finite: {v_val}")
        self.assertTrue(np.all(np.isfinite(d_val)), f"sw_deriv not finite: {d_val}")
        # Atom 1 (isolated) must have zero switch value and derivatives.
        np.testing.assert_allclose(v_val[0, 1], 0.0, rtol=0, atol=0)
        np.testing.assert_allclose(d_val[0, 3:6], 0.0, rtol=0, atol=0)
        # Atom 0 (with a neighbor within rmin=1.0) must have switch value 1.
        np.testing.assert_allclose(v_val[0, 0], 1.0, rtol=1e-5, atol=1e-5)

    def test_isolated_float(self) -> None:
        self._run_isolated(np.float32)

    def test_isolated_double(self) -> None:
        self._run_isolated(np.float64)

    def test_mixed_float(self) -> None:
        self._run_mixed(np.float32)

    def test_mixed_double(self) -> None:
        self._run_mixed(np.float64)


if __name__ == "__main__":
    tf.test.main()
