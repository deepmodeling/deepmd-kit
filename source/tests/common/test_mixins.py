# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared test mixins and utilities used by both pt and pt_expt tests.

These are kept in ``tests.common`` so that importing them does NOT trigger
``tests.pt.__init__`` (which sets ``torch.set_default_device("cuda:9999999")``
and pushes a ``DeviceContext`` onto the mode stack, breaking pt_expt tests
on CPU-only machines).
"""

import numpy as np


class TestCaseSingleFrameWithNlist:
    """Mixin providing a small 2-frame, 2-type test system with neighbor list."""

    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 2, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall, 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        self.mapping = np.array([0, 1, 2, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.sel_mix = [7]
        self.natoms = [3, 3, 2, 1]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 2.2
        self.rcut_smth = 0.4
        # permutations
        self.perm = np.array([2, 0, 1, 3], dtype=np.int32)
        inv_perm = np.array([1, 2, 0, 3], dtype=np.int32)
        # permute the coord and atype
        self.coord_ext = np.concatenate(
            [self.coord_ext, self.coord_ext[:, self.perm, :]], axis=0
        ).reshape(self.nf, self.nall * 3)
        self.atype_ext = np.concatenate(
            [self.atype_ext, self.atype_ext[:, self.perm]], axis=0
        )
        self.mapping = np.concatenate(
            [self.mapping, self.mapping[:, self.perm]], axis=0
        )

        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)
        self.atol = 1e-12


def get_tols(prec):
    """Return (rtol, atol) for a given precision string."""
    if prec in ["single", "float32"]:
        rtol, atol = 0.0, 1e-4
    elif prec in ["double", "float64"]:
        rtol, atol = 0.0, 1e-12
    else:
        raise ValueError(f"unknown prec {prec}")
    return rtol, atol
