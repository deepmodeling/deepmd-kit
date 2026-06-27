# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptNep,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptNep(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def _make(self, **kwargs) -> DescrptNep:
        params = {
            "rcut_radial": self.rcut,
            "rcut_angular": self.rcut * 0.8,
            "sel": self.sel,
            "n_max_radial": 2,
            "n_max_angular": 2,
            "basis_size_radial": 3,
            "basis_size_angular": 3,
            "l_max": 2,
            "l_max_4body": 2,
            "l_max_5body": 1,
            "type_map": ["A", "B"],
            "precision": "float64",
            "seed": GLOBAL_SEED,
        }
        params.update(kwargs)
        return DescrptNep(**params)

    def _randomize(self, dd: DescrptNep) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        dd.radial_coeff.coeff = rng.normal(size=dd.radial_coeff.coeff.shape)
        dd.angular_coeff.coeff = rng.normal(size=dd.angular_coeff.coeff.shape)
        dim = dd.get_dim_out()
        dd.davg = rng.normal(size=(dim,))
        dd.dstd = 0.1 + np.abs(rng.normal(size=(dim,)))

    def test_self_consistency(self) -> None:
        dd0 = self._make()
        self._randomize(dd0)
        dd1 = DescrptNep.deserialize(dd0.serialize())
        mm0 = dd0.call(self.coord_ext, self.atype_ext, self.nlist)
        mm1 = dd1.call(self.coord_ext, self.atype_ext, self.nlist)
        for ii in (0, 4):  # descriptor and switch
            np.testing.assert_allclose(mm0[ii], mm1[ii])

    def test_permutation(self) -> None:
        # Frame 1 of the fixture is a permutation of frame 0; the per-atom
        # descriptor must follow the same permutation.
        dd0 = self._make()
        self._randomize(dd0)
        rd = dd0.call(self.coord_ext, self.atype_ext, self.nlist)[0]
        np.testing.assert_allclose(
            rd[0][self.perm[: self.nloc]], rd[1], atol=1e-12
        )

    def test_rotation_translation_invariance(self) -> None:
        # The descriptor depends only on relative geometry, hence it is invariant
        # under a rigid rotation and translation of all coordinates.
        dd0 = self._make()
        self._randomize(dd0)
        rng = np.random.default_rng(GLOBAL_SEED)
        coord = self.coord_ext.reshape(self.nf, self.nall, 3)
        theta = 0.7
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        moved = (coord @ rot.T + rng.normal(size=(self.nf, 1, 3))).reshape(
            self.nf, self.nall * 3
        )
        ref = dd0.call(self.coord_ext, self.atype_ext, self.nlist)[0]
        out = dd0.call(moved, self.atype_ext, self.nlist)[0]
        np.testing.assert_allclose(ref, out, atol=1e-11)

    def test_dim_out(self) -> None:
        dd = self._make()
        # radial (n_max_radial + 1) + angular (n_max_angular + 1) * num_L
        self.assertEqual(dd.get_dim_out(), 3 + 3 * (2 + 1 + 1))
        self.assertFalse(dd.mixed_types())


if __name__ == "__main__":
    unittest.main()
