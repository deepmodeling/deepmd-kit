# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDPAtomicModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]

        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            md0 = DPAtomicModel(ds, ft, type_map=type_map)
            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            md1 = DPAtomicModel.deserialize(md0.serialize())

            ret0 = md0.forward_common_atomic(self.coord_ext, self.atype_ext, self.nlist)
            ret1 = md1.forward_common_atomic(self.coord_ext, self.atype_ext, self.nlist)

            np.testing.assert_allclose(ret0["energy"], ret1["energy"])

    def test_excl_consistency(self):
        type_map = ["foo", "bar"]

        # test the case of exclusion
        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            ds = DescrptSeA(
                self.rcut,
                self.rcut_smth,
                self.sel,
            )
            ft = InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
            )
            md0 = DPAtomicModel(
                ds,
                ft,
                type_map=type_map,
            )
            md1 = DPAtomicModel.deserialize(md0.serialize())

            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            # hacking!
            md1.descriptor.reinit_exclude(pair_excl)
            md1.fitting.reinit_exclude(atom_excl)

            # check energy consistency
            args = [self.coord_ext, self.atype_ext, self.nlist]
            ret0 = md0.forward_common_atomic(*args)
            ret1 = md1.forward_common_atomic(*args)
            np.testing.assert_allclose(
                ret0["energy"],
                ret1["energy"],
            )

            # check output def
            out_names = [vv.name for vv in md0.atomic_output_def().get_data().values()]
            if atom_excl == []:
                self.assertEqual(out_names, ["energy"])
            else:
                self.assertEqual(out_names, ["energy", "mask"])
                for ii in md0.atomic_output_def().get_data().values():
                    if ii.name == "mask":
                        self.assertEqual(ii.shape, [1])
                        self.assertFalse(ii.reduciable)
                        self.assertFalse(ii.r_differentiable)
                        self.assertFalse(ii.c_differentiable)

            # check mask
            if atom_excl == []:
                pass
            elif atom_excl == [1]:
                self.assertIn("mask", ret0.keys())
                expected = np.array([1, 1, 0], dtype=int)
                expected = np.concatenate(
                    [expected, expected[self.perm[: self.nloc]]]
                ).reshape(2, 3)
                np.testing.assert_array_equal(ret0["mask"], expected)
            else:
                raise ValueError(f"not expected atom_excl {atom_excl}")
