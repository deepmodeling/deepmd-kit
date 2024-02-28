# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.common import (
    RESERVED_PRECISON_DICT,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model import (
    DPModel,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
    TestCaseSingleFrameWithoutNlist,
)


class TestDPModelLower(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        nf, nloc, nnei = self.nlist.shape
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
        md0 = DPModel(ds, ft, type_map=type_map)
        md1 = DPModel.deserialize(md0.serialize())

        ret0 = md0.call_lower(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = md1.call_lower(self.coord_ext, self.atype_ext, self.nlist)

        np.testing.assert_allclose(ret0["energy"], ret1["energy"])
        np.testing.assert_allclose(ret0["energy_redu"], ret1["energy_redu"])

    def test_prec_consistency(self):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
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
        nfp, nap = 2, 3
        type_map = ["foo", "bar"]
        # fparam, aparam are converted to coordinate precision by model
        fparam = rng.normal(size=[self.nf, nfp])
        aparam = rng.normal(size=[self.nf, nloc, nap])

        md1 = DPModel(ds, ft, type_map=type_map)

        args64 = [self.coord_ext, self.atype_ext, self.nlist]
        args64[0] = args64[0].astype(np.float64)
        args32 = [self.coord_ext, self.atype_ext, self.nlist]
        args32[0] = args32[0].astype(np.float32)

        model_l_ret_64 = md1.call_lower(*args64, fparam=fparam, aparam=aparam)
        model_l_ret_32 = md1.call_lower(*args32, fparam=fparam, aparam=aparam)

        for ii in model_l_ret_32.keys():
            if model_l_ret_32[ii] is None:
                continue
            self.assertEqual(
                model_l_ret_32[ii].dtype.name, RESERVED_PRECISON_DICT[np.float32]
            )
            np.testing.assert_allclose(
                model_l_ret_32[ii],
                model_l_ret_64[ii],
            )


class TestDPModel(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self):
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_prec_consistency(self):
        rng = np.random.default_rng()
        nf, nloc = self.atype.shape
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
        nfp, nap = 2, 3
        type_map = ["foo", "bar"]
        # fparam, aparam are converted to coordinate precision by model
        fparam = rng.normal(size=[self.nf, nfp])
        aparam = rng.normal(size=[self.nf, nloc, nap])

        md1 = DPModel(ds, ft, type_map=type_map)

        args64 = [self.coord, self.atype, self.cell]
        args64[0] = args64[0].astype(np.float64)
        args64[2] = args64[2].astype(np.float64)
        args32 = [self.coord, self.atype, self.cell]
        args32[0] = args32[0].astype(np.float32)
        args32[2] = args32[2].astype(np.float32)

        model_l_ret_64 = md1.call(*args64, fparam=fparam, aparam=aparam)
        model_l_ret_32 = md1.call(*args32, fparam=fparam, aparam=aparam)

        for ii in model_l_ret_32.keys():
            if model_l_ret_32[ii] is None:
                continue
            self.assertEqual(model_l_ret_32[ii].dtype, np.float32)
            np.testing.assert_allclose(
                model_l_ret_32[ii],
                model_l_ret_64[ii],
            )
