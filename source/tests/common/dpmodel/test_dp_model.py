# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
    TestCaseSingleFrameWithoutNlist,
)


class TestDPModelLower(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ) -> None:
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
        md0 = EnergyModel(ds, ft, type_map=type_map)
        md1 = EnergyModel.deserialize(md0.serialize())

        ret0 = md0.call_lower(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = md1.call_lower(self.coord_ext, self.atype_ext, self.nlist)

        np.testing.assert_allclose(ret0["energy"], ret1["energy"])
        np.testing.assert_allclose(ret0["energy_redu"], ret1["energy_redu"])

    def test_prec_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
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

        md1 = EnergyModel(ds, ft, type_map=type_map)

        args64 = [self.coord_ext, self.atype_ext, self.nlist]
        args64[0] = args64[0].astype(np.float64)
        args32 = [self.coord_ext, self.atype_ext, self.nlist]
        args32[0] = args32[0].astype(np.float32)

        model_l_ret_64 = md1.call_lower(*args64, fparam=fparam, aparam=aparam)
        model_l_ret_32 = md1.call_lower(*args32, fparam=fparam, aparam=aparam)

        for ii in model_l_ret_32.keys():
            if model_l_ret_32[ii] is None:
                continue
            if ii[-4:] == "redu":
                self.assertEqual(model_l_ret_32[ii].dtype, np.float64)
            else:
                self.assertEqual(model_l_ret_32[ii].dtype, np.float32)
            if ii != "mask":
                self.assertEqual(model_l_ret_64[ii].dtype, np.float64)
            else:
                self.assertEqual(model_l_ret_64[ii].dtype, np.int32)
            np.testing.assert_allclose(
                model_l_ret_32[ii],
                model_l_ret_64[ii],
            )


class TestDPModel(unittest.TestCase, TestCaseSingleFrameWithoutNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithoutNlist.setUp(self)

    def test_prec_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
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

        md1 = EnergyModel(ds, ft, type_map=type_map)

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
            if ii[-4:] == "redu":
                self.assertEqual(model_l_ret_32[ii].dtype, np.float64)
            else:
                self.assertEqual(model_l_ret_32[ii].dtype, np.float32)
            if ii != "mask":
                self.assertEqual(model_l_ret_64[ii].dtype, np.float64)
            else:
                self.assertEqual(model_l_ret_64[ii].dtype, np.int32)
            np.testing.assert_allclose(
                model_l_ret_32[ii],
                model_l_ret_64[ii],
            )
