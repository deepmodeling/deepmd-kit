# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
)
from deepmd.pt.utils import (
    PairExcludeMask,
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class DescrptExcludeTypes(TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)
        self.input_dict = {
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
        }

    def test_exclude_types(
        self,
    ):
        dtype = PRECISION_DICT["float64"]
        rtol, atol = get_tols("float64")
        coord_ext_device = torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE)
        atype_ext_device = torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE)
        nlist_device = torch.tensor(self.nlist, dtype=int, device=env.DEVICE)
        mapping_device = torch.tensor(self.mapping, dtype=int, device=env.DEVICE)
        dd = self.descrpt(**self.input_dict).to(env.DEVICE)

        for em in [[[0, 1]], [[1, 1]]]:
            dd0 = self.descrpt(**self.input_dict, exclude_types=em).to(env.DEVICE)
            # only equal when set_davg_zero is True
            # dd0.sea.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            # dd0.sea.dstd = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd.load_state_dict(dd0.state_dict())

            ex_pair = PairExcludeMask(self.nt, em)
            pair_mask = ex_pair(nlist_device, atype_ext_device)
            # exclude neighbors in the nlist
            nlist_exclude = torch.where(pair_mask == 1, nlist_device, -1)

            rd0, _, _, _, sw0 = dd0(
                coord_ext_device,
                atype_ext_device,
                nlist_device,
                mapping=mapping_device,
            )

            rd_ex, _, _, _, sw_ex = dd(
                coord_ext_device,
                atype_ext_device,
                nlist_exclude,
                mapping=mapping_device,
            )

            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd_ex.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )


class TestDescrptExcludeTypesSeA(DescrptExcludeTypes, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.descrpt = DescrptSeA


class TestDescrptExcludeTypesSeR(DescrptExcludeTypes, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.descrpt = DescrptSeR


class TestDescrptExcludeTypesSeT(DescrptExcludeTypes, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.descrpt = DescrptSeT


class TestDescrptExcludeTypesDPA1(DescrptExcludeTypes, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.descrpt = DescrptDPA1


class TestDescrptExcludeTypesDPA2(DescrptExcludeTypes, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.descrpt = DescrptDPA2
        self.input_dict = {
            "ntypes": self.nt,
            "repinit": {
                "rcut": self.rcut,
                "rcut_smth": self.rcut_smth,
                "nsel": self.sel_mix,
            },
            "repformer": {
                "rcut": self.rcut / 2,
                "rcut_smth": self.rcut_smth,
                "nsel": self.sel_mix[0] // 2,
            },
        }


class TestDescrptExcludeTypesHybrid(DescrptExcludeTypes, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.descrpt = DescrptHybrid
        ddsub0 = {
            "type": "se_e2_a",
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel,
        }
        ddsub1 = {
            "type": "dpa1",
            "ntypes": self.nt,
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
            "sel": self.sel_mix,
        }
        self.input_dict = {
            "list": [ddsub0, ddsub1],
        }
