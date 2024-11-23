# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import (
    DPZBLLinearEnergyAtomicModel as DPDPZBLLinearEnergyAtomicModel,
)
from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    DPZBLLinearEnergyAtomicModel,
    PairTabAtomicModel,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
)
from deepmd.pt.model.model import (
    DPZBLModel,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestWeightCalculation(unittest.TestCase):
    @patch("numpy.loadtxt")
    def test_pairwise(self, mock_loadtxt) -> None:
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.05, 1.0, 2.0, 3.0],
                [0.1, 0.8, 1.6, 2.4],
                [0.15, 0.5, 1.0, 1.5],
                [0.2, 0.25, 0.4, 0.75],
                [0.25, 0.0, 0.0, 0.0],
            ]
        )
        extended_atype = torch.tensor([[0, 0]], device=env.DEVICE)
        nlist = torch.tensor([[[1], [-1]]], device=env.DEVICE)

        ds = DescrptDPA1(
            rcut_smth=0.3,
            rcut=0.4,
            sel=[3],
            ntypes=2,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            2,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)

        type_map = ["foo", "bar"]
        zbl_model = PairTabAtomicModel(
            tab_file=file_path, rcut=0.3, sel=2, type_map=type_map[::-1]
        )
        dp_model = DPAtomicModel(ds, ft, type_map=type_map).to(env.DEVICE)
        wgt_model = DPZBLLinearEnergyAtomicModel(
            dp_model,
            zbl_model,
            sw_rmin=0.1,
            sw_rmax=0.25,
            type_map=type_map,
        ).to(env.DEVICE)
        wgt_res = []
        for dist in np.linspace(0.05, 0.3, 10):
            extended_coord = torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, dist, 0.0],
                    ],
                ],
                device=env.DEVICE,
            )

            wgt_model.forward_atomic(extended_coord, extended_atype, nlist)

            wgt_res.append(wgt_model.zbl_weight)
        results = torch.stack(wgt_res).reshape(10, 2)
        excepted_res = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.9995, 0.0],
                [0.9236, 0.0],
                [0.6697, 0.0],
                [0.3303, 0.0],
                [0.0764, 0.0],
                [0.0005, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=torch.float64,
            device=env.DEVICE,
        )
        torch.testing.assert_close(results, excepted_res, rtol=0.0001, atol=0.0001)


class TestIntegration(unittest.TestCase, TestCaseSingleFrameWithNlist):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )
        ds = DescrptDPA1(
            self.rcut,
            self.rcut_smth,
            sum(self.sel),
            self.nt,
        ).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        dp_model = DPAtomicModel(ds, ft, type_map=type_map).to(env.DEVICE)
        zbl_model = PairTabAtomicModel(
            file_path, self.rcut, sum(self.sel), type_map=type_map
        )
        self.md0 = DPZBLLinearEnergyAtomicModel(
            dp_model,
            zbl_model,
            sw_rmin=0.1,
            sw_rmax=0.25,
            type_map=type_map,
        ).to(env.DEVICE)
        self.md1 = DPZBLLinearEnergyAtomicModel.deserialize(self.md0.serialize()).to(
            env.DEVICE
        )
        self.md2 = DPDPZBLLinearEnergyAtomicModel.deserialize(self.md0.serialize())
        self.md3 = DPZBLModel(
            dp_model, zbl_model, sw_rmin=0.1, sw_rmax=0.25, type_map=type_map
        )

    def test_self_consistency(self) -> None:
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        ret0 = self.md0.forward_atomic(*args)
        ret1 = self.md1.forward_atomic(*args)
        ret2 = self.md2.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]),
            to_numpy_array(ret1["energy"]),
        )

        np.testing.assert_allclose(
            to_numpy_array(ret0["energy"]), ret2["energy"], atol=0.001, rtol=0.001
        )

    def test_jit(self) -> None:
        md1 = torch.jit.script(self.md1)
        # atomic model no more export methods
        # self.assertEqual(md1.get_rcut(), self.rcut)
        # self.assertEqual(md1.get_type_map(), ["foo", "bar"])
        md3 = torch.jit.script(self.md3)
        # atomic model no more export methods
        # self.assertEqual(md3.get_rcut(), self.rcut)
        # self.assertEqual(md3.get_type_map(), ["foo", "bar"])


class TestRemmapMethod(unittest.TestCase):
    def test_valid(self) -> None:
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        atype = torch.randint(0, 3, (4, 20), device=env.DEVICE, generator=generator)
        commonl = ["H", "O", "S"]
        originl = ["Si", "H", "O", "S"]
        mapping = DPZBLLinearEnergyAtomicModel.remap_atype(originl, commonl)
        new_atype = mapping[atype]

        def trans(atype, map):
            idx = atype.flatten().tolist()
            res = []
            for i in idx:
                res.append(map[i])
            return res

        assert trans(atype, commonl) == trans(new_atype, originl)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
