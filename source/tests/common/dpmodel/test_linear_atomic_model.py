# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel,
)
from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)


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
        extended_atype = np.array([[0, 0]])
        nlist = np.array([[[1], [-1]]])

        ds = DescrptDPA1(
            rcut_smth=0.3,
            rcut=0.4,
            sel=[3],
            ntypes=2,
        )
        ft = InvarFitting(
            "energy",
            2,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )

        type_map = ["foo", "bar"]
        zbl_model = PairTabAtomicModel(
            tab_file=file_path, rcut=0.3, sel=2, type_map=type_map
        )
        dp_model = DPAtomicModel(ds, ft, type_map=type_map)

        wgt_model = DPZBLLinearEnergyAtomicModel(
            dp_model,
            zbl_model,
            sw_rmin=0.1,
            sw_rmax=0.25,
            type_map=type_map,
        )
        wgt_res = []
        for dist in np.linspace(0.05, 0.3, 10):
            extended_coord = np.array(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, dist, 0.0],
                    ],
                ]
            )

            wgt_model.forward_atomic(extended_coord, extended_atype, nlist)

            wgt_res.append(wgt_model.zbl_weight)
        results = np.stack(wgt_res).reshape(10, 2)
        excepted_res = np.array(
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
        )
        np.testing.assert_allclose(results, excepted_res, rtol=0.0001, atol=0.0001)


class TestIntegration(unittest.TestCase):
    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut_smth = 0.4
        self.rcut = 2.2

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
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        dp_model = DPAtomicModel(ds, ft, type_map=type_map)
        zbl_model = PairTabAtomicModel(
            file_path, self.rcut, sum(self.sel), type_map=type_map
        )
        self.md0 = DPZBLLinearEnergyAtomicModel(
            dp_model,
            zbl_model,
            sw_rmin=0.1,
            sw_rmax=0.25,
            type_map=type_map,
        )
        self.md1 = DPZBLLinearEnergyAtomicModel.deserialize(self.md0.serialize())

    def test_self_consistency(self) -> None:
        ret0 = self.md0.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = self.md1.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        np.testing.assert_allclose(
            ret0["energy"],
            ret1["energy"],
        )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
