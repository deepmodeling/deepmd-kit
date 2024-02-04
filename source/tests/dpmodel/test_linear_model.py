# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.model.linear_model import LinearModel as LinearModel
from deepmd.dpmodel.model.pair_tab_model import (
    PairTabModel,
)

class TestWeightCalculation(unittest.TestCase):
    @patch("numpy.loadtxt")
    def test_pairwise(self, mock_loadtxt):
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

        ds = DescrptSeA(
            rcut=0.3,
            rcut_smth=0.4,
            sel=[3],
        )
        ft = InvarFitting(
            "energy",
            2,
            ds.get_dim_out(),
            1,
            distinguish_types=ds.distinguish_types(),
        )

        type_map = ["foo", "bar"]
        zbl_model = PairTabModel(tab_file=file_path, rcut=0.3, sel=2)
        dp_model = DPAtomicModel(ds, ft, type_map=type_map)

        wgt_model = LinearModel(dp_model, zbl_model)
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

            wgt_model.forward_atomic(
                extended_coord, extended_atype, nlist, ra=0.1, rb=0.25
            )

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
    def setUp(self, mock_loadtxt):
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
        self.rcut = 0.4
        self.rcut_smth = 2.2

        file_path = "dummy_path"
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0],
                [0.01, 0.8, 1.6, 2.4],
                [0.015, 0.5, 1.0, 1.5],
                [0.02, 0.25, 0.4, 0.75],
            ]
        )
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
            distinguish_types=ds.distinguish_types(),
        )
        type_map = ["foo", "bar"]
        dp_model = DPAtomicModel(ds, ft, type_map=type_map)
        zbl_model = PairTabModel(file_path, self.rcut, sum(self.sel))
        self.md0 = LinearModel(dp_model, zbl_model)
        self.md1 = LinearModel.deserialize(self.md0.serialize())

    def test_self_consistency(self):
        nlist_copy = self.nlist.copy()
        ret0 = self.md0.forward_atomic(
            self.coord_ext, self.atype_ext, self.nlist, ra=0.2, rb=0.5
        )
        ret1 = self.md1.forward_atomic(
            self.coord_ext, self.atype_ext, nlist_copy, ra=0.2, rb=0.5
        )
        np.testing.assert_allclose(
            ret0["energy"],
            ret1["energy"],
        )
