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
    LinearEnergyAtomicModel,
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


class _RecordingAtomicModel:
    """Minimal mixed-type submodel that records the atom types it receives."""

    def __init__(self, type_map: list[str]) -> None:
        self.type_map = type_map
        self.received_atype: list[np.ndarray] = []

    def mixed_types(self) -> bool:
        return True

    def get_type_map(self) -> list[str]:
        return self.type_map

    def change_type_map(
        self,
        type_map: list[str],
        model_with_new_type_stat: object | None = None,
    ) -> None:
        self.type_map = type_map

    def get_rcut(self) -> float:
        return 2.0

    def get_nsel(self) -> int:
        return 1

    def get_sel(self) -> list[int]:
        return [1]

    def forward_atomic(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlist: np.ndarray,
        *args: object,
        **kwargs: object,
    ) -> dict[str, np.ndarray]:
        self.received_atype.append(extended_atype.copy())
        return {"energy": np.zeros((*nlist.shape[:2], 1), dtype=extended_coord.dtype)}


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


class TestLinearWeights(unittest.TestCase):
    """Test that the weights parameter is honored by LinearEnergyAtomicModel."""

    def setUp(self) -> None:
        from deepmd.dpmodel.atomic_model.linear_atomic_model import (
            LinearEnergyAtomicModel,
        )

        self.nloc = 3
        self.nall = 4
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
        sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(sel)])
        rcut_smth = 0.4
        rcut = 2.2
        nt = 2

        ds1 = DescrptDPA1(rcut, rcut_smth, sum(sel), nt, seed=1)
        ft1 = InvarFitting(
            "energy", nt, ds1.get_dim_out(), 1, mixed_types=ds1.mixed_types(), seed=1
        )
        ds2 = DescrptDPA1(rcut, rcut_smth, sum(sel), nt, seed=2)
        ft2 = InvarFitting(
            "energy", nt, ds2.get_dim_out(), 1, mixed_types=ds2.mixed_types(), seed=2
        )

        type_map = ["foo", "bar"]
        m1 = DPAtomicModel(ds1, ft1, type_map=type_map)
        m2 = DPAtomicModel(ds2, ft2, type_map=type_map)

        # Get individual sub-model predictions for reference
        ret1 = m1.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        ret2 = m2.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        self.ener1 = ret1["energy"]
        self.ener2 = ret2["energy"]

        self.md_mean = LinearEnergyAtomicModel(
            models=[m1, m2], type_map=type_map, weights="mean"
        )
        self.md_sum = LinearEnergyAtomicModel(
            models=[m1, m2], type_map=type_map, weights="sum"
        )
        self.md_custom = LinearEnergyAtomicModel(
            models=[m1, m2], type_map=type_map, weights=[0.3, 0.7]
        )

    def test_mean_weights(self) -> None:
        ret = self.md_mean.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        expected = 0.5 * self.ener1 + 0.5 * self.ener2
        np.testing.assert_allclose(ret["energy"], expected)

    def test_sum_weights(self) -> None:
        ret = self.md_sum.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        expected = self.ener1 + self.ener2
        np.testing.assert_allclose(ret["energy"], expected)

    def test_custom_weights(self) -> None:
        ret = self.md_custom.forward_atomic(self.coord_ext, self.atype_ext, self.nlist)
        expected = 0.3 * self.ener1 + 0.7 * self.ener2
        np.testing.assert_allclose(ret["energy"], expected)


class TestChangeTypeMap(unittest.TestCase):
    def test_missing_submodel_type_raises_validation_error(self) -> None:
        """Unsupported common types should produce an actionable ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            r"contains types \['bar'\].*not supported by submodel type_map \['foo'\]",
        ):
            LinearEnergyAtomicModel(
                models=[_RecordingAtomicModel(["foo"])],
                type_map=["foo", "bar"],
                weights="sum",
            )

    def test_rebuilds_submodel_type_mappings(self) -> None:
        """Runtime type IDs must follow the submodels after a map change."""
        submodels = [
            _RecordingAtomicModel(["bar", "foo"]),
            _RecordingAtomicModel(["bar", "foo"]),
        ]
        model = LinearEnergyAtomicModel(
            models=submodels,
            type_map=["foo", "bar"],
            weights="sum",
        )

        # Reorder existing species and add one.  The old two-entry mapping would
        # both swap foo/bar and fail when the new type ID 2 is encountered.
        new_type_map = ["bar", "foo", "baz"]
        model.change_type_map(new_type_map)
        coord = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        atype = np.array([[0, 1, 2]], dtype=np.int64)
        nlist = np.array([[[1], [0], [1]]], dtype=np.int64)

        model.forward_atomic(coord, atype, nlist)

        for mapping, submodel in zip(model.mapping_list, submodels, strict=True):
            np.testing.assert_array_equal(mapping, [0, 1, 2])
            np.testing.assert_array_equal(submodel.received_atype[-1], atype)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
