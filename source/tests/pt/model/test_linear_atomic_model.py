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
    LinearEnergyAtomicModel,
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


class _RecordingAtomicModel(torch.nn.Module):
    """Minimal torch submodel that records remapped runtime atom types."""

    def __init__(self, type_map: list[str]) -> None:
        super().__init__()
        self.type_map = type_map
        self.received_atype: list[torch.Tensor] = []

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

    def forward_common_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.received_atype.append(extended_atype.detach().clone())
        return {
            "energy": torch.zeros(
                (*nlist.shape[:2], 1),
                dtype=extended_coord.dtype,
                device=extended_coord.device,
            )
        }


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

    def test_forward_atomic_accepts_leaf_view_input(self) -> None:
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        coord = args[0]
        coord_view = coord.view(self.nf, self.nall, 3)
        coord_view_before = coord_view.detach().clone()
        self.assertTrue(coord.is_leaf)
        self.assertTrue(coord_view._is_view())
        args[0] = coord_view
        ret = self.md0.forward_atomic(*args)

        self.assertFalse(coord_view.requires_grad)
        torch.testing.assert_close(coord_view, coord_view_before)
        self.assertIn("energy", ret)

    def test_forward_atomic_preserves_grad_enabled_input(self) -> None:
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        args[0] = args[0].view(self.nf, self.nall, 3).clone().requires_grad_(True)
        ret = self.md0.forward_atomic(*args)
        ret["energy"].sum().backward()

        self.assertTrue(args[0].requires_grad)
        self.assertIsNotNone(args[0].grad)

    def test_jit(self) -> None:
        md1 = torch.jit.script(self.md1)
        # atomic model no more export methods
        # self.assertEqual(md1.get_rcut(), self.rcut)
        # self.assertEqual(md1.get_type_map(), ["foo", "bar"])
        md3 = torch.jit.script(self.md3)
        # atomic model no more export methods
        # self.assertEqual(md3.get_rcut(), self.rcut)
        # self.assertEqual(md3.get_type_map(), ["foo", "bar"])

    def test_forward_embedding(self) -> None:
        # The embedding of a DP+ZBL model is taken from its DP sub-model.
        self.assertTrue(self.md0.has_embedding())
        args = [
            to_torch_tensor(ii) for ii in [self.coord_ext, self.atype_ext, self.nlist]
        ]
        emb = self.md0.forward_embedding(*args)
        for key in ("descriptor", "atomic_feature", "structural_feature"):
            self.assertIn(key, emb)
        self.assertEqual(tuple(emb["descriptor"].shape[:2]), (self.nf, self.nloc))
        self.assertEqual(tuple(emb["atomic_feature"].shape[:2]), (self.nf, self.nloc))
        self.assertEqual(
            tuple(emb["structural_feature"].shape),
            (self.nf, emb["atomic_feature"].shape[2]),
        )


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

    def test_change_type_map_rebuilds_mapping(self) -> None:
        submodels = [
            _RecordingAtomicModel(["bar", "foo"]),
            _RecordingAtomicModel(["bar", "foo"]),
        ]
        model = LinearEnergyAtomicModel(
            models=submodels,
            type_map=["foo", "bar"],
            weights="sum",
        ).to(env.DEVICE)

        new_type_map = ["bar", "foo", "baz"]
        model.change_type_map(new_type_map)
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
            dtype=dtype,
            device=env.DEVICE,
        )
        atype = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=env.DEVICE)
        nlist = torch.tensor([[[1], [0], [1]]], dtype=torch.int64, device=env.DEVICE)

        model.forward_atomic(coord, atype, nlist)

        for mapping, submodel in zip(model.mapping_list, submodels, strict=True):
            torch.testing.assert_close(
                mapping,
                torch.tensor([0, 1, 2], dtype=mapping.dtype, device=env.DEVICE),
            )
            torch.testing.assert_close(
                submodel.received_atype[-1],
                atype.to(dtype=submodel.received_atype[-1].dtype),
            )


if __name__ == "__main__":
    unittest.main(warnings="ignore")
