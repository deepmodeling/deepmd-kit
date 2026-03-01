# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.dpmodel.fitting import PropertyFittingNet as DPPropertyFittingNet
from deepmd.dpmodel.model.property_model import PropertyModel as DPPropertyModel
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt_expt.model import (
    PropertyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestPropertyModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = env.DEVICE
        self.natoms = 5
        self.rcut = 4.0
        self.rcut_smth = 0.5
        self.sel = [8, 6]
        self.nt = 2
        self.type_map = ["foo", "bar"]

        generator = torch.Generator(device=self.device).manual_seed(GLOBAL_SEED)
        cell = torch.rand(
            [3, 3], dtype=torch.float64, device=self.device, generator=generator
        )
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device=self.device)
        self.cell = cell.unsqueeze(0)
        coord = torch.rand(
            [self.natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0).to(self.device)
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device
        )

    def _make_dp_model(self):
        ds = DPDescrptSeA(self.rcut, self.rcut_smth, self.sel)
        ft = DPPropertyFittingNet(
            self.nt,
            ds.get_dim_out(),
            task_dim=3,
            seed=GLOBAL_SEED,
        )
        return DPPropertyModel(ds, ft, type_map=self.type_map)

    def _prepare_lower_inputs(self):
        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        coord_normalized = normalize_coord(
            coord_np.reshape(1, self.natoms, 3),
            cell_np.reshape(1, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized, atype_np, cell_np, self.rcut
        )
        nlist = build_neighbor_list(
            extended_coord,
            extended_atype,
            self.natoms,
            self.rcut,
            self.sel,
            distinguish_types=True,
        )
        extended_coord = extended_coord.reshape(1, -1, 3)
        return (
            torch.tensor(extended_coord, dtype=torch.float64, device=self.device),
            torch.tensor(extended_atype, dtype=torch.int64, device=self.device),
            torch.tensor(nlist, dtype=torch.int64, device=self.device),
            torch.tensor(mapping, dtype=torch.int64, device=self.device),
        )

    def test_dp_consistency(self) -> None:
        md_dp = self._make_dp_model()
        md_pt = PropertyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        ret_dp = md_dp(coord_np.reshape(1, -1), atype_np, cell_np)

        coord = self.coord.clone().requires_grad_(True)
        ret_pt = md_pt(coord, self.atype, self.cell.reshape(1, 9))

        var_name = md_pt.get_var_name()
        np.testing.assert_allclose(
            ret_dp[var_name],
            ret_pt[var_name].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret_dp[f"atom_{var_name}"],
            ret_pt[f"atom_{var_name}"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_output_keys(self) -> None:
        md_dp = self._make_dp_model()
        md_pt = PropertyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()
        coord = self.coord.clone().requires_grad_(True)
        ret = md_pt(coord, self.atype, self.cell.reshape(1, 9))
        var_name = md_pt.get_var_name()
        self.assertIn(var_name, ret)
        self.assertIn(f"atom_{var_name}", ret)

    def test_forward_lower_exportable(self) -> None:
        md_dp = self._make_dp_model()
        md_pt = PropertyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        ext_coord, ext_atype, nlist_t, mapping_t = self._prepare_lower_inputs()
        fparam = None
        aparam = None

        ret_eager = md_pt.forward_lower(
            ext_coord.requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
        )

        traced = md_pt.forward_lower_exportable(
            ext_coord,
            ext_atype,
            nlist_t,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
        )
        self.assertIsInstance(traced, torch.nn.Module)

        exported = torch.export.export(
            traced,
            (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam),
            strict=False,
        )
        self.assertIsNotNone(exported)

        ret_traced = traced(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
        ret_exported = exported.module()(
            ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam
        )

        var_name = md_pt.get_var_name()
        for key in (f"atom_{var_name}", var_name):
            np.testing.assert_allclose(
                ret_eager[key].detach().cpu().numpy(),
                ret_traced[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"traced vs eager: {key}",
            )
            np.testing.assert_allclose(
                ret_eager[key].detach().cpu().numpy(),
                ret_exported[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"exported vs eager: {key}",
            )


if __name__ == "__main__":
    unittest.main()
