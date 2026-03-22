# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.dpmodel.descriptor import DescrptDPA1 as DPDescrptDPA1
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.dpmodel.model.dp_zbl_model import DPZBLModel as DPDPZBLModel
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt_expt.model import (
    DPZBLModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TAB_FILE = os.path.join(
    TESTS_DIR,
    "pt",
    "model",
    "water",
    "data",
    "zbl_tab_potential",
    "H2O_tab_potential.txt",
)


class TestDPZBLModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = env.DEVICE
        self.natoms = 5
        self.rcut = 4.0
        self.rcut_smth = 0.5
        self.sel = 20
        self.nt = 3
        self.type_map = ["O", "H", "B"]

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
            [[0, 0, 1, 1, 2]], dtype=torch.int64, device=self.device
        )

    def _make_dp_model(self):
        ds = DPDescrptDPA1(
            rcut_smth=self.rcut_smth,
            rcut=self.rcut,
            sel=self.sel,
            ntypes=self.nt,
            neuron=[3, 6],
            axis_neuron=2,
            attn=4,
            attn_layer=2,
            attn_dotr=True,
            attn_mask=False,
            activation_function="tanh",
            set_davg_zero=True,
            type_one_side=True,
            seed=GLOBAL_SEED,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        dp_model = DPAtomicModel(ds, ft, type_map=self.type_map)
        zbl_model = PairTabAtomicModel(
            tab_file=TAB_FILE,
            rcut=self.rcut,
            sel=self.sel,
            type_map=self.type_map,
        )
        return DPDPZBLModel(
            dp_model,
            zbl_model,
            sw_rmin=0.2,
            sw_rmax=4.0,
            type_map=self.type_map,
        )

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
            [self.sel],
            distinguish_types=False,
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
        md_pt = DPZBLModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        ret_dp = md_dp(coord_np.reshape(1, -1), atype_np, cell_np)

        coord = self.coord.clone().requires_grad_(True)
        ret_pt = md_pt(coord, self.atype, self.cell.reshape(1, 9))

        np.testing.assert_allclose(
            ret_dp["energy"],
            ret_pt["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret_dp["atom_energy"],
            ret_pt["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_output_keys(self) -> None:
        md_dp = self._make_dp_model()
        md_pt = DPZBLModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()
        coord = self.coord.clone().requires_grad_(True)
        ret = md_pt(coord, self.atype, self.cell.reshape(1, 9))
        self.assertIn("energy", ret)
        self.assertIn("atom_energy", ret)
        self.assertIn("force", ret)
        self.assertIn("virial", ret)

    def test_forward_lower_exportable(self) -> None:
        md_dp = self._make_dp_model()
        md_pt = DPZBLModel.deserialize(md_dp.serialize()).to(self.device)
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

        for key in ("atom_energy", "energy"):
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
