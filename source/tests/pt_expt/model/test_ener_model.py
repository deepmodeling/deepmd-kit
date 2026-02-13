# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.dpmodel.model.ener_model import EnergyModel as DPEnergyModel
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestEnergyModel(unittest.TestCase):
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
        self.cell = cell.unsqueeze(0)  # [1, 3, 3]
        coord = torch.rand(
            [self.natoms, 3],
            dtype=torch.float64,
            device=self.device,
            generator=generator,
        )
        coord = torch.matmul(coord, cell)
        self.coord = coord.unsqueeze(0).to(self.device)  # [1, natoms, 3]
        self.atype = torch.tensor(
            [[0, 0, 0, 1, 1]], dtype=torch.int64, device=self.device
        )

    def _make_model(self) -> EnergyModel:
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(self.device)
        return EnergyModel(ds, ft, type_map=self.type_map).to(self.device)

    def test_output_keys(self) -> None:
        """Test that EnergyModel produces expected output keys."""
        md = self._make_model()
        md.eval()
        coord = self.coord.clone().requires_grad_(True)
        ret = md(coord, self.atype, self.cell.reshape(1, 9))
        self.assertIn("energy", ret)
        self.assertIn("atom_energy", ret)
        self.assertIn("force", ret)
        self.assertIn("virial", ret)

    def test_output_shapes(self) -> None:
        """Test that output shapes are correct."""
        md = self._make_model()
        md.eval()
        coord = self.coord.clone().requires_grad_(True)
        ret = md(coord, self.atype, self.cell.reshape(1, 9))
        self.assertEqual(ret["energy"].shape, (1, 1))
        self.assertEqual(ret["atom_energy"].shape, (1, self.natoms, 1))
        self.assertEqual(ret["force"].shape, (1, self.natoms, 3))
        self.assertEqual(ret["virial"].shape, (1, 9))

    def test_forward_lower_exportable(self) -> None:
        """Test that EnergyModel.forward_lower returns an exportable module.

        forward_lower() uses make_fx to trace through torch.autograd.grad,
        decomposing the backward pass into primitive ops. The returned module
        can be passed directly to torch.export.export.
        """
        md = self._make_model()
        md.eval()

        # Prepare extended coords and neighbor list using dpmodel utilities
        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        coord_normalized = normalize_coord(
            coord_np.reshape(1, self.natoms, 3),
            cell_np.reshape(1, 3, 3),
        )
        extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
            coord_normalized,
            atype_np,
            cell_np,
            self.rcut,
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

        # Convert to torch tensors
        ext_coord = torch.tensor(
            extended_coord,
            dtype=torch.float64,
            device=self.device,
        )
        ext_atype = torch.tensor(
            extended_atype,
            dtype=torch.int64,
            device=self.device,
        )
        nlist_t = torch.tensor(nlist, dtype=torch.int64, device=self.device)
        mapping_t = torch.tensor(mapping, dtype=torch.int64, device=self.device)

        # Eager reference via _forward_lower
        ret0 = md._forward_lower(
            ext_coord.requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
            do_atomic_virial=True,
        )
        self.assertIn("energy", ret0)
        self.assertIn("extended_force", ret0)
        self.assertIn("virial", ret0)
        self.assertIn("extended_virial", ret0)

        # forward_lower returns a traced module
        traced = md.forward_lower(
            ext_coord,
            ext_atype,
            nlist_t,
            mapping_t,
            do_atomic_virial=True,
        )
        self.assertIsInstance(traced, torch.nn.Module)

        # The traced module should be directly exportable
        exported = torch.export.export(
            traced,
            (ext_coord, ext_atype, nlist_t, mapping_t),
            strict=False,
        )
        self.assertIsNotNone(exported)

        # Verify exported model produces same output
        ret1 = exported.module()(ext_coord, ext_atype, nlist_t, mapping_t)
        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret0["extended_force"].detach().cpu().numpy(),
            ret1["extended_force"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret0["virial"].detach().cpu().numpy(),
            ret1["virial"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret0["extended_virial"].detach().cpu().numpy(),
            ret1["extended_virial"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_dp_consistency(self) -> None:
        """Test numerical consistency with dpmodel (energy values)."""
        # Build dpmodel version
        ds_dp = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft_dp = DPInvarFitting(
            "energy",
            self.nt,
            ds_dp.get_dim_out(),
            1,
            mixed_types=ds_dp.mixed_types(),
            seed=GLOBAL_SEED,
        )
        md_dp = DPEnergyModel(ds_dp, ft_dp, type_map=self.type_map)

        # Build pt_expt version from serialized dpmodel
        md_pt = EnergyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        # dpmodel inference
        coord_np = self.coord.detach().cpu().numpy()
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        ret_dp = md_dp(coord_np.reshape(1, -1), atype_np, cell_np)

        # pt_expt inference
        coord = self.coord.clone().requires_grad_(True)
        ret_pt = md_pt(coord, self.atype, self.cell.reshape(1, 9))

        np.testing.assert_allclose(
            ret_dp["energy_redu"],
            ret_pt["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret_dp["energy"],
            ret_pt["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
