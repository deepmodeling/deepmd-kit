# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.dpmodel.model.ener_model import EnergyModel as DPEnergyModel
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

    @unittest.expectedFailure
    def test_exportable(self) -> None:
        """Test that EnergyModel can be exported with torch.export.

        Currently expected to fail because the full model's call() path includes
        extend_coord_with_ghosts and neighbor list building, which involve
        data-dependent shapes (item() calls) that torch.export cannot trace.
        Individual components (descriptor, fitting, atomic model) are exportable.
        """
        md = self._make_model()
        md.eval()
        coord = self.coord.clone().requires_grad_(True)
        cell = self.cell.reshape(1, 9)

        # Test forward pass
        ret0 = md(coord, self.atype, cell)
        self.assertIn("energy", ret0)

        # Test torch.export
        exported = torch.export.export(
            md,
            (coord, self.atype, cell),
            strict=False,
        )
        self.assertIsNotNone(exported)

        # Test exported model produces same output
        coord2 = self.coord.clone().requires_grad_(True)
        ret1 = exported.module()(coord2, self.atype, cell)
        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            ret0["force"].detach().cpu().numpy(),
            ret1["force"].detach().cpu().numpy(),
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
