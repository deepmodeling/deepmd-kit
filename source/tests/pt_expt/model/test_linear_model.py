# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.descriptor import DescrptDPA1 as DPDescrptDPA1
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.dpmodel.model.make_model import (
    make_model,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.dpmodel.utils.region import (
    normalize_coord,
)
from deepmd.pt_expt.model import (
    LinearEnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestLinearModel(unittest.TestCase):
    def setUp(self) -> None:
        self.device = env.DEVICE
        self.natoms = 5
        self.rcut = 4.0
        self.rcut_smth = 0.5
        self.sel = 20
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

    def _make_dp_atomic_model(self, seed: int) -> DPAtomicModel:
        """Build a dpmodel DPAtomicModel with DPA1 descriptor (mixed type)."""
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
            seed=seed,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=seed,
        )
        return DPAtomicModel(ds, ft, type_map=self.type_map)

    def _make_dp_linear_model(self) -> LinearEnergyAtomicModel:
        """Build a dpmodel LinearEnergyAtomicModel with two sub-models."""
        model1 = self._make_dp_atomic_model(seed=GLOBAL_SEED)
        model2 = self._make_dp_atomic_model(seed=GLOBAL_SEED + 1)
        return LinearEnergyAtomicModel(
            models=[model1, model2],
            type_map=self.type_map,
        )

    def _prepare_lower_inputs(self):
        """Build extended coords, atype, nlist, mapping as torch tensors."""
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

    def test_linear_model_consistency(self) -> None:
        """Create a LinearEnergyModel, run forward() and forward_lower(),
        verify outputs have correct keys and shapes.
        """
        md_dp = self._make_dp_linear_model()
        md_pt = LinearEnergyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt.eval()

        # Test forward()
        coord = self.coord.clone().requires_grad_(True)
        ret = md_pt(coord, self.atype, self.cell.reshape(1, 9))

        self.assertIn("energy", ret)
        self.assertIn("atom_energy", ret)
        self.assertIn("force", ret)
        self.assertIn("virial", ret)

        self.assertEqual(ret["energy"].shape, (1, 1))
        self.assertEqual(ret["atom_energy"].shape, (1, self.natoms, 1))
        self.assertEqual(ret["force"].shape, (1, self.natoms, 3))
        self.assertEqual(ret["virial"].shape, (1, 9))

        # Test forward_lower()
        ext_coord, ext_atype, nlist_t, mapping_t = self._prepare_lower_inputs()
        ret_lower = md_pt.forward_lower(
            ext_coord.requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
        )

        self.assertIn("energy", ret_lower)
        self.assertIn("atom_energy", ret_lower)
        self.assertIn("extended_force", ret_lower)
        self.assertIn("virial", ret_lower)

        nall = ext_coord.shape[1]
        self.assertEqual(ret_lower["energy"].shape, (1, 1))
        self.assertEqual(ret_lower["atom_energy"].shape, (1, self.natoms, 1))
        self.assertEqual(ret_lower["extended_force"].shape, (1, nall, 3))
        self.assertEqual(ret_lower["virial"].shape, (1, 9))

    def test_linear_model_serialize(self) -> None:
        """Create a LinearEnergyModel, serialize, deserialize, verify
        outputs match.
        """
        md_dp = self._make_dp_linear_model()
        md_pt0 = LinearEnergyModel.deserialize(md_dp.serialize()).to(self.device)
        md_pt0.eval()

        # Serialize and deserialize
        md_pt1 = LinearEnergyModel.deserialize(md_pt0.serialize()).to(self.device)
        md_pt1.eval()

        coord = self.coord.clone().requires_grad_(True)
        ret0 = md_pt0(coord, self.atype, self.cell.reshape(1, 9))

        coord = self.coord.clone().requires_grad_(True)
        ret1 = md_pt1(coord, self.atype, self.cell.reshape(1, 9))

        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="energy mismatch after serialize/deserialize",
        )
        np.testing.assert_allclose(
            ret0["atom_energy"].detach().cpu().numpy(),
            ret1["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="atom_energy mismatch after serialize/deserialize",
        )
        np.testing.assert_allclose(
            ret0["force"].detach().cpu().numpy(),
            ret1["force"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="force mismatch after serialize/deserialize",
        )
        np.testing.assert_allclose(
            ret0["virial"].detach().cpu().numpy(),
            ret1["virial"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="virial mismatch after serialize/deserialize",
        )

    def test_linear_model_dpmodel_consistency(self) -> None:
        """Compare pt_expt LinearEnergyModel output with dpmodel
        LinearEnergyAtomicModel output (same weights) to verify
        cross-backend consistency.
        """
        md_dp_atomic = self._make_dp_linear_model()

        # Build pt_expt version from the same serialized data
        md_pt = LinearEnergyModel.deserialize(md_dp_atomic.serialize()).to(self.device)
        md_pt.eval()

        # Use forward_lower for both backends to compare
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

        # dpmodel forward_lower via make_model wrapper
        DPLinearModel = make_model(LinearEnergyAtomicModel)
        md_dp = DPLinearModel.deserialize(md_dp_atomic.serialize())
        ret_dp = md_dp.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
        )

        # pt_expt forward_lower
        ext_coord = torch.tensor(
            extended_coord, dtype=torch.float64, device=self.device
        )
        ext_atype = torch.tensor(extended_atype, dtype=torch.int64, device=self.device)
        nlist_t = torch.tensor(nlist, dtype=torch.int64, device=self.device)
        mapping_t = torch.tensor(mapping, dtype=torch.int64, device=self.device)
        ret_pt = md_pt.forward_lower(
            ext_coord.requires_grad_(True),
            ext_atype,
            nlist_t,
            mapping_t,
        )

        np.testing.assert_allclose(
            ret_dp["energy_redu"],
            ret_pt["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="energy mismatch between dpmodel and pt_expt",
        )
        np.testing.assert_allclose(
            ret_dp["energy"],
            ret_pt["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg="atom_energy mismatch between dpmodel and pt_expt",
        )

    def test_forward_lower_exportable(self) -> None:
        """Test that LinearEnergyModel.forward_lower_exportable returns
        an exportable module whose outputs match eager execution.
        """
        md_dp = self._make_dp_linear_model()
        md_pt = LinearEnergyModel.deserialize(md_dp.serialize()).to(self.device)
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

        # --- symbolic trace + export with dynamic shapes + .pte round-trip ---
        import tempfile

        from deepmd.pt_expt.utils.serialization import (
            _build_dynamic_shapes,
        )

        inputs_2f = tuple(
            torch.cat([t, t], dim=0) if t is not None else None
            for t in (ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
        )

        traced_sym = md_pt.forward_lower_exportable(
            inputs_2f[0],
            inputs_2f[1],
            inputs_2f[2],
            inputs_2f[3],
            fparam=inputs_2f[4],
            aparam=inputs_2f[5],
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
        )

        dynamic_shapes = _build_dynamic_shapes(*inputs_2f)
        exported_dyn = torch.export.export(
            traced_sym,
            inputs_2f,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

        with tempfile.NamedTemporaryFile(suffix=".pte") as f:
            torch.export.save(exported_dyn, f.name)
            loaded = torch.export.load(f.name).module()

        ret_loaded_1f = loaded(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)
        for key in ("atom_energy", "energy"):
            np.testing.assert_allclose(
                ret_eager[key].detach().cpu().numpy(),
                ret_loaded_1f[key].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"loaded vs eager (nf=1): {key}",
            )


if __name__ == "__main__":
    unittest.main()
