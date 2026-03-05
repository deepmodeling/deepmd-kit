# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pt_expt inference via the DeepPot / DeepEval interface.

Verifies the full pipeline:
    model.serialize() → deserialize_to_file(.pte) → DeepPot(.pte) → eval()
"""

import importlib
import tempfile
import unittest

import numpy as np
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.serialization import (
    _make_sample_inputs,
    deserialize_to_file,
    serialize_from_file,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestDeepEvalEner(unittest.TestCase):
    """Test pt_expt inference for energy models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]

        # Build pt_expt model
        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        cls.model = EnergyModel(ds, ft, type_map=cls.type_map)
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()

        # Serialize and save to .pte
        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.tmpfile.close()
        deserialize_to_file(cls.tmpfile.name, cls.model_data)

        # Create DeepPot for testing
        cls.dp = DeepPot(cls.tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.unlink(cls.tmpfile.name)

    def test_get_rcut(self) -> None:
        self.assertAlmostEqual(self.dp.deep_eval.get_rcut(), self.rcut)

    def test_get_ntypes(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_ntypes(), self.nt)

    def test_get_type_map(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_type_map(), self.type_map)

    def test_get_dim_fparam(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_dim_fparam(), 0)

    def test_get_dim_aparam(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_dim_aparam(), 0)

    def test_get_sel_type(self) -> None:
        sel_type = self.dp.deep_eval.get_sel_type()
        self.assertEqual(sel_type, self.model.get_sel_type())

    def test_model_type(self) -> None:
        self.assertIs(self.dp.deep_eval.model_type, DeepPot)

    def test_get_model(self) -> None:
        mod = self.dp.deep_eval.get_model()
        self.assertIsInstance(mod, torch.nn.Module)

    def test_get_model_def_script(self) -> None:
        mds = self.dp.deep_eval.get_model_def_script()
        self.assertIsInstance(mds, dict)
        self.assertEqual(mds["type_map"], self.type_map)
        self.assertAlmostEqual(mds["rcut"], self.rcut)
        self.assertEqual(mds["sel"], list(self.sel))

    def test_eval_consistency(self) -> None:
        """Test that DeepPot.eval gives same results as direct model forward."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        # .pte inference
        e, f, v, ae, av = self.dp.eval(coords, cells, atom_types, atomic=True)

        # Direct model forward
        coord_t = torch.tensor(
            coords, dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype_t = torch.tensor(
            atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
        )
        cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
        ref = self.model.forward(coord_t, atype_t, cell_t, do_atomic_virial=True)

        np.testing.assert_allclose(
            e, ref["energy"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            f, ref["force"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            v, ref["virial"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            ae, ref["atom_energy"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            av, ref["atom_virial"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )

    def test_multiple_frames(self) -> None:
        """Test evaluation with multiple frames."""
        rng = np.random.default_rng(GLOBAL_SEED + 7)
        natoms = 4
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        for nframes in [2, 5]:
            coords = rng.random((nframes, natoms, 3)) * 8.0
            cells = np.tile(np.eye(3).reshape(1, 9) * 10.0, (nframes, 1))

            e, f, v, ae, av = self.dp.eval(coords, cells, atom_types, atomic=True)

            coord_t = torch.tensor(
                coords, dtype=torch.float64, device=DEVICE
            ).requires_grad_(True)
            atype_t = torch.tensor(
                np.tile(atom_types, (nframes, 1)), dtype=torch.int64, device=DEVICE
            )
            cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
            ref = self.model.forward(coord_t, atype_t, cell_t, do_atomic_virial=True)

            np.testing.assert_allclose(
                e,
                ref["energy"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, energy",
            )
            np.testing.assert_allclose(
                f,
                ref["force"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, force",
            )
            np.testing.assert_allclose(
                v,
                ref["virial"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, virial",
            )
            np.testing.assert_allclose(
                ae,
                ref["atom_energy"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, atom_energy",
            )
            np.testing.assert_allclose(
                av,
                ref["atom_virial"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, atom_virial",
            )

    def test_dynamic_shapes(self) -> None:
        """Test that the exported model handles different atom counts.

        Compares exported module output against direct forward_common_lower
        for multiple nloc values.
        """
        extra_files = {"model_def_script.json": ""}
        exported = torch.export.load(self.tmpfile.name, extra_files=extra_files)
        exported_mod = exported.module()

        for nloc in [2, 5, 10]:
            ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = (
                _make_sample_inputs(self.model, nloc=nloc)
            )

            pte_ret = exported_mod(
                ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam
            )

            ec = ext_coord.detach().requires_grad_(True)
            ref_ret = self.model.forward_common_lower(
                ec,
                ext_atype,
                nlist_t,
                mapping_t,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=True,
            )

            for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_c"):
                if ref_ret[key] is not None and key in pte_ret:
                    np.testing.assert_allclose(
                        ref_ret[key].detach().cpu().numpy(),
                        pte_ret[key].detach().cpu().numpy(),
                        rtol=1e-10,
                        atol=1e-10,
                        err_msg=f"nloc={nloc}, key={key}",
                    )

    def test_serialize_round_trip(self) -> None:
        """Test .pte → serialize_from_file → deserialize → model gives same outputs."""
        loaded_data = serialize_from_file(self.tmpfile.name)

        model2 = EnergyModel.deserialize(loaded_data["model"])
        model2 = model2.to(torch.float64)
        model2.eval()

        for nloc in [3, 7]:
            ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = (
                _make_sample_inputs(self.model, nloc=nloc)
            )
            ec1 = ext_coord.detach().requires_grad_(True)
            ec2 = ext_coord.detach().requires_grad_(True)

            ret1 = self.model.forward_common_lower(
                ec1,
                ext_atype,
                nlist_t,
                mapping_t,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=True,
            )
            ret2 = model2.forward_common_lower(
                ec2,
                ext_atype,
                nlist_t,
                mapping_t,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=True,
            )

            for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_c"):
                if ret1[key] is not None:
                    np.testing.assert_allclose(
                        ret1[key].detach().cpu().numpy(),
                        ret2[key].detach().cpu().numpy(),
                        rtol=1e-10,
                        atol=1e-10,
                        err_msg=f"round-trip nloc={nloc}, key={key}",
                    )

    def test_no_pbc(self) -> None:
        """Test evaluation without periodic boundary conditions."""
        rng = np.random.default_rng(GLOBAL_SEED + 3)
        natoms = 3
        coords = rng.random((1, natoms, 3)) * 5.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        e, f, v = self.dp.eval(coords, None, atom_types)

        coord_t = torch.tensor(
            coords, dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype_t = torch.tensor(
            atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
        )
        ref = self.model.forward(coord_t, atype_t, box=None)

        np.testing.assert_allclose(
            e, ref["energy"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            f, ref["force"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            v, ref["virial"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None, "ase not installed"
    )
    def test_ase_neighbor_list_consistency(self) -> None:
        """Test that ASE neighbor list gives same results as native nlist."""
        import ase.neighborlist

        rng = np.random.default_rng(GLOBAL_SEED + 11)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        # Eval without ASE neighbor list (native)
        e1, f1, v1, ae1, av1 = self.dp.eval(
            coords,
            cells,
            atom_types,
            atomic=True,
        )

        # Eval with ASE neighbor list
        dp_ase = DeepPot(
            self.tmpfile.name,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=self.rcut,
                bothways=True,
            ),
        )
        e2, f2, v2, ae2, av2 = dp_ase.eval(
            coords,
            cells,
            atom_types,
            atomic=True,
        )

        np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10, err_msg="energy")
        np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10, err_msg="force")
        np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10, err_msg="virial")
        np.testing.assert_allclose(
            ae1,
            ae2,
            rtol=1e-10,
            atol=1e-10,
            err_msg="atom_energy",
        )
        np.testing.assert_allclose(
            av1,
            av2,
            rtol=1e-10,
            atol=1e-10,
            err_msg="atom_virial",
        )

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None, "ase not installed"
    )
    def test_build_nlist_ase(self) -> None:
        """Test _build_nlist_ase produces the same neighbor sets as native."""
        import ase.neighborlist

        from deepmd.dpmodel.utils.nlist import (
            build_neighbor_list,
            extend_coord_with_ghosts,
        )
        from deepmd.dpmodel.utils.region import (
            normalize_coord,
        )

        rng = np.random.default_rng(GLOBAL_SEED + 13)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)
        atom_types_2d = atom_types.reshape(1, -1)

        dp_ase = DeepPot(
            self.tmpfile.name,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=self.rcut,
                bothways=True,
            ),
        )
        deep_eval = dp_ase.deep_eval

        # ASE path
        ext_coord_ase, _ext_atype_ase, nlist_ase, _mapping_ase = (
            deep_eval._build_nlist_ase(coords, cells, atom_types_2d)
        )

        # Native path
        box_input = cells.reshape(1, 3, 3)
        coord_normalized = normalize_coord(coords, box_input)
        ext_coord_nat, ext_atype_nat, _mapping_nat = extend_coord_with_ghosts(
            coord_normalized,
            atom_types_2d,
            cells,
            self.rcut,
        )
        sel = self.sel
        nlist_nat = build_neighbor_list(
            ext_coord_nat,
            ext_atype_nat,
            natoms,
            self.rcut,
            sel,
            distinguish_types=not self.model.mixed_types(),
        )
        ext_coord_nat = ext_coord_nat.reshape(1, -1, 3)

        # Compare: for each local atom, the set of neighbor relative
        # coordinates should match (ghost ordering may differ).
        for ii in range(natoms):
            # ASE neighbors
            nn_ase = nlist_ase[0, ii]
            mask_ase = nn_ase >= 0
            rel_ase = ext_coord_ase[0, nn_ase[mask_ase]] - coords[0, ii]

            # Native neighbors
            nn_nat = nlist_nat[0, ii]
            mask_nat = nn_nat >= 0
            rel_nat = ext_coord_nat[0, nn_nat[mask_nat]] - coords[0, ii]

            # Sort by distance then by coordinates for deterministic order
            def _sort_key(rel: np.ndarray) -> np.ndarray:
                dist = np.linalg.norm(rel, axis=-1, keepdims=True)
                return np.concatenate([dist, rel], axis=-1)

            order_ase = np.lexsort(_sort_key(rel_ase).T)
            order_nat = np.lexsort(_sort_key(rel_nat).T)

            np.testing.assert_allclose(
                rel_ase[order_ase],
                rel_nat[order_nat],
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"atom {ii}: neighbor relative coords differ",
            )

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None, "ase not installed"
    )
    def test_ase_nlist_multiple_frames(self) -> None:
        """Test ASE neighbor list with multiple frames and auto_batch_size=False."""
        import ase.neighborlist

        rng = np.random.default_rng(GLOBAL_SEED + 17)
        natoms = 4
        nframes = 3
        coords = rng.random((nframes, natoms, 3)) * 8.0
        cells = np.tile(np.eye(3).reshape(1, 9) * 10.0, (nframes, 1))
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        # Native eval (no ASE nlist)
        e1, f1, v1 = self.dp.eval(coords, cells, atom_types)

        # ASE nlist with auto_batch_size=False to exercise multi-frame path
        dp_ase = DeepPot(
            self.tmpfile.name,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=self.rcut,
                bothways=True,
            ),
            auto_batch_size=False,
        )
        e2, f2, v2 = dp_ase.eval(coords, cells, atom_types)

        np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10, err_msg="energy")
        np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10, err_msg="force")
        np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10, err_msg="virial")


if __name__ == "__main__":
    unittest.main()
