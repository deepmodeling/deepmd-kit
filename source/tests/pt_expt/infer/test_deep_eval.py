# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pt_expt inference via the DeepPot / DeepEval interface.

Verifies the full pipeline:
    model.serialize() → deserialize_to_file(.pte) → DeepPot(.pte) → eval()
"""

import importlib
import os
import tempfile
import unittest
import zipfile

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

        # Serialize and save to .pte (with atomic virial for test_dynamic_shapes)
        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.tmpfile.close()
        deserialize_to_file(cls.tmpfile.name, cls.model_data, do_atomic_virial=True)

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

    def test_use_spin_non_spin_model(self) -> None:
        self.assertFalse(self.dp.has_spin)
        self.assertEqual(self.dp.use_spin, [])

    def test_model_type(self) -> None:
        self.assertIs(self.dp.deep_eval.model_type, DeepPot)

    def test_get_model(self) -> None:
        mod = self.dp.deep_eval.get_model()
        self.assertIsInstance(mod, torch.nn.Module)

    def test_get_model_def_script(self) -> None:
        """Without model_params, get_model_def_script returns {}."""
        mds = self.dp.deep_eval.get_model_def_script()
        self.assertIsInstance(mds, dict)
        self.assertEqual(mds, {})

    def test_get_model_def_script_with_params(self) -> None:
        """Export with model_params → get_model_def_script returns them."""
        training_config = {"type_map": self.type_map, "descriptor": {"type": "se_e2_a"}}
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            tmpfile2 = f.name
        try:
            data_with_config = {**self.model_data, "model_def_script": training_config}
            deserialize_to_file(tmpfile2, data_with_config)
            dp2 = DeepPot(tmpfile2)
            mds = dp2.deep_eval.get_model_def_script()
            self.assertEqual(mds, training_config)
        finally:
            import os

            os.unlink(tmpfile2)

    def test_model_api_delegation(self) -> None:
        """Verify that model API calls are delegated to the deserialized dpmodel."""
        de = self.dp.deep_eval
        self.assertIsNotNone(de._dpmodel)
        self.assertAlmostEqual(de.get_rcut(), self.rcut)
        self.assertEqual(de.get_type_map(), self.type_map)
        self.assertEqual(de.get_dim_fparam(), 0)
        self.assertEqual(de.get_dim_aparam(), 0)
        self.assertEqual(de.get_sel_type(), self.model.get_sel_type())

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
        exported = torch.export.load(self.tmpfile.name)
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

    def test_oversized_nlist(self) -> None:
        """Test that the exported model handles nlist with more neighbors than nnei.

        In LAMMPS, the neighbor list is built with rcut + skin, so atoms
        typically have more neighbors than sum(sel).  The compiled
        format_nlist must sort by distance and truncate correctly.

        The test verifies two things:

        1. **Correctness**: the exported model with an oversized, shuffled
           nlist produces the same results as the eager model (both sort by
           distance and keep the closest sum(sel) neighbors).

        2. **Naive truncation produces wrong results**: simply taking the
           first sum(sel) columns of the shuffled nlist (simulating a C++
           implementation that truncates without sorting) gives a different
           energy.  This proves the distance sort is necessary.
        """
        exported = torch.export.load(self.tmpfile.name)
        exported_mod = exported.module()

        nnei = sum(self.sel)  # model's expected neighbor count
        nloc = 5
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = _make_sample_inputs(
            self.model, nloc=nloc
        )

        # Pad nlist with -1 columns, then shuffle column order so real
        # neighbors are interspersed with absent ones beyond column sum(sel).
        n_extra = nnei  # double the nlist width
        nlist_padded = torch.cat(
            [
                nlist_t,
                -torch.ones(
                    (*nlist_t.shape[:2], n_extra),
                    dtype=nlist_t.dtype,
                    device=nlist_t.device,
                ),
            ],
            dim=-1,
        )
        # Shuffle columns: move some real neighbors past sum(sel) boundary.
        rng = np.random.default_rng(42)
        perm = rng.permutation(nlist_padded.shape[-1])
        nlist_shuffled = nlist_padded[:, :, perm]
        assert nlist_shuffled.shape[-1] > nnei

        # --- Part 1: exported model sorts correctly ---
        # Reference: eager model with shuffled oversized nlist
        ec = ext_coord.detach().requires_grad_(True)
        ref_ret = self.model.forward_common_lower(
            ec,
            ext_atype,
            nlist_shuffled,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=True,
        )

        # Exported model with same shuffled oversized nlist
        pte_ret = exported_mod(
            ext_coord, ext_atype, nlist_shuffled, mapping_t, fparam, aparam
        )

        for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_c"):
            if ref_ret[key] is not None and key in pte_ret:
                np.testing.assert_allclose(
                    ref_ret[key].detach().cpu().numpy(),
                    pte_ret[key].detach().cpu().numpy(),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"oversized nlist, key={key}",
                )

        # --- Part 2: naive truncation gives wrong results ---
        # Simulate the old C++ bug: truncate shuffled nlist to sum(sel) columns
        # without distance sorting.  Some close neighbors that were shuffled
        # beyond column sum(sel) are lost, producing wrong energy.
        nlist_truncated = nlist_shuffled[:, :, :nnei]
        ec2 = ext_coord.detach().requires_grad_(True)
        trunc_ret = self.model.forward_common_lower(
            ec2,
            ext_atype,
            nlist_truncated,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=True,
        )
        # The truncated result MUST differ from the correctly sorted result,
        # proving that naive truncation discards real neighbors.
        e_ref = ref_ret["energy_redu"].detach().cpu().numpy()
        e_trunc = trunc_ret["energy_redu"].detach().cpu().numpy()
        assert not np.allclose(e_ref, e_trunc, rtol=1e-10, atol=1e-10), (
            "Naive truncation of shuffled nlist should give different energy, "
            "but got the same result.  The test data may not have enough "
            "neighbors shuffled beyond sum(sel) to trigger the bug."
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


class TestDeepEvalEnerPt2(unittest.TestCase):
    """Test pt_expt inference for energy models via .pt2 (AOTInductor)."""

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

        # Serialize and save to .pt2
        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pt2", delete=False)
        cls.tmpfile.close()
        # Temporarily clear default device to avoid poisoning AOTInductor
        # compilation (tests/pt/__init__.py may set a fake CUDA device).
        prev = torch.get_default_device()
        torch.set_default_device(None)
        try:
            deserialize_to_file(cls.tmpfile.name, cls.model_data, do_atomic_virial=True)
        finally:
            torch.set_default_device(prev)

        # Also save to .pte for cross-format comparison
        cls.pte_tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.pte_tmpfile.close()
        deserialize_to_file(cls.pte_tmpfile.name, cls.model_data, do_atomic_virial=True)

        # Create DeepPot for .pt2
        cls.dp = DeepPot(cls.tmpfile.name)
        # Create DeepPot for .pte reference
        cls.dp_pte = DeepPot(cls.pte_tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.unlink(cls.tmpfile.name)
        os.unlink(cls.pte_tmpfile.name)

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

    def test_use_spin_non_spin_model(self) -> None:
        self.assertFalse(self.dp.has_spin)
        self.assertEqual(self.dp.use_spin, [])

    def test_model_type(self) -> None:
        self.assertIs(self.dp.deep_eval.model_type, DeepPot)

    def test_get_model_def_script(self) -> None:
        """Without model_params, get_model_def_script returns {}."""
        mds = self.dp.deep_eval.get_model_def_script()
        self.assertIsInstance(mds, dict)
        self.assertEqual(mds, {})

    def test_get_model_def_script_with_params(self) -> None:
        """Export with model_params → get_model_def_script returns them."""
        training_config = {"type_map": self.type_map, "descriptor": {"type": "se_e2_a"}}
        with tempfile.NamedTemporaryFile(suffix=".pt2", delete=False) as f:
            tmpfile2 = f.name
        try:
            prev = torch.get_default_device()
            torch.set_default_device(None)
            try:
                data_with_config = {
                    **self.model_data,
                    "model_def_script": training_config,
                }
                deserialize_to_file(tmpfile2, data_with_config)
            finally:
                torch.set_default_device(prev)
            dp2 = DeepPot(tmpfile2)
            mds = dp2.deep_eval.get_model_def_script()
            self.assertEqual(mds, training_config)
        finally:
            import os

            os.unlink(tmpfile2)

    def test_model_api_delegation(self) -> None:
        """Verify that model API calls are delegated to the deserialized dpmodel."""
        de = self.dp.deep_eval
        self.assertIsNotNone(de._dpmodel)
        self.assertAlmostEqual(de.get_rcut(), self.rcut)
        self.assertEqual(de.get_type_map(), self.type_map)
        self.assertEqual(de.get_dim_fparam(), 0)
        self.assertEqual(de.get_dim_aparam(), 0)
        self.assertEqual(de.get_sel_type(), self.model.get_sel_type())

    def test_pt2_file_is_zip(self) -> None:
        """The .pt2 file should be a valid ZIP archive."""
        self.assertTrue(zipfile.is_zipfile(self.tmpfile.name))

    def test_pt2_has_metadata(self) -> None:
        """The .pt2 ZIP should contain metadata entries."""
        with zipfile.ZipFile(self.tmpfile.name, "r") as zf:
            names = zf.namelist()
            self.assertIn("extra/metadata.json", names)
            self.assertIn("extra/model_def_script.json", names)
            self.assertIn("extra/model.json", names)
            self.assertNotIn("extra/output_keys.json", names)
            self.assertNotIn("extra/model_params.json", names)

    def test_eval_consistency(self) -> None:
        """Test that DeepPot.eval gives same results as direct model forward."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        # .pt2 inference
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

    def test_oversized_nlist(self) -> None:
        """Test that the exported model handles nlist with more neighbors than nnei.

        In LAMMPS, the neighbor list is built with rcut + skin, so atoms
        typically have more neighbors than sum(sel).  The compiled
        format_nlist must sort by distance and truncate correctly.

        The test verifies two things:

        1. **Correctness**: the exported model with an oversized, shuffled
           nlist produces the same results as the eager model (both sort by
           distance and keep the closest sum(sel) neighbors).

        2. **Naive truncation produces wrong results**: simply taking the
           first sum(sel) columns of the shuffled nlist (simulating a C++
           implementation that truncates without sorting) gives a different
           energy.  This proves the distance sort is necessary.
        """
        exported = torch.export.load(self.pte_tmpfile.name)
        exported_mod = exported.module()

        nnei = sum(self.sel)  # model's expected neighbor count
        nloc = 5
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = _make_sample_inputs(
            self.model, nloc=nloc
        )

        # Pad nlist with -1 columns, then shuffle column order so real
        # neighbors are interspersed with absent ones beyond column sum(sel).
        n_extra = nnei  # double the nlist width
        nlist_padded = torch.cat(
            [
                nlist_t,
                -torch.ones(
                    (*nlist_t.shape[:2], n_extra),
                    dtype=nlist_t.dtype,
                    device=nlist_t.device,
                ),
            ],
            dim=-1,
        )
        # Shuffle columns: move some real neighbors past sum(sel) boundary.
        rng = np.random.default_rng(42)
        perm = rng.permutation(nlist_padded.shape[-1])
        nlist_shuffled = nlist_padded[:, :, perm]
        assert nlist_shuffled.shape[-1] > nnei

        # --- Part 1: exported model sorts correctly ---
        ec = ext_coord.detach().requires_grad_(True)
        ref_ret = self.model.forward_common_lower(
            ec,
            ext_atype,
            nlist_shuffled,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=True,
        )

        pte_ret = exported_mod(
            ext_coord, ext_atype, nlist_shuffled, mapping_t, fparam, aparam
        )

        for key in ("energy", "energy_redu", "energy_derv_r", "energy_derv_c"):
            if ref_ret[key] is not None and key in pte_ret:
                np.testing.assert_allclose(
                    ref_ret[key].detach().cpu().numpy(),
                    pte_ret[key].detach().cpu().numpy(),
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"oversized nlist, key={key}",
                )

        # --- Part 2: naive truncation gives wrong results ---
        nlist_truncated = nlist_shuffled[:, :, :nnei]
        ec2 = ext_coord.detach().requires_grad_(True)
        trunc_ret = self.model.forward_common_lower(
            ec2,
            ext_atype,
            nlist_truncated,
            mapping_t,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=True,
        )
        e_ref = ref_ret["energy_redu"].detach().cpu().numpy()
        e_trunc = trunc_ret["energy_redu"].detach().cpu().numpy()
        assert not np.allclose(e_ref, e_trunc, rtol=1e-10, atol=1e-10), (
            "Naive truncation of shuffled nlist should give different energy, "
            "but got the same result.  The test data may not have enough "
            "neighbors shuffled beyond sum(sel) to trigger the bug."
        )

    def test_serialize_round_trip(self) -> None:
        """Test .pt2 → serialize_from_file → deserialize → model gives same outputs."""
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

    def test_pt2_vs_pte_consistency(self) -> None:
        """Outputs from .pt2 DeepPot.eval should match .pte DeepPot.eval."""
        rng = np.random.default_rng(GLOBAL_SEED + 19)
        natoms = 5
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        for nframes in [1, 3]:
            coords = rng.random((nframes, natoms, 3)) * 8.0
            cells = np.tile(np.eye(3).reshape(1, 9) * 10.0, (nframes, 1))

            e1, f1, v1, ae1, av1 = self.dp.eval(coords, cells, atom_types, atomic=True)
            e2, f2, v2, ae2, av2 = self.dp_pte.eval(
                coords, cells, atom_types, atomic=True
            )

            np.testing.assert_allclose(
                e1,
                e2,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, energy",
            )
            np.testing.assert_allclose(
                f1,
                f2,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, force",
            )
            np.testing.assert_allclose(
                v1,
                v2,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, virial",
            )
            np.testing.assert_allclose(
                ae1,
                ae2,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, atom_energy",
            )
            np.testing.assert_allclose(
                av1,
                av2,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"nframes={nframes}, atom_virial",
            )


class TestDeepEvalEnerDefaultFparam(unittest.TestCase):
    """Test .pte inference with default fparam (non-spin model)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]
        cls.numb_fparam = 1
        cls.default_fparam = [0.5]

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            numb_fparam=cls.numb_fparam,
            default_fparam=cls.default_fparam,
            seed=GLOBAL_SEED,
        )
        cls.model = EnergyModel(ds, ft, type_map=cls.type_map)
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()

        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.tmpfile.close()
        deserialize_to_file(cls.tmpfile.name, cls.model_data)

        cls.dp = DeepPot(cls.tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.unlink(cls.tmpfile.name)

    def test_get_dim_fparam(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_dim_fparam(), self.numb_fparam)

    def test_eval_without_fparam_matches_explicit(self) -> None:
        """Eval without fparam should use default and match explicit fparam."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        # Eval WITHOUT fparam — should use default_fparam=[0.5]
        e_no, f_no, v_no = self.dp.eval(coords, cells, atom_types)
        # Eval WITH explicit fparam=[0.5]
        e_ex, f_ex, v_ex = self.dp.eval(
            coords, cells, atom_types, fparam=self.default_fparam
        )

        np.testing.assert_allclose(e_no, e_ex, atol=1e-10)
        np.testing.assert_allclose(f_no, f_ex, atol=1e-10)
        np.testing.assert_allclose(v_no, v_ex, atol=1e-10)

    def test_fparam_takes_effect(self) -> None:
        """Different fparam values must produce different outputs."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        e0, f0, v0 = self.dp.eval(coords, cells, atom_types, fparam=[0.0])
        e1, f1, v1 = self.dp.eval(coords, cells, atom_types, fparam=[1.0])

        assert not np.allclose(e0, e1), (
            "Changing fparam did not change output — fparam may be ignored"
        )


class TestDeepEvalEnerDefaultFparamPt2(unittest.TestCase):
    """Test .pt2 inference with default fparam (non-spin model)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]
        cls.numb_fparam = 1
        cls.default_fparam = [0.5]

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            numb_fparam=cls.numb_fparam,
            default_fparam=cls.default_fparam,
            seed=GLOBAL_SEED,
        )
        cls.model = EnergyModel(ds, ft, type_map=cls.type_map)
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()

        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pt2", delete=False)
        cls.tmpfile.close()
        prev = torch.get_default_device()
        torch.set_default_device(None)
        try:
            deserialize_to_file(cls.tmpfile.name, cls.model_data)
        finally:
            torch.set_default_device(prev)

        # Also save .pte for cross-format comparison
        cls.pte_tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.pte_tmpfile.close()
        deserialize_to_file(cls.pte_tmpfile.name, cls.model_data)

        cls.dp = DeepPot(cls.tmpfile.name)
        cls.dp_pte = DeepPot(cls.pte_tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.unlink(cls.tmpfile.name)
        os.unlink(cls.pte_tmpfile.name)

    def test_get_dim_fparam(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_dim_fparam(), self.numb_fparam)

    def test_eval_without_fparam_matches_explicit(self) -> None:
        """Eval without fparam should use default and match explicit fparam."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        e_no, f_no, v_no = self.dp.eval(coords, cells, atom_types)
        e_ex, f_ex, v_ex = self.dp.eval(
            coords, cells, atom_types, fparam=self.default_fparam
        )

        np.testing.assert_allclose(e_no, e_ex, atol=1e-10)
        np.testing.assert_allclose(f_no, f_ex, atol=1e-10)
        np.testing.assert_allclose(v_no, v_ex, atol=1e-10)

    def test_fparam_takes_effect(self) -> None:
        """Different fparam values must produce different outputs."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        e0, f0, v0 = self.dp.eval(coords, cells, atom_types, fparam=[0.0])
        e1, f1, v1 = self.dp.eval(coords, cells, atom_types, fparam=[1.0])

        assert not np.allclose(e0, e1), (
            "Changing fparam did not change output — fparam may be ignored"
        )

    def test_pt2_vs_pte_consistency(self) -> None:
        """Outputs from .pt2 with default fparam should match .pte."""
        rng = np.random.default_rng(GLOBAL_SEED + 19)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        # Both use default fparam (no explicit fparam)
        e1, f1, v1 = self.dp.eval(coords, cells, atom_types)
        e2, f2, v2 = self.dp_pte.eval(coords, cells, atom_types)

        np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10, err_msg="energy")
        np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10, err_msg="force")
        np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10, err_msg="virial")


class TestDeepEvalEnerAparam(unittest.TestCase):
    """Test .pte inference with aparam (non-spin model)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]
        cls.numb_aparam = 2

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            numb_aparam=cls.numb_aparam,
            seed=GLOBAL_SEED,
        )
        cls.model = EnergyModel(ds, ft, type_map=cls.type_map)
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()

        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.tmpfile.close()
        deserialize_to_file(cls.tmpfile.name, cls.model_data, do_atomic_virial=True)

        cls.dp = DeepPot(cls.tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.unlink(cls.tmpfile.name)

    def test_get_dim_aparam(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_dim_aparam(), self.numb_aparam)

    def test_aparam_takes_effect(self) -> None:
        """Different aparam values must produce different outputs."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        aparam_zero = np.zeros(natoms * self.numb_aparam, dtype=np.float64)
        aparam_nonzero = np.full(natoms * self.numb_aparam, 0.5, dtype=np.float64)

        e0, f0, v0 = self.dp.eval(coords, cells, atom_types, aparam=aparam_zero)
        e1, f1, v1 = self.dp.eval(coords, cells, atom_types, aparam=aparam_nonzero)

        assert not np.allclose(e0, e1), (
            "Changing aparam did not change output — aparam may be ignored"
        )

    def test_eval_without_aparam_raises(self) -> None:
        """Model with dim_aparam > 0 must raise when aparam not provided."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        with self.assertRaises(ValueError):
            self.dp.eval(coords, cells, atom_types)

    def test_eval_consistency(self) -> None:
        """Test that DeepPot.eval with aparam matches direct model forward."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)
        aparam = rng.random(natoms * self.numb_aparam)

        e, f, v, ae, av = self.dp.eval(
            coords, cells, atom_types, atomic=True, aparam=aparam
        )

        coord_t = torch.tensor(
            coords, dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype_t = torch.tensor(
            atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
        )
        cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
        aparam_t = torch.tensor(
            aparam.reshape(1, natoms, self.numb_aparam),
            dtype=torch.float64,
            device=DEVICE,
        )
        ref = self.model.forward(
            coord_t, atype_t, cell_t, aparam=aparam_t, do_atomic_virial=True
        )

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


class TestDeepEvalEnerAparamPt2(unittest.TestCase):
    """Test .pt2 inference with aparam (non-spin model)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]
        cls.numb_aparam = 2

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            numb_aparam=cls.numb_aparam,
            seed=GLOBAL_SEED,
        )
        cls.model = EnergyModel(ds, ft, type_map=cls.type_map)
        cls.model = cls.model.to(torch.float64)
        cls.model.eval()

        cls.model_data = {"model": cls.model.serialize()}
        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pt2", delete=False)
        cls.tmpfile.close()
        prev = torch.get_default_device()
        torch.set_default_device(None)
        try:
            deserialize_to_file(cls.tmpfile.name, cls.model_data, do_atomic_virial=True)
        finally:
            torch.set_default_device(prev)

        # Also save .pte for cross-format comparison
        cls.pte_tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.pte_tmpfile.close()
        deserialize_to_file(cls.pte_tmpfile.name, cls.model_data, do_atomic_virial=True)

        cls.dp = DeepPot(cls.tmpfile.name)
        cls.dp_pte = DeepPot(cls.pte_tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        os.unlink(cls.tmpfile.name)
        os.unlink(cls.pte_tmpfile.name)

    def test_get_dim_aparam(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_dim_aparam(), self.numb_aparam)

    def test_aparam_takes_effect(self) -> None:
        """Different aparam values must produce different outputs."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        aparam_zero = np.zeros(natoms * self.numb_aparam, dtype=np.float64)
        aparam_nonzero = np.full(natoms * self.numb_aparam, 0.5, dtype=np.float64)

        e0, f0, v0 = self.dp.eval(coords, cells, atom_types, aparam=aparam_zero)
        e1, f1, v1 = self.dp.eval(coords, cells, atom_types, aparam=aparam_nonzero)

        assert not np.allclose(e0, e1), (
            "Changing aparam did not change output — aparam may be ignored"
        )

    def test_eval_without_aparam_raises(self) -> None:
        """Model with dim_aparam > 0 must raise when aparam not provided."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        with self.assertRaises(ValueError):
            self.dp.eval(coords, cells, atom_types)

    def test_eval_consistency(self) -> None:
        """Test that .pt2 DeepPot.eval with aparam matches direct model forward."""
        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)
        aparam = rng.random(natoms * self.numb_aparam)

        e, f, v, ae, av = self.dp.eval(
            coords, cells, atom_types, atomic=True, aparam=aparam
        )

        coord_t = torch.tensor(
            coords, dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype_t = torch.tensor(
            atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
        )
        cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
        aparam_t = torch.tensor(
            aparam.reshape(1, natoms, self.numb_aparam),
            dtype=torch.float64,
            device=DEVICE,
        )
        ref = self.model.forward(
            coord_t, atype_t, cell_t, aparam=aparam_t, do_atomic_virial=True
        )

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

    def test_pt2_vs_pte_consistency(self) -> None:
        """Outputs from .pt2 with aparam should match .pte."""
        rng = np.random.default_rng(GLOBAL_SEED + 19)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)
        aparam = rng.random(natoms * self.numb_aparam)

        e1, f1, v1 = self.dp.eval(coords, cells, atom_types, aparam=aparam)
        e2, f2, v2 = self.dp_pte.eval(coords, cells, atom_types, aparam=aparam)

        np.testing.assert_allclose(e1, e2, rtol=1e-10, atol=1e-10, err_msg="energy")
        np.testing.assert_allclose(f1, f2, rtol=1e-10, atol=1e-10, err_msg="force")
        np.testing.assert_allclose(v1, v2, rtol=1e-10, atol=1e-10, err_msg="virial")


class TestEvalTypeEbd(unittest.TestCase):
    """Test eval_typeebd for pt_expt models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]

        # se_e2_a model (no type embedding)
        ds_sea = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft_sea = EnergyFittingNet(
            cls.nt,
            ds_sea.get_dim_out(),
            mixed_types=ds_sea.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model_sea = EnergyModel(ds_sea, ft_sea, type_map=cls.type_map)
        model_sea = model_sea.to(torch.float64)
        model_sea.eval()
        cls._tmpdir = tempfile.mkdtemp()
        pte_sea = os.path.join(cls._tmpdir, "sea.pte")
        deserialize_to_file(pte_sea, {"model": model_sea.serialize()})
        cls.dp_sea = DeepPot(pte_sea)

        # DPA1 model (has type embedding)
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        ds_dpa1 = DescrptDPA1(
            cls.rcut,
            cls.rcut_smth,
            cls.sel,
            ntypes=cls.nt,
            seed=GLOBAL_SEED,
        )
        ft_dpa1 = EnergyFittingNet(
            cls.nt,
            ds_dpa1.get_dim_out(),
            mixed_types=ds_dpa1.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model_dpa1 = EnergyModel(ds_dpa1, ft_dpa1, type_map=cls.type_map)
        model_dpa1 = model_dpa1.to(torch.float64)
        model_dpa1.eval()
        pte_dpa1 = os.path.join(cls._tmpdir, "dpa1.pte")
        deserialize_to_file(pte_dpa1, {"model": model_dpa1.serialize()})
        cls.dp_dpa1 = DeepPot(pte_dpa1)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_typeebd_dpa1(self) -> None:
        """DPA1 model has type embedding, should return valid array."""
        typeebd = self.dp_dpa1.deep_eval.eval_typeebd()
        self.assertEqual(typeebd.ndim, 2)
        # DPA1 TypeEmbedNet outputs (ntypes+1) rows (padding type included)
        self.assertIn(typeebd.shape[0], (self.nt, self.nt + 1))
        self.assertGreater(typeebd.shape[1], 0)

    def test_typeebd_sea_raises(self) -> None:
        """se_e2_a model has no type embedding, should raise KeyError."""
        with self.assertRaises(KeyError):
            self.dp_sea.deep_eval.eval_typeebd()


class TestEvalDescriptor(unittest.TestCase):
    """Test eval_descriptor for pt_expt models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]

        # se_e2_a model
        ds_sea = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft_sea = EnergyFittingNet(
            cls.nt,
            ds_sea.get_dim_out(),
            mixed_types=ds_sea.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model_sea = EnergyModel(ds_sea, ft_sea, type_map=cls.type_map)
        model_sea = model_sea.to(torch.float64)
        model_sea.eval()
        cls._tmpdir = tempfile.mkdtemp()
        pte_sea = os.path.join(cls._tmpdir, "sea.pte")
        deserialize_to_file(pte_sea, {"model": model_sea.serialize()})
        cls.dp_sea = DeepPot(pte_sea)
        cls.dim_descrpt_sea = ds_sea.get_dim_out()

        # DPA1 model
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        ds_dpa1 = DescrptDPA1(
            cls.rcut,
            cls.rcut_smth,
            cls.sel,
            ntypes=cls.nt,
            seed=GLOBAL_SEED,
        )
        ft_dpa1 = EnergyFittingNet(
            cls.nt,
            ds_dpa1.get_dim_out(),
            mixed_types=ds_dpa1.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model_dpa1 = EnergyModel(ds_dpa1, ft_dpa1, type_map=cls.type_map)
        model_dpa1 = model_dpa1.to(torch.float64)
        model_dpa1.eval()
        pte_dpa1 = os.path.join(cls._tmpdir, "dpa1.pte")
        deserialize_to_file(pte_dpa1, {"model": model_dpa1.serialize()})
        cls.dp_dpa1 = DeepPot(pte_dpa1)
        cls.dim_descrpt_dpa1 = ds_dpa1.get_dim_out()

        # se_e2_a model with fparam (dim_fparam=1, no default; swap _dpmodel
        # because se_e2_a + fparam hits GuardOnDataDependentSymNode in export)
        ds_fp = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft_fp = EnergyFittingNet(
            cls.nt,
            ds_fp.get_dim_out(),
            mixed_types=ds_fp.mixed_types(),
            numb_fparam=1,
            seed=GLOBAL_SEED,
        )
        model_fp = EnergyModel(ds_fp, ft_fp, type_map=cls.type_map)
        pte_fp = os.path.join(cls._tmpdir, "fp.pte")
        deserialize_to_file(pte_fp, {"model": model_sea.serialize()})
        cls.dp_fp = DeepPot(pte_fp)
        cls.dp_fp.deep_eval._dpmodel = model_fp
        cls.dim_descrpt_fp = ds_fp.get_dim_out()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _make_inputs(self):
        nframes = 1
        natoms = 6
        coords = (
            np.random.default_rng(42).random((nframes, natoms, 3)).astype(np.float64)
        )
        cells = 5.0 * np.eye(3, dtype=np.float64).reshape(1, 3, 3).repeat(
            nframes, axis=0
        )
        atom_types = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        return coords, cells, atom_types

    def test_descriptor_shape_sea(self) -> None:
        """se_e2_a descriptor has correct shape."""
        coords, cells, atom_types = self._make_inputs()
        descpt = self.dp_sea.deep_eval.eval_descriptor(coords, cells, atom_types)
        self.assertEqual(descpt.shape, (1, 6, self.dim_descrpt_sea))

    def test_descriptor_shape_dpa1(self) -> None:
        """DPA1 descriptor has correct shape."""
        coords, cells, atom_types = self._make_inputs()
        descpt = self.dp_dpa1.deep_eval.eval_descriptor(coords, cells, atom_types)
        self.assertEqual(descpt.shape, (1, 6, self.dim_descrpt_dpa1))

    def test_descriptor_deterministic_sea(self) -> None:
        """Calling eval_descriptor twice gives same result for se_e2_a."""
        coords, cells, atom_types = self._make_inputs()
        d1 = self.dp_sea.deep_eval.eval_descriptor(coords, cells, atom_types)
        d2 = self.dp_sea.deep_eval.eval_descriptor(coords, cells, atom_types)
        np.testing.assert_array_equal(d1, d2)

    def test_descriptor_deterministic_dpa1(self) -> None:
        """Calling eval_descriptor twice gives same result for DPA1."""
        coords, cells, atom_types = self._make_inputs()
        d1 = self.dp_dpa1.deep_eval.eval_descriptor(coords, cells, atom_types)
        d2 = self.dp_dpa1.deep_eval.eval_descriptor(coords, cells, atom_types)
        np.testing.assert_array_equal(d1, d2)

    def test_descriptor_with_fparam(self) -> None:
        """eval_descriptor works with fparam."""
        coords, cells, atom_types = self._make_inputs()
        fparam = np.array([0.5], dtype=np.float64)
        descpt = self.dp_fp.deep_eval.eval_descriptor(
            coords, cells, atom_types, fparam=fparam
        )
        self.assertEqual(descpt.shape, (1, 6, self.dim_descrpt_fp))

    def test_descriptor_without_fparam_raises(self) -> None:
        """eval_descriptor raises when fparam is required but not provided."""
        coords, cells, atom_types = self._make_inputs()
        with self.assertRaises(ValueError):
            self.dp_fp.deep_eval.eval_descriptor(coords, cells, atom_types)

    def test_descriptor_accepts_list_inputs(self) -> None:
        """eval_descriptor accepts Python lists (not just np.ndarray)."""
        coords, cells, atom_types = self._make_inputs()
        descpt_arr = self.dp_sea.deep_eval.eval_descriptor(coords, cells, atom_types)
        descpt_list = self.dp_sea.deep_eval.eval_descriptor(
            coords.tolist(), cells.tolist(), atom_types.tolist()
        )
        np.testing.assert_allclose(descpt_arr, descpt_list)

    def test_fitting_last_layer_accepts_list_inputs(self) -> None:
        """eval_fitting_last_layer accepts Python lists (not just np.ndarray)."""
        coords, cells, atom_types = self._make_inputs()
        mid_arr = self.dp_sea.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        mid_list = self.dp_sea.deep_eval.eval_fitting_last_layer(
            coords.tolist(), cells.tolist(), atom_types.tolist()
        )
        np.testing.assert_allclose(mid_arr, mid_list)


class TestEvalFittingLastLayer(unittest.TestCase):
    """Test eval_fitting_last_layer for pt_expt models."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]
        cls.neuron = [120, 120, 120]  # default fitting net neurons

        # se_e2_a model (mixed_types=False)
        ds_sea = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft_sea = EnergyFittingNet(
            cls.nt,
            ds_sea.get_dim_out(),
            mixed_types=ds_sea.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model_sea = EnergyModel(ds_sea, ft_sea, type_map=cls.type_map)
        model_sea = model_sea.to(torch.float64)
        model_sea.eval()
        cls._tmpdir = tempfile.mkdtemp()
        pte_sea = os.path.join(cls._tmpdir, "sea.pte")
        deserialize_to_file(pte_sea, {"model": model_sea.serialize()})
        cls.dp_sea = DeepPot(pte_sea)

        # DPA1 model (mixed_types=True)
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        ds_dpa1 = DescrptDPA1(
            cls.rcut,
            cls.rcut_smth,
            cls.sel,
            ntypes=cls.nt,
            seed=GLOBAL_SEED,
        )
        ft_dpa1 = EnergyFittingNet(
            cls.nt,
            ds_dpa1.get_dim_out(),
            mixed_types=ds_dpa1.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model_dpa1 = EnergyModel(ds_dpa1, ft_dpa1, type_map=cls.type_map)
        model_dpa1 = model_dpa1.to(torch.float64)
        model_dpa1.eval()
        pte_dpa1 = os.path.join(cls._tmpdir, "dpa1.pte")
        deserialize_to_file(pte_dpa1, {"model": model_dpa1.serialize()})
        cls.dp_dpa1 = DeepPot(pte_dpa1)

        # se_e2_a model with fparam and aparam (swap _dpmodel because
        # se_e2_a + fparam hits GuardOnDataDependentSymNode in export)
        ds_fp = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft_fp = EnergyFittingNet(
            cls.nt,
            ds_fp.get_dim_out(),
            mixed_types=ds_fp.mixed_types(),
            numb_fparam=1,
            numb_aparam=2,
            seed=GLOBAL_SEED,
        )
        model_fp = EnergyModel(ds_fp, ft_fp, type_map=cls.type_map)
        pte_fp = os.path.join(cls._tmpdir, "fp.pte")
        deserialize_to_file(pte_fp, {"model": model_sea.serialize()})
        cls.dp_fp = DeepPot(pte_fp)
        cls.dp_fp.deep_eval._dpmodel = model_fp

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _make_inputs(self):
        nframes = 1
        natoms = 6
        coords = (
            np.random.default_rng(42).random((nframes, natoms, 3)).astype(np.float64)
        )
        cells = 5.0 * np.eye(3, dtype=np.float64).reshape(1, 3, 3).repeat(
            nframes, axis=0
        )
        atom_types = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        return coords, cells, atom_types

    def test_fitting_ll_shape_sea(self) -> None:
        """se_e2_a fitting last layer has correct shape."""
        coords, cells, atom_types = self._make_inputs()
        fit_ll = self.dp_sea.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        self.assertEqual(fit_ll.shape, (1, 6, self.neuron[-1]))

    def test_fitting_ll_shape_dpa1(self) -> None:
        """DPA1 fitting last layer has correct shape."""
        coords, cells, atom_types = self._make_inputs()
        fit_ll = self.dp_dpa1.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        self.assertEqual(fit_ll.shape, (1, 6, self.neuron[-1]))

    def test_fitting_ll_deterministic_sea(self) -> None:
        """Verify calling twice gives the same result for se_e2_a."""
        coords, cells, atom_types = self._make_inputs()
        fit_ll1 = self.dp_sea.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        fit_ll2 = self.dp_sea.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        np.testing.assert_array_equal(fit_ll1, fit_ll2)

    def test_fitting_ll_deterministic_dpa1(self) -> None:
        """Verify calling twice gives the same result for DPA1."""
        coords, cells, atom_types = self._make_inputs()
        fit_ll1 = self.dp_dpa1.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        fit_ll2 = self.dp_dpa1.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        np.testing.assert_array_equal(fit_ll1, fit_ll2)

    def test_fitting_ll_with_fparam_aparam(self) -> None:
        """eval_fitting_last_layer works with fparam and aparam."""
        coords, cells, atom_types = self._make_inputs()
        fparam = np.array([0.5], dtype=np.float64)
        aparam = np.zeros((1, 6, 2), dtype=np.float64)
        fit_ll = self.dp_fp.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types, fparam=fparam, aparam=aparam
        )
        self.assertEqual(fit_ll.shape, (1, 6, self.neuron[-1]))

    def test_fitting_ll_without_fparam_raises(self) -> None:
        """eval_fitting_last_layer raises when fparam is required but not provided."""
        coords, cells, atom_types = self._make_inputs()
        with self.assertRaises(ValueError):
            self.dp_fp.deep_eval.eval_fitting_last_layer(coords, cells, atom_types)


class TestGetDpAtomicModel(unittest.TestCase):
    """Test get_dp_atomic_model() API on various model types."""

    def test_energy_model(self) -> None:
        """Standard energy model returns a DPAtomicModel."""
        from deepmd.dpmodel.atomic_model.dp_atomic_model import (
            DPAtomicModel as DPAtomicModelDP,
        )

        ds = DescrptSeA(4.0, 0.5, [8, 6])
        ft = EnergyFittingNet(2, ds.get_dim_out(), mixed_types=False, seed=GLOBAL_SEED)
        model = EnergyModel(ds, ft, type_map=["foo", "bar"])
        dp_am = model.get_dp_atomic_model()
        self.assertIsNotNone(dp_am)
        self.assertIsInstance(dp_am, DPAtomicModelDP)
        self.assertTrue(hasattr(dp_am, "descriptor"))
        self.assertTrue(hasattr(dp_am, "fitting_net"))

    def test_zbl_model_returns_none(self) -> None:
        """DPZBLModel wraps a LinearEnergyAtomicModel, so get_dp_atomic_model returns None."""
        from deepmd.dpmodel.atomic_model.dp_atomic_model import (
            DPAtomicModel as DPAtomicModelDP,
        )
        from deepmd.dpmodel.atomic_model.linear_atomic_model import (
            DPZBLLinearEnergyAtomicModel,
        )
        from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
            PairTabAtomicModel,
        )
        from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP

        ds = DescrptDPA1DP(4.0, 0.5, [14], ntypes=2)
        ft = EnergyFittingNet(2, ds.get_dim_out(), mixed_types=True, seed=GLOBAL_SEED)
        dp_am = DPAtomicModelDP(ds, ft, type_map=["foo", "bar"])
        pair_tab = PairTabAtomicModel(
            tab_file=None, rcut=4.0, sel=14, type_map=["foo", "bar"]
        )
        zbl_am = DPZBLLinearEnergyAtomicModel(
            dp_am, pair_tab, sw_rmin=1.0, sw_rmax=2.0, type_map=["foo", "bar"]
        )
        # LinearEnergyAtomicModel is not a DPAtomicModel
        self.assertFalse(isinstance(zbl_am, DPAtomicModelDP))

    def test_spin_model_delegates(self) -> None:
        """SpinModel.get_dp_atomic_model() delegates to backbone."""
        from deepmd.dpmodel.atomic_model.dp_atomic_model import (
            DPAtomicModel as DPAtomicModelDP,
        )
        from deepmd.dpmodel.model.spin_model import (
            SpinModel,
        )
        from deepmd.utils.spin import (
            Spin,
        )

        ds = DescrptSeA(4.0, 0.5, [8, 6])
        ft = EnergyFittingNet(2, ds.get_dim_out(), mixed_types=False, seed=GLOBAL_SEED)
        model = EnergyModel(ds, ft, type_map=["foo", "bar"])
        spin = Spin(
            use_spin=[False, False],
            virtual_scale=[0.0, 0.0],
        )
        spin_model = SpinModel(backbone_model=model, spin=spin)
        dp_am = spin_model.get_dp_atomic_model()
        self.assertIsNotNone(dp_am)
        self.assertIsInstance(dp_am, DPAtomicModelDP)

    def test_deserialized_model_delegates(self) -> None:
        """Model deserialized from .pte exposes get_dp_atomic_model()."""
        from deepmd.dpmodel.atomic_model.dp_atomic_model import (
            DPAtomicModel as DPAtomicModelDP,
        )

        ds = DescrptSeA(4.0, 0.5, [8, 6])
        ft = EnergyFittingNet(2, ds.get_dim_out(), mixed_types=False, seed=GLOBAL_SEED)
        model = EnergyModel(ds, ft, type_map=["foo", "bar"])
        model = model.to(torch.float64)
        model.eval()

        tmpdir = tempfile.mkdtemp()
        try:
            pte_path = os.path.join(tmpdir, "test.pte")
            deserialize_to_file(pte_path, {"model": model.serialize()})
            dp = DeepPot(pte_path)
            dp_am = dp.deep_eval._dpmodel.get_dp_atomic_model()
            self.assertIsNotNone(dp_am)
            self.assertIsInstance(dp_am, DPAtomicModelDP)
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEvalDiagSpinModel(unittest.TestCase):
    """Test eval diagnostic methods on spin models."""

    @classmethod
    def setUpClass(cls) -> None:
        from deepmd.dpmodel.model.spin_model import (
            SpinModel,
        )
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )
        from deepmd.utils.spin import (
            Spin,
        )

        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]

        # DPA1 model with spin wrapper
        ds = DescrptDPA1(
            cls.rcut, cls.rcut_smth, cls.sel, ntypes=cls.nt, seed=GLOBAL_SEED
        )
        ft = EnergyFittingNet(
            cls.nt, ds.get_dim_out(), mixed_types=ds.mixed_types(), seed=GLOBAL_SEED
        )
        backbone = EnergyModel(ds, ft, type_map=cls.type_map)
        backbone = backbone.to(torch.float64)
        backbone.eval()

        spin = Spin(use_spin=[True, False], virtual_scale=[0.5, 0.0])
        cls.spin_model = SpinModel(backbone_model=backbone, spin=spin)

        # Export backbone as .pte, then swap _dpmodel to SpinModel
        cls._tmpdir = tempfile.mkdtemp()
        pte_path = os.path.join(cls._tmpdir, "spin.pte")
        deserialize_to_file(pte_path, {"model": backbone.serialize()})
        cls.dp = DeepPot(pte_path)
        cls.dp.deep_eval._dpmodel = cls.spin_model

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _make_inputs(self):
        nframes = 1
        natoms = 6
        coords = (
            np.random.default_rng(42).random((nframes, natoms, 3)).astype(np.float64)
        )
        cells = 5.0 * np.eye(3, dtype=np.float64).reshape(1, 3, 3).repeat(
            nframes, axis=0
        )
        atom_types = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        return coords, cells, atom_types

    def test_eval_typeebd_spin(self) -> None:
        """eval_typeebd traverses backbone_model for spin models."""
        typeebd = self.dp.deep_eval.eval_typeebd()
        self.assertEqual(typeebd.ndim, 2)
        # DPA1 TypeEmbedNet outputs ntypes or ntypes+1
        self.assertIn(typeebd.shape[0], (self.nt, self.nt + 1))
        self.assertGreater(typeebd.shape[1], 0)

    def test_eval_descriptor_spin_raises(self) -> None:
        """eval_descriptor raises NotImplementedError for spin models."""
        coords, cells, atom_types = self._make_inputs()
        with self.assertRaises(NotImplementedError):
            self.dp.deep_eval.eval_descriptor(coords, cells, atom_types)

    def test_eval_fitting_last_layer_spin_raises(self) -> None:
        """eval_fitting_last_layer raises NotImplementedError for spin models."""
        coords, cells, atom_types = self._make_inputs()
        with self.assertRaises(NotImplementedError):
            self.dp.deep_eval.eval_fitting_last_layer(coords, cells, atom_types)


class TestEvalDescriptorASE(unittest.TestCase):
    """Test eval_descriptor with ASE neighbor list."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.type_map = ["foo", "bar"]

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = EnergyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        model = EnergyModel(ds, ft, type_map=cls.type_map)
        model = model.to(torch.float64)
        model.eval()
        cls.dim_descrpt = ds.get_dim_out()

        cls._tmpdir = tempfile.mkdtemp()
        pte_path = os.path.join(cls._tmpdir, "sea.pte")
        deserialize_to_file(pte_path, {"model": model.serialize()})
        cls.dp_native = DeepPot(pte_path)

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil

        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None, "ase not installed"
    )
    def test_eval_descriptor_ase_vs_native(self) -> None:
        """eval_descriptor with ASE nlist matches native nlist."""
        import ase.neighborlist

        pte_path = os.path.join(self._tmpdir, "sea.pte")
        dp_ase = DeepPot(
            pte_path,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=self.rcut, bothways=True
            ),
        )

        rng = np.random.default_rng(GLOBAL_SEED + 99)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        d_native = self.dp_native.deep_eval.eval_descriptor(coords, cells, atom_types)
        d_ase = dp_ase.deep_eval.eval_descriptor(coords, cells, atom_types)

        self.assertEqual(d_native.shape, d_ase.shape)
        np.testing.assert_allclose(d_native, d_ase, rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(
        importlib.util.find_spec("ase") is not None, "ase not installed"
    )
    def test_eval_fitting_last_layer_ase_vs_native(self) -> None:
        """eval_fitting_last_layer with ASE nlist matches native nlist."""
        import ase.neighborlist

        pte_path = os.path.join(self._tmpdir, "sea.pte")
        dp_ase = DeepPot(
            pte_path,
            neighbor_list=ase.neighborlist.NewPrimitiveNeighborList(
                cutoffs=self.rcut, bothways=True
            ),
        )

        rng = np.random.default_rng(GLOBAL_SEED + 99)
        natoms = 5
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % self.nt for i in range(natoms)], dtype=np.int32)

        f_native = self.dp_native.deep_eval.eval_fitting_last_layer(
            coords, cells, atom_types
        )
        f_ase = dp_ase.deep_eval.eval_fitting_last_layer(coords, cells, atom_types)

        self.assertEqual(f_native.shape, f_ase.shape)
        np.testing.assert_allclose(f_native, f_ase, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
