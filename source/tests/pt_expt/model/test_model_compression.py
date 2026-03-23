# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end model compression test for the pt_expt backend.

Tests the full pipeline: build → freeze (.pte) → compress → eval,
verifying consistency between frozen and compressed models.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
    serialize_from_file,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestModelCompression(unittest.TestCase):
    def setUp(self) -> None:
        from deepmd.pt.cxx_op import (
            ENABLE_CUSTOMIZED_OP,
        )

        if not ENABLE_CUSTOMIZED_OP:
            self.skipTest("Custom OP library not built")

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

    def _make_model(self) -> EnergyModel:
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            seed=GLOBAL_SEED,
        ).to(self.device)
        ft = EnergyFittingNet(
            self.nt,
            ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(self.device)
        return EnergyModel(ds, ft, type_map=self.type_map).to(self.device)

    def _eval_model(self, model):
        model.eval()
        coord = self.coord.clone().requires_grad_(True)
        ret = model(coord, self.atype, self.cell.reshape(1, 9))
        return {k: v.detach().cpu().numpy() for k, v in ret.items()}

    def test_model_enable_compression(self) -> None:
        """Test that model.enable_compression preserves energy/force/virial."""
        md = self._make_model()
        md.min_nbor_dist = 0.5
        ret_ref = self._eval_model(md)

        md.enable_compression(5, 0.01, 0.1, -1)
        ret_cmp = self._eval_model(md)

        for key in ("energy", "force", "virial"):
            np.testing.assert_allclose(
                ret_ref[key], ret_cmp[key], atol=1e-7, err_msg=key
            )

    def test_freeze_compress_eval(self) -> None:
        """Test full pipeline: build → freeze → compress → eval.

        The frozen (uncompressed) and compressed models must produce
        consistent energy, force, and virial.
        """
        from deepmd.pt_expt.entrypoints.compress import (
            enable_compression as compress_entry,
        )

        # 1. Build and freeze to .pte
        md = self._make_model()
        md.min_nbor_dist = 0.5
        md.eval()
        ret_frozen = self._eval_model(md)

        model_data = {"model": md.serialize(), "min_nbor_dist": 0.5}
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            frozen_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            compressed_path = f.name
        try:
            deserialize_to_file(frozen_path, model_data)

            # 2. Compress via entry point (same as `dp --pt_expt compress`)
            compress_entry(
                input_file=frozen_path,
                output=compressed_path,
                stride=0.01,
                extrapolate=5,
                check_frequency=-1,
            )

            # 3. Verify compressed .pte has compression state in model.json
            compressed_data = serialize_from_file(compressed_path)
            descrpt_data = compressed_data["model"]["descriptor"]
            self.assertIn("compress", descrpt_data)
            self.assertGreaterEqual(descrpt_data["@version"], 3)
            self.assertIn("min_nbor_dist", compressed_data)

            # 4. Deserialize and verify the model is actually compressed
            md_cmp = EnergyModel.deserialize(compressed_data["model"])
            md_cmp = md_cmp.to(self.device)
            desc = md_cmp.atomic_model.descriptor
            self.assertTrue(desc.compress)
            self.assertTrue(hasattr(desc, "compress_data"))

            # 5. Inference with compressed model matches frozen
            ret_compressed = self._eval_model(md_cmp)
            for key in ("energy", "force", "virial"):
                np.testing.assert_allclose(
                    ret_frozen[key], ret_compressed[key], atol=1e-7, err_msg=key
                )
        finally:
            os.unlink(frozen_path)
            os.unlink(compressed_path)

    def test_descriptor_preserved_in_model(self) -> None:
        """Test that pt_expt descriptor type is preserved inside the model."""
        md = self._make_model()
        desc = md.atomic_model.descriptor
        self.assertTrue(hasattr(desc, "enable_compression"))
        self.assertIsInstance(desc, torch.nn.Module)

    def test_min_nbor_dist_roundtrip(self) -> None:
        """Test that min_nbor_dist survives freeze → load round-trip."""
        md = self._make_model()
        self.assertIsNone(md.get_min_nbor_dist())

        md.min_nbor_dist = 0.5
        self.assertAlmostEqual(md.get_min_nbor_dist(), 0.5)

        # Freeze to .pte
        model_data = {"model": md.serialize(), "min_nbor_dist": 0.5}
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            frozen_path = f.name
        try:
            deserialize_to_file(frozen_path, model_data)
            loaded_data = serialize_from_file(frozen_path)
            self.assertAlmostEqual(loaded_data["min_nbor_dist"], 0.5)
        finally:
            os.unlink(frozen_path)

    def test_compress_state_serialized(self) -> None:
        """Test that compression state persists through serialize/deserialize.

        After enable_compression, the descriptor's compress flag and tabulated
        data must survive serialize → deserialize round-trip.
        """
        md = self._make_model()
        md.min_nbor_dist = 0.5
        desc = md.atomic_model.descriptor
        desc.enable_compression(0.5)
        self.assertTrue(desc.compress)

        # Serialize and deserialize
        data = desc.serialize()
        from deepmd.pt_expt.descriptor.se_e2_a import (
            DescrptSeA,
        )

        desc2 = DescrptSeA.deserialize(data)
        self.assertTrue(desc2.compress)
        self.assertTrue(hasattr(desc2, "compress_data"))

        # Compressed forward should match
        dtype = torch.float64
        # Build proper inputs for descriptor-level test
        from deepmd.dpmodel.utils.nlist import (
            build_neighbor_list,
            extend_coord_with_ghosts,
        )
        from deepmd.dpmodel.utils.region import (
            normalize_coord,
        )

        coord_np = self.coord.detach().cpu().numpy().reshape(1, self.natoms, 3)
        atype_np = self.atype.detach().cpu().numpy()
        cell_np = self.cell.reshape(1, 9).detach().cpu().numpy()
        coord_normalized = normalize_coord(coord_np, cell_np.reshape(1, 3, 3))
        ext_coord, ext_atype, _ = extend_coord_with_ghosts(
            coord_normalized, atype_np, cell_np, self.rcut
        )
        nlist_np = build_neighbor_list(
            ext_coord,
            ext_atype,
            self.natoms,
            self.rcut,
            self.sel,
            distinguish_types=True,
        )
        ext_coord_t = torch.tensor(
            ext_coord.reshape(1, -1, 3), dtype=dtype, device=self.device
        )
        ext_atype_t = torch.tensor(ext_atype, dtype=torch.int64, device=self.device)
        nlist_t = torch.tensor(nlist_np, dtype=torch.int64, device=self.device)

        rd1, _, _, _, _ = desc(ext_coord_t, ext_atype_t, nlist_t)
        rd2, _, _, _, _ = desc2(ext_coord_t, ext_atype_t, nlist_t)
        np.testing.assert_allclose(
            rd1.detach().cpu().numpy(),
            rd2.detach().cpu().numpy(),
            atol=1e-10,
        )

    def test_compress_cli_entry_point(self) -> None:
        """Test that the CLI entry point (main) dispatches compress correctly.

        This exercises the FLAGS.input argument parsing path, which previously
        had a bug (FLAGS.INPUT instead of FLAGS.input).
        """
        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        md = self._make_model()
        md.min_nbor_dist = 0.5
        md.eval()

        model_data = {"model": md.serialize(), "min_nbor_dist": 0.5}
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            frozen_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as f:
            compressed_path = f.name
        try:
            deserialize_to_file(frozen_path, model_data)

            # Call via CLI entry point
            main(
                [
                    "compress",
                    "-i",
                    frozen_path,
                    "-o",
                    compressed_path,
                ]
            )

            # Verify the compressed file was created and is loadable
            compressed_data = serialize_from_file(compressed_path)
            self.assertIn("model", compressed_data)
        finally:
            os.unlink(frozen_path)
            if os.path.exists(compressed_path):
                os.unlink(compressed_path)


if __name__ == "__main__":
    unittest.main()
