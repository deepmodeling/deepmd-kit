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

            # 3. Load compressed .pte and eval
            compressed_data = serialize_from_file(compressed_path)
            md_cmp = EnergyModel.deserialize(compressed_data["model"])
            md_cmp = md_cmp.to(self.device)
            ret_compressed = self._eval_model(md_cmp)

            # 4. Compare frozen vs compressed
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
