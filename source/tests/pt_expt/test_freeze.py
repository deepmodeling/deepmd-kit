# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pt_expt freeze (checkpoint → .pte / .pt2 export).

Covers both the ``freeze()`` function directly and the ``main()``
CLI dispatcher (checkpoint-dir resolution, suffix defaulting, etc.).
"""

import argparse
import os
import shutil
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
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)

from ..seed import (
    GLOBAL_SEED,
)


def _build_model_and_checkpoint(tmpdir: str) -> tuple:
    """Build a pt_expt model, wrap it, and save a fake training checkpoint.

    Returns (model, checkpoint_path, model_params).
    """
    rcut = 4.0
    rcut_smth = 0.5
    sel = [8, 6]
    nt = 2
    type_map = ["foo", "bar"]

    model_params = {
        "type_map": type_map,
        "descriptor": {
            "type": "se_e2_a",
            "rcut": rcut,
            "rcut_smth": rcut_smth,
            "sel": sel,
            "seed": GLOBAL_SEED,
        },
        "fitting_net": {
            "type": "ener",
            "seed": GLOBAL_SEED,
        },
    }

    ds = DescrptSeA(rcut, rcut_smth, sel, seed=GLOBAL_SEED)
    ft = EnergyFittingNet(
        nt,
        ds.get_dim_out(),
        mixed_types=ds.mixed_types(),
        seed=GLOBAL_SEED,
    )
    model = EnergyModel(ds, ft, type_map=type_map)
    model = model.to(torch.float64).to(DEVICE)
    model.eval()

    # Save a fake training checkpoint (same format as Trainer.save_model)
    wrapper = ModelWrapper(model, model_params=model_params)
    ckpt_path = os.path.join(tmpdir, "model.ckpt-100.pt")
    state = {
        "model": wrapper.state_dict(),
        "optimizer": {},
    }
    torch.save(state, ckpt_path)
    return model, ckpt_path, model_params


def _assert_eval_consistency(
    dp: DeepPot,
    model: torch.nn.Module,
    seed: int = GLOBAL_SEED,
) -> None:
    """Assert that DeepPot inference matches direct model forward."""
    rng = np.random.default_rng(seed)
    natoms = 5
    nt = 2
    coords = rng.random((1, natoms, 3)) * 8.0
    cells = np.eye(3).reshape(1, 9) * 10.0
    atom_types = np.array([i % nt for i in range(natoms)], dtype=np.int32)

    e, f, v, ae, av = dp.eval(coords, cells, atom_types, atomic=True)

    coord_t = torch.tensor(coords, dtype=torch.float64, device=DEVICE).requires_grad_(
        True
    )
    atype_t = torch.tensor(atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE)
    cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
    ref = model.forward(coord_t, atype_t, cell_t, do_atomic_virial=True)

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


# ---------------------------------------------------------------------------
# Tests for freeze() function directly
# ---------------------------------------------------------------------------


class TestFreezePte(unittest.TestCase):
    """Test freeze() to .pte format."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.ckpt_path, cls.model_params = _build_model_and_checkpoint(
            cls.tmpdir
        )
        cls.output_path = os.path.join(cls.tmpdir, "frozen_model.pte")

        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )

        freeze(model=cls.ckpt_path, output=cls.output_path)
        cls.dp = DeepPot(cls.output_path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_output_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.output_path))

    def test_eval_consistency(self) -> None:
        """Frozen .pte inference matches direct model forward."""
        _assert_eval_consistency(self.dp, self.model)


class TestFreezePt2(unittest.TestCase):
    """Test freeze() to .pt2 format."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.ckpt_path, cls.model_params = _build_model_and_checkpoint(
            cls.tmpdir
        )
        cls.output_path = os.path.join(cls.tmpdir, "frozen_model.pt2")

        from deepmd.pt_expt.entrypoints.main import (
            freeze,
        )

        freeze(model=cls.ckpt_path, output=cls.output_path)
        cls.dp = DeepPot(cls.output_path)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_output_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.output_path))

    def test_eval_consistency(self) -> None:
        """Frozen .pt2 inference matches direct model forward."""
        _assert_eval_consistency(self.dp, self.model)


# ---------------------------------------------------------------------------
# Tests for main() CLI dispatcher — freeze command
# ---------------------------------------------------------------------------


def _make_flags(**kwargs) -> argparse.Namespace:
    """Build a minimal argparse.Namespace resembling ``parse_args()`` output."""
    defaults = {
        "command": "freeze",
        "log_level": 3,  # WARNING — suppress info messages
        "log_path": None,
        "checkpoint_folder": ".",
        "output": "frozen_model",
        "head": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestFreezeMainFilePte(unittest.TestCase):
    """Test main() freeze path with a direct checkpoint file → .pte output."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.ckpt_path, _ = _build_model_and_checkpoint(cls.tmpdir)
        cls.output_path = os.path.join(cls.tmpdir, "frozen_via_main.pte")

        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        flags = _make_flags(
            checkpoint_folder=cls.ckpt_path,
            output=cls.output_path,
        )
        main(flags)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_output_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.output_path))

    def test_eval_consistency(self) -> None:
        dp = DeepPot(self.output_path)
        _assert_eval_consistency(dp, self.model)


class TestFreezeMainFilePt2(unittest.TestCase):
    """Test main() freeze path with a direct checkpoint file → .pt2 output."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.ckpt_path, _ = _build_model_and_checkpoint(cls.tmpdir)
        cls.output_path = os.path.join(cls.tmpdir, "frozen_via_main.pt2")

        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        flags = _make_flags(
            checkpoint_folder=cls.ckpt_path,
            output=cls.output_path,
        )
        main(flags)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_output_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.output_path))

    def test_eval_consistency(self) -> None:
        dp = DeepPot(self.output_path)
        _assert_eval_consistency(dp, self.model)


class TestFreezeMainCheckpointDir(unittest.TestCase):
    """Test main() freeze with a checkpoint directory containing a ``checkpoint`` file."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.ckpt_path, _ = _build_model_and_checkpoint(cls.tmpdir)

        # Create a checkpoint directory with a ``checkpoint`` pointer file,
        # mimicking the layout produced by Trainer.save_model.
        cls.ckpt_dir = os.path.join(cls.tmpdir, "ckpt_dir")
        os.makedirs(cls.ckpt_dir)
        ckpt_basename = os.path.basename(cls.ckpt_path)
        # Copy the checkpoint into the directory
        shutil.copy2(cls.ckpt_path, os.path.join(cls.ckpt_dir, ckpt_basename))
        # Write the pointer file
        with open(os.path.join(cls.ckpt_dir, "checkpoint"), "w") as f:
            f.write(ckpt_basename)

        cls.output_path = os.path.join(cls.tmpdir, "frozen_dir.pte")

        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        flags = _make_flags(
            checkpoint_folder=cls.ckpt_dir,
            output=cls.output_path,
        )
        main(flags)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_output_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.output_path))

    def test_eval_consistency(self) -> None:
        dp = DeepPot(self.output_path)
        _assert_eval_consistency(dp, self.model)


class TestFreezeMainDefaultSuffix(unittest.TestCase):
    """Test that main() defaults to .pt2 when output has no recognized suffix."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp()
        cls.model, cls.ckpt_path, _ = _build_model_and_checkpoint(cls.tmpdir)
        # Use a bare name without .pte/.pt2 suffix — main() should append .pt2
        cls.output_bare = os.path.join(cls.tmpdir, "frozen_model")

        from deepmd.pt_expt.entrypoints.main import (
            main,
        )

        flags = _make_flags(
            checkpoint_folder=cls.ckpt_path,
            output=cls.output_bare,
        )
        main(flags)
        cls.expected_output = cls.output_bare + ".pt2"

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir)

    def test_output_file_has_pt2_suffix(self) -> None:
        self.assertTrue(os.path.exists(self.expected_output))

    def test_eval_consistency(self) -> None:
        dp = DeepPot(self.expected_output)
        _assert_eval_consistency(dp, self.model)


if __name__ == "__main__":
    unittest.main()
