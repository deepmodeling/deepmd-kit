# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the JAX trainer writes its checkpoint pointer beside save_ckpt.

JAX training writes checkpoint directories and the stable ``.jax`` link relative
to ``save_ckpt`` (which may include a directory), but used to always write the
``checkpoint`` pointer file to the current working directory with a value that
still carried the directory prefix. The freeze entrypoint expects the pointer
inside the folder it is given and resolves its value relative to that folder, so
a directory-valued ``save_ckpt`` broke freeze/restart tooling.
"""

import os
import tempfile
import unittest
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

from deepmd.jax.train.trainer import (
    DPTrainer,
)


class TestCheckpointPointer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir.name)

    def tearDown(self) -> None:
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def _save(self, save_ckpt: str) -> None:
        trainer = DPTrainer.__new__(DPTrainer)
        trainer.save_ckpt = save_ckpt
        with (
            patch.object(DPTrainer, "_write_checkpoint"),
            patch.object(DPTrainer, "_cleanup_old_checkpoints"),
            patch("deepmd.jax.train.trainer._link_checkpoint"),
        ):
            trainer._save_checkpoint(1)

    def test_pointer_beside_ckpt_for_subdir(self) -> None:
        # save_ckpt with a directory: pointer must land in that directory with a
        # value relative to it (basename only), so freeze(subdir) resolves it.
        self._save("subdir/model.ckpt")
        pointer = Path("subdir") / "checkpoint"
        self.assertTrue(pointer.is_file())
        self.assertEqual(pointer.read_text(), "model.ckpt.jax")
        self.assertFalse(Path("checkpoint").exists())

    def test_pointer_in_cwd_for_bare_name(self) -> None:
        # default/bare save_ckpt: pointer stays in the CWD (parent is ".").
        self._save("model.ckpt")
        pointer = Path("checkpoint")
        self.assertTrue(pointer.is_file())
        self.assertEqual(pointer.read_text(), "model.ckpt.jax")


if __name__ == "__main__":
    unittest.main()
