# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test for the TF trainer init-mode dispatch.

``RunOptions`` records ``--init-model`` as ``init_mode == "init_from_model"``,
but ``DPTrainer`` used to dispatch on the literal ``"init_model"``.  The
mismatch silently skipped ``_init_from_ckpt`` for ``--init-model``, so the
compressed-checkpoint pre-inspection that sets ``ckpt_meta`` before graph
construction never ran.  This checks that each init mode routes to the right
initializer, using the mode strings ``RunOptions`` actually produces.
"""

import types
import unittest
from unittest import (
    mock,
)

from deepmd.tf.train.trainer import (
    DPTrainer,
)


def _dispatch(init_mode: str) -> str | None:
    """Run the trainer's init-mode dispatch for ``init_mode`` and report the route.

    A bare ``DPTrainer`` instance is used (constructor bypassed) with the three
    concrete initializers patched to record which one fires; the return value is
    the name of the initializer that was called (or ``None`` for scratch).
    """
    trainer = DPTrainer.__new__(DPTrainer)
    trainer.run_opt = types.SimpleNamespace(
        init_mode=init_mode,
        init_model="some/init.ckpt",
        restart="some/restart.ckpt",
        init_frz_model="some/frozen.pb",
        finetune="some/pretrained.pb",
    )
    with (
        mock.patch.object(DPTrainer, "_init_from_frz_model") as frz,
        mock.patch.object(DPTrainer, "_init_from_ckpt") as ckpt,
        mock.patch.object(DPTrainer, "_init_from_pretrained_model") as pre,
    ):
        trainer._init_from_run_opt(data=None, origin_type_map=None)
        if frz.called:
            return "frz"
        if ckpt.called:
            return f"ckpt:{ckpt.call_args.args[0]}"
        if pre.called:
            return "pretrained"
    return None


class TestTrainerInitMode(unittest.TestCase):
    def test_init_from_model_uses_ckpt(self) -> None:
        # RunOptions sets this string for `dp train --init-model`; it must reach
        # _init_from_ckpt so compressed-checkpoint metadata is pre-inspected.
        self.assertEqual(_dispatch("init_from_model"), "ckpt:some/init.ckpt")

    def test_restart_uses_ckpt(self) -> None:
        self.assertEqual(_dispatch("restart"), "ckpt:some/restart.ckpt")

    def test_init_from_frz_model_uses_frz(self) -> None:
        self.assertEqual(_dispatch("init_from_frz_model"), "frz")

    def test_finetune_uses_pretrained(self) -> None:
        self.assertEqual(_dispatch("finetune"), "pretrained")

    def test_scratch_is_noop(self) -> None:
        self.assertIsNone(_dispatch("init_from_scratch"))


if __name__ == "__main__":
    unittest.main()
