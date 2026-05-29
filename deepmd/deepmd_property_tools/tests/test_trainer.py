# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)
from unittest import (
    mock,
)

from deepmd_property_tools.tasks.trainer import (
    Trainer,
)


def test_latest_checkpoint_prefers_newest_numbered_checkpoint(tmp_path: Path) -> None:
    fallback = tmp_path / "model.ckpt.pt"
    fallback.write_text("fallback", encoding="utf-8")
    checkpoint = tmp_path / "model.ckpt-10.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")

    trainer = Trainer(save_path=tmp_path)

    assert trainer.latest_checkpoint() == checkpoint


def test_torchrun_command_includes_options() -> None:
    trainer = Trainer(
        save_path="exp",
        finetune="pretrained.pt",
        nproc_per_node=2,
        use_pretrain_script=True,
        force_load=True,
        skip_neighbor_stat=True,
        model_branch="Default",
    )

    with mock.patch("subprocess.run") as run_mock:
        trainer._run_torchrun(Path("input.json"))

    cmd = run_mock.call_args.args[0]
    assert "--nproc_per_node=2" in cmd
    assert "--finetune" in cmd
    assert "--use-pretrain-script" in cmd
    assert "--force-load" in cmd
    assert "--skip-neighbor-stat" in cmd
    assert "--model-branch" in cmd
