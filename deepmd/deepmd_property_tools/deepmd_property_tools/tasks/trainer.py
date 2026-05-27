# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training task wrapper."""

from __future__ import (
    annotations,
)

import os
import subprocess
from pathlib import (
    Path,
)


class Trainer:
    def __init__(
        self,
        *,
        save_path: str | Path,
        finetune: str | None = None,
        nproc_per_node: int = 1,
        freeze: bool = False,
        use_pretrain_script: bool = False,
        skip_neighbor_stat: bool = False,
        force_load: bool = False,
        model_branch: str = "",
    ) -> None:
        self.save_path = Path(save_path)
        self.finetune = finetune
        self.nproc_per_node = nproc_per_node
        self.freeze_model = freeze
        self.use_pretrain_script = use_pretrain_script
        self.skip_neighbor_stat = skip_neighbor_stat
        self.force_load = force_load
        self.model_branch = model_branch

    def run(self, input_path: str | Path) -> None:
        input_path = Path(input_path)
        if self.nproc_per_node == 1:
            from deepmd.pt.entrypoints.main import (
                train,
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(self.save_path)
                train(
                    input_file=str(input_path),
                    init_model=None,
                    restart=None,
                    finetune=self.finetune,
                    init_frz_model=None,
                    model_branch=self.model_branch,
                    skip_neighbor_stat=self.skip_neighbor_stat,
                    use_pretrain_script=self.use_pretrain_script,
                    force_load=self.force_load,
                    output=str(self.save_path / "out.json"),
                )
            finally:
                os.chdir(old_cwd)
        else:
            self._run_torchrun(input_path)
        if self.freeze_model:
            self.freeze()

    def _run_torchrun(self, input_path: Path) -> None:
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.nproc_per_node}",
            "--no-python",
            "dp",
            "--pt",
            "train",
            str(input_path),
            "--output",
            str(self.save_path / "out.json"),
        ]
        if self.finetune is not None:
            cmd.extend(["--finetune", self.finetune])
        if self.model_branch:
            cmd.extend(["--model-branch", self.model_branch])
        if self.skip_neighbor_stat:
            cmd.append("--skip-neighbor-stat")
        if self.use_pretrain_script:
            cmd.append("--use-pretrain-script")
        if self.force_load:
            cmd.append("--force-load")
        subprocess.run(cmd, check=True, cwd=self.save_path)

    def freeze(self) -> None:
        from deepmd.pt.entrypoints.main import (
            freeze,
        )

        checkpoint = self.latest_checkpoint()
        try:
            freeze(
                model=str(checkpoint),
                output=str(self.save_path / "frozen_model.pth"),
                head=None,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Training finished, but DeePMD failed to freeze the checkpoint with TorchScript. "
                f"Use the checkpoint directly instead: {checkpoint}"
            ) from exc

    def latest_checkpoint(self) -> Path:
        candidates = sorted(
            self.save_path.glob("model.ckpt-*.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.append(self.save_path / "model.ckpt.pt")
        for checkpoint in candidates:
            if checkpoint.exists():
                return checkpoint
        raise FileNotFoundError(f"No model checkpoint found in {self.save_path}")
