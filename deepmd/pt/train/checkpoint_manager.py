# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkpoint management for model saving, loading, and recovery.

This module provides a clean interface for managing model checkpoints,
including saving, loading, automatic cleanup, and fine-tuning support.
"""

from __future__ import (
    annotations,
)

import logging
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.distributed as dist

from deepmd.common import (
    symlink_prefix_files,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

if TYPE_CHECKING:
    from deepmd.pt.train.config import (
        CheckpointConfig,
    )

log = logging.getLogger(__name__)


class CheckpointManager:
    """Manages model checkpoints throughout training.

    This class handles saving checkpoints, loading for resume/finetune,
    automatic cleanup of old checkpoints, and symlink management.

    Attributes
    ----------
    config : CheckpointConfig
        Configuration for checkpoint behavior.
    rank : int
        Distributed training rank.
    latest_model : Path | None
        Path to the most recent checkpoint.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        rank: int = 0,
    ) -> None:
        """Initialize checkpoint manager.

        Parameters
        ----------
        config : CheckpointConfig
            Configuration for checkpoint behavior.
        rank : int
            Distributed training rank (only rank 0 saves).
        """
        self.config = config
        self.rank = rank
        self.latest_model: Path | None = None
        self._saved_checkpoints: list[Path] = []

    def save(
        self,
        step: int,
        wrapper: torch.nn.Module,
        optimizer: torch.optim.Optimizer | Any,
        lr: float = 0.0,
    ) -> Path | None:
        """Save a checkpoint.

        Parameters
        ----------
        step : int
            Current training step.
        wrapper : torch.nn.Module
            Model wrapper (possibly wrapped in DDP).
        optimizer : torch.optim.Optimizer | Any
            Optimizer instance.
        lr : float
            Current learning rate.

        Returns
        -------
        Path | None
            Path to saved checkpoint, or None if not saved.
        """
        if self.rank != 0:
            return None

        # Get unwrapped module if using DDP
        module = wrapper
        if dist.is_available() and dist.is_initialized():
            if hasattr(wrapper, "module"):
                module = wrapper.module

        # Update training info
        if hasattr(module, "train_infos"):
            module.train_infos["lr"] = float(lr)
            module.train_infos["step"] = step

        # Prepare checkpoint path
        save_path = Path(self.config.save_ckpt + f"-{step + 1}.pt")

        # Prepare optimizer state
        optim_state = deepcopy(optimizer.state_dict())
        if "param_groups" in optim_state:
            for group in optim_state["param_groups"]:
                if "lr" in group:
                    group["lr"] = float(group["lr"])

        # Save checkpoint
        checkpoint = {
            "model": module.state_dict(),
            "optimizer": optim_state,
            "step": step,
            "lr": float(lr),
        }

        torch.save(checkpoint, save_path)
        self.latest_model = save_path
        self._saved_checkpoints.append(save_path)

        # Update symlinks
        symlink_prefix_files(save_path.stem, self.config.save_ckpt)

        # Write checkpoint file
        with open("checkpoint", "w") as f:
            f.write(str(save_path))

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        log.info(f"Saved checkpoint to {save_path}")
        return save_path

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints keeping only max_ckpt_keep most recent."""
        if len(self._saved_checkpoints) <= self.config.max_ckpt_keep:
            return

        # Sort by modification time
        checkpoint_files = [
            f
            for f in Path(".").glob(f"{self.config.save_ckpt}*.pt")
            if not f.is_symlink() and f.name.startswith(self.config.save_ckpt)
        ]
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime)

        # Remove oldest
        while len(checkpoint_files) > self.config.max_ckpt_keep:
            old_file = checkpoint_files.pop(0)
            try:
                old_file.unlink()
                log.debug(f"Removed old checkpoint: {old_file}")
            except OSError as e:
                log.warning(f"Failed to remove old checkpoint {old_file}: {e}")

    def load(
        self,
        checkpoint_path: str | Path,
        wrapper: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | Any = None,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to checkpoint file.
        wrapper : torch.nn.Module | None
            Model wrapper to load state into.
        optimizer : torch.optim.Optimizer | Any | None
            Optimizer to load state into.
        strict : bool
            Whether to strictly enforce state dict matching.

        Returns
        -------
        dict[str, Any]
            Loaded checkpoint dictionary.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        log.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        # Load model state
        if wrapper is not None:
            module = wrapper
            if dist.is_available() and dist.is_initialized():
                if hasattr(wrapper, "module"):
                    module = wrapper.module

            state_dict = checkpoint.get("model", checkpoint)
            if "model" in checkpoint:
                state_dict = checkpoint["model"]

            module.load_state_dict(state_dict, strict=strict)
            log.info("Model state loaded successfully")

        # Load optimizer state
        if optimizer is not None and "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                log.info("Optimizer state loaded successfully")
            except Exception as e:
                log.warning(f"Failed to load optimizer state: {e}")

        return checkpoint

    def load_for_finetune(
        self,
        checkpoint_path: str | Path,
        wrapper: torch.nn.Module,
        force_load: bool = False,
    ) -> dict[str, Any]:
        """Load checkpoint for fine-tuning with optional key mapping.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to pretrained checkpoint.
        wrapper : torch.nn.Module
            Model wrapper to load into.
        force_load : bool
            If True, initialize missing keys from current model.

        Returns
        -------
        dict[str, Any]
            Loaded checkpoint info.
        """
        checkpoint_path = Path(checkpoint_path)
        log.info(f"Loading pretrained model from {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        module = wrapper
        if dist.is_available() and dist.is_initialized():
            if hasattr(wrapper, "module"):
                module = wrapper.module

        state_dict = checkpoint.get("model", checkpoint)
        if "model" in checkpoint:
            state_dict = checkpoint["model"]

        target_state_dict = module.state_dict()
        input_keys = set(state_dict.keys())
        target_keys = set(target_state_dict.keys())

        missing_keys = target_keys - input_keys
        unexpected_keys = input_keys - target_keys

        if missing_keys and force_load:
            log.warning(
                f"Force load: initializing {len(missing_keys)} missing keys from model"
            )
            for key in missing_keys:
                state_dict[key] = target_state_dict[key].clone().detach()

        # Load with strict=False to handle architecture differences
        module.load_state_dict(state_dict, strict=False)

        if missing_keys and not force_load:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        return {
            "state_dict": state_dict,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        }

    def get_start_step(self, checkpoint_path: str | Path | None) -> int:
        """Get the starting step from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path | None
            Path to checkpoint, or None.

        Returns
        -------
        int
            Step to resume from, or 0 for fresh start.
        """
        if checkpoint_path is None:
            return 0

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            step = checkpoint.get("step", 0)
            log.info(f"Resuming from step {step}")
            return step
        except Exception as e:
            log.warning(f"Failed to get step from checkpoint: {e}")
            return 0

    def get_latest_checkpoint(self) -> Path | None:
        """Get the path to the latest checkpoint.

        Returns
        -------
        Path | None
            Path to latest checkpoint, or None.
        """
        checkpoint_file = Path("checkpoint")
        if checkpoint_file.exists():
            try:
                latest = Path(checkpoint_file.read_text().strip())
                if latest.exists():
                    return latest
            except Exception:
                pass

        # Fallback: find newest checkpoint file
        checkpoints = list(Path(".").glob(f"{self.config.save_ckpt}*.pt"))
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)

        return None

    def save_final(
        self,
        step: int,
        wrapper: torch.nn.Module,
        lr: float = 0.0,
    ) -> Path | None:
        """Save final checkpoint at end of training.

        Parameters
        ----------
        step : int
            Final step number.
        wrapper : torch.nn.Module
            Model wrapper.
        lr : float
            Final learning rate.

        Returns
        -------
        Path | None
            Path to saved checkpoint.
        """
        if self.rank != 0:
            return None

        # Get unwrapped module
        module = wrapper
        if dist.is_available() and dist.is_initialized():
            if hasattr(wrapper, "module"):
                module = wrapper.module

        # Update training info
        if hasattr(module, "train_infos"):
            module.train_infos["lr"] = float(lr)
            module.train_infos["step"] = step

        save_path = Path(self.config.save_ckpt + f"-{step}.pt")

        checkpoint = {
            "model": module.state_dict(),
            "step": step,
            "lr": float(lr),
        }

        torch.save(checkpoint, save_path)
        self.latest_model = save_path

        symlink_prefix_files(save_path.stem, self.config.save_ckpt)

        with open("checkpoint", "w") as f:
            f.write(str(save_path))

        log.info(f"Saved final checkpoint to {save_path}")
        return save_path
