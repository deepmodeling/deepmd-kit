# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent naming of exponential-moving-average artifacts."""

from pathlib import (
    Path,
)

EMA_CHECKPOINT_KEY = "ema"
EMA_DECAY_KEY = "decay"
EMA_MODEL_STATE_KEY = "model"
EMA_VALIDATION_STATE_KEY = "validation_state"


def _append_suffix(path_like: str | Path, suffix: str) -> Path:
    """Append a suffix before the final file suffix when present."""
    path = Path(path_like)
    if path.suffix:
        return path.with_name(f"{path.stem}{suffix}{path.suffix}")
    return path.with_name(f"{path.name}{suffix}")


def get_ema_checkpoint_prefix(save_ckpt: str | Path) -> str:
    """Derive the EMA checkpoint prefix from the regular checkpoint prefix."""
    return str(_append_suffix(save_ckpt, "_ema"))


def get_ema_validation_log_path(full_val_file: str | Path) -> Path:
    """Derive the EMA validation log path from the regular validation log path."""
    return _append_suffix(full_val_file, "_ema")
