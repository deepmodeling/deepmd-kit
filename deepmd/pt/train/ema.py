#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

import logging
from contextlib import (
    contextmanager,
)
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

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

EMA_CHECKPOINT_KEY = "ema"
EMA_DECAY_KEY = "decay"
EMA_MODEL_STATE_KEY = "model"
EMA_VALIDATION_STATE_KEY = "validation_state"

log = logging.getLogger(__name__)


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


class ModelEMA:
    """Maintain an exponential moving average of model parameters.

    This helper assumes DDP/ZeRO-1 style training where every rank owns the
    same full, consistently ordered model parameters. It is not a sharded
    parameter EMA implementation.
    """

    def __init__(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
        decay: float,
        state: dict[str, Any] | None = None,
    ) -> None:
        self.decay = float(decay)
        self.shadow_params = self._clone_model_parameters(model)
        self.validation_state: dict[str, Any] = {}
        if state is not None:
            self.load_state_dict(state)

    @staticmethod
    def _named_model_parameters(
        model: torch.nn.Module | dict[str, torch.nn.Module],
    ) -> list[tuple[str, torch.nn.Parameter]]:
        """Collect all floating-point model parameters in a deterministic order."""
        if isinstance(model, dict):
            named_parameters = []
            for model_key in sorted(model):
                named_parameters.extend(
                    [
                        (f"{model_key}.{name}", param)
                        for name, param in model[model_key].named_parameters()
                        if torch.is_floating_point(param)
                    ]
                )
            return named_parameters
        return [
            (name, param)
            for name, param in model.named_parameters()
            if torch.is_floating_point(param)
        ]

    def _clone_model_parameters(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
    ) -> dict[str, torch.Tensor]:
        """Clone model parameters to initialize the EMA shadow state."""
        with torch.no_grad():
            return {
                name: param.detach().clone()
                for name, param in self._named_model_parameters(model)
            }

    def update(self, model: torch.nn.Module | dict[str, torch.nn.Module]) -> None:
        """Update EMA shadow parameters from the current model parameters."""
        with torch.no_grad():
            for name, param in self._named_model_parameters(model):
                self.shadow_params[name].lerp_(param.detach(), weight=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        """Serialize EMA state for restart."""
        return {
            EMA_DECAY_KEY: self.decay,
            EMA_MODEL_STATE_KEY: {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.shadow_params.items()
            },
            EMA_VALIDATION_STATE_KEY: deepcopy(self.validation_state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore EMA shadow parameters and validator state."""
        if EMA_DECAY_KEY in state:
            checkpoint_decay = float(state[EMA_DECAY_KEY])
            if checkpoint_decay != self.decay:
                log.warning(
                    "Ignoring EMA checkpoint decay=%s because training.ema_decay=%s "
                    "is configured.",
                    checkpoint_decay,
                    self.decay,
                )
        model_state = state.get(EMA_MODEL_STATE_KEY, {})
        if not isinstance(model_state, dict):
            raise TypeError("EMA checkpoint field `model` must be a dict.")

        current_keys = set(self.shadow_params)
        loaded_keys = set(model_state)
        missing_keys = sorted(current_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - current_keys)
        if missing_keys or unexpected_keys:
            raise KeyError(
                "EMA checkpoint parameter keys do not match the current model. "
                f"Missing keys: {missing_keys[:5]}, unexpected keys: {unexpected_keys[:5]}."
            )

        with torch.no_grad():
            for name, shadow_param in self.shadow_params.items():
                loaded_param = model_state[name]
                if not isinstance(loaded_param, torch.Tensor):
                    raise TypeError(
                        f"EMA checkpoint tensor for {name!r} must be a torch.Tensor."
                    )
                if loaded_param.shape != shadow_param.shape:
                    raise ValueError(
                        "EMA checkpoint parameter shape does not match the current "
                        f"model for {name!r}: expected {tuple(shadow_param.shape)}, "
                        f"got {tuple(loaded_param.shape)}."
                    )
                shadow_param.copy_(
                    loaded_param.to(
                        device=shadow_param.device,
                        dtype=shadow_param.dtype,
                    )
                )

        validation_state = state.get(EMA_VALIDATION_STATE_KEY, {})
        if validation_state is None:
            validation_state = {}
        if not isinstance(validation_state, dict):
            raise TypeError("EMA checkpoint field `validation_state` must be a dict.")
        self.validation_state = deepcopy(validation_state)

    @contextmanager
    def apply_shadow(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
    ) -> Iterator[None]:
        """Temporarily replace model parameters with the EMA shadow state."""
        backups: dict[str, torch.Tensor] = {}
        try:
            with torch.no_grad():
                for name, param in self._named_model_parameters(model):
                    backups[name] = param.detach().clone()
                    param.copy_(
                        self.shadow_params[name].to(
                            device=param.device,
                            dtype=param.dtype,
                        )
                    )
            yield
        finally:
            with torch.no_grad():
                for name, param in self._named_model_parameters(model):
                    if name in backups:
                        param.copy_(backups[name])
