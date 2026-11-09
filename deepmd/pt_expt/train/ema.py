#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later

from __future__ import (
    annotations,
)

import logging
from collections import (
    Counter,
)
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


def _foreach_compatible(shadow: torch.Tensor, param: torch.Tensor) -> bool:
    """Match the foreach fast path, including transposed/channels-last pairs.

    Only non-overlapping dense strided tensors with corresponding strides can
    share a fast-path group. Other supported strided pairs use scalar lerp.
    """
    return (
        shadow.layout == param.layout == torch.strided
        and torch.ops.aten.is_non_overlapping_and_dense(shadow)
        and torch.ops.aten.is_non_overlapping_and_dense(param)
        and all(
            size == 1 or shadow_stride == param_stride
            for size, shadow_stride, param_stride in zip(
                param.shape, shadow.stride(), param.stride()
            )
        )
    )


def _check_layout(name: str, tensor: torch.Tensor) -> None:
    """Reject unsupported layouts before cloning or building the update plan."""
    if tensor.layout != torch.strided:
        raise ValueError(
            f"EMA parameter {name!r} requires torch.strided layout, got {tensor.layout}."
        )


class ModelEMA:
    """Maintain an exponential moving average of model parameters.

    This helper assumes DDP/ZeRO-1 style training where every rank owns the
    same full, consistently ordered model parameters. It is not a sharded
    parameter EMA implementation.

    The per-step update path uses grouped ``torch._foreach_lerp_`` launches
    built once when EMA binds to a model. Call ``update()`` after each optimizer step. Call ``rebind(model)`` after
    replacing Parameter objects (including ``load_state_dict(assign=True)``);
    ordinary in-place state loads do not require rebinding. Only strided
    floating-point parameters are supported, not sparse parameters.
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
        self._update_groups: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = []
        self._update_fallback: list[tuple[torch.Tensor, torch.Tensor]] = []
        if state is not None:
            self.load_state_dict(state)
        self._rebuild_update_plan(model)

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
        named_parameters = self._named_model_parameters(model)
        for name, param in named_parameters:
            _check_layout(name, param)
        with torch.no_grad():
            return {name: param.detach().clone() for name, param in named_parameters}

    def _rebuild_update_plan(
        self,
        model: torch.nn.Module | dict[str, torch.nn.Module],
    ) -> None:
        """Bind live parameter refs into foreach groups for the hot update path.

        Group compatible strided pairs by ``(device, dtype)``. Other strided
        pairs and duplicate names use ordered scalar lerp. Validate the complete structure before replacing
        the plan, so a rejected rebind leaves the previous binding intact.
        """
        named_parameters = self._named_model_parameters(model)
        names = Counter(name for name, _ in named_parameters)
        current_keys = set(self.shadow_params)
        model_keys = set(names)
        if current_keys != model_keys:
            missing = sorted(current_keys - model_keys)
            unexpected = sorted(model_keys - current_keys)
            raise KeyError(
                "EMA update plan parameter keys do not match shadow state. "
                f"Missing keys: {missing[:5]}, unexpected keys: {unexpected[:5]}."
            )

        grouped: dict[
            tuple[torch.device, torch.dtype],
            tuple[list[torch.Tensor], list[torch.Tensor]],
        ] = {}
        fallback: list[tuple[torch.Tensor, torch.Tensor]] = []
        for name, param in named_parameters:
            shadow = self.shadow_params[name]
            _check_layout(name, param)
            _check_layout(name, shadow)
            if param.shape != shadow.shape:
                raise ValueError(
                    f"EMA parameter {name!r} shape mismatch: expected "
                    f"{tuple(shadow.shape)}, got {tuple(param.shape)}."
                )
            if param.device != shadow.device or param.dtype != shadow.dtype:
                raise ValueError(
                    f"EMA parameter {name!r} device/dtype mismatch: expected "
                    f"{shadow.device}/{shadow.dtype}, got {param.device}/{param.dtype}."
                )
            # Flattened multitask names can collide. Preserve the original ordered
            # scalar updates instead of racing writes to one shadow on CUDA.
            if names[name] > 1 or not _foreach_compatible(shadow, param):
                fallback.append((shadow, param))
                continue
            key = (param.device, param.dtype)
            shadows, params = grouped.setdefault(key, ([], []))
            shadows.append(shadow)
            params.append(param)

        self._update_groups = list(grouped.values())
        self._update_fallback = fallback

    def rebind(self, model: torch.nn.Module | dict[str, torch.nn.Module]) -> None:
        """Bind new parameters with the same names, shapes, devices and dtypes.

        Preserve EMA history. A failed validation leaves the old binding intact.
        """
        self._rebuild_update_plan(model)

    def update(self) -> None:
        """Update shadows from bound parameters without walking model names."""
        with torch.no_grad():
            weight = 1.0 - self.decay
            for shadows, params in self._update_groups:
                torch._foreach_lerp_(shadows, params, weight)
            for shadow, param in self._update_fallback:
                shadow.lerp_(param.detach(), weight=weight)

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
        """Restore EMA shadow parameters and validator state.

        Shadow tensor objects are preserved (values are ``copy_``'d), so an
        existing update plan remains valid. Rebind only if the model's Parameter
        objects themselves have been replaced.
        """
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
        named_parameters = self._named_model_parameters(model)
        with torch.no_grad():
            backups = {name: param.detach().clone() for name, param in named_parameters}
        try:
            with torch.no_grad():
                for name, param in named_parameters:
                    param.copy_(
                        self.shadow_params[name].to(
                            device=param.device,
                            dtype=param.dtype,
                        )
                    )
            yield
        finally:
            with torch.no_grad():
                for name, param in named_parameters:
                    param.copy_(backups[name])
