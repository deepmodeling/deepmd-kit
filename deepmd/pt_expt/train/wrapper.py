# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import torch

log = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    """Simplified model wrapper that bundles a model and a loss.

    Single-task only for now (no multi-task support).

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    loss : torch.nn.Module
        The loss module.
    model_params : dict, optional
        Model parameters to store as extra state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: torch.nn.Module | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_params = model_params if model_params is not None else {}
        self.train_infos: dict[str, Any] = {
            "lr": 0,
            "step": 0,
        }
        self.model = model
        self.loss = loss
        self.inference_only = self.loss is None

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        cur_lr: float | torch.Tensor | None = None,
        label: dict[str, torch.Tensor] | None = None,
        do_atomic_virial: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict | None]:
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": box,
            "do_atomic_virial": do_atomic_virial,
            "fparam": fparam,
            "aparam": aparam,
        }

        model_pred = self.model(**input_dict)

        if self.inference_only or label is None:
            return model_pred, None, None
        else:
            natoms = atype.shape[-1]
            loss, more_loss = self.loss(
                cur_lr,
                natoms,
                model_pred,
                label,
            )
            return model_pred, loss, more_loss

    def set_extra_state(self, state: dict) -> None:
        self.model_params = state.get("model_params", {})
        self.train_infos = state.get("train_infos", {"lr": 0, "step": 0})

    def get_extra_state(self) -> dict:
        return {
            "model_params": self.model_params,
            "train_infos": self.train_infos,
        }
