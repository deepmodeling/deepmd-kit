# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Generator,
)
from contextlib import (
    contextmanager,
)
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.multi_task import (
    apply_shared_links,
)

log = logging.getLogger(__name__)


def _share_descriptor(
    link_model: torch.nn.Module,
    link_type: str,
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    resume: bool,
) -> None:
    del link_model, link_type
    link_class.share_params(
        base_class,
        shared_level,
        model_prob=model_prob,
        resume=resume,
    )


def _share_fitting(
    link_class: Any,
    base_class: Any,
    shared_level: int,
    model_prob: float,
    *,
    protection: float,
    resume: bool,
) -> None:
    link_class.share_params(
        base_class,
        shared_level,
        model_prob=model_prob,
        protection=protection,
        resume=resume,
    )


class ModelWrapper(torch.nn.Module):
    """Model wrapper that bundles model(s) and loss(es).

    Supports both single-task and multi-task training.

    Parameters
    ----------
    model : torch.nn.Module or dict
        Single model or dict of models keyed by task name.
    loss : torch.nn.Module or dict or None
        Single loss or dict of losses keyed by task name.
    model_params : dict, optional
        Model parameters to store as extra state.
    """

    def __init__(
        self,
        model: torch.nn.Module | dict,
        loss: torch.nn.Module | dict | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_params = model_params if model_params is not None else {}
        self.train_infos: dict[str, Any] = {
            "lr": 0,
            "step": 0,
        }
        self.multi_task = False
        self.model = torch.nn.ModuleDict()
        # Model
        if isinstance(model, torch.nn.Module):
            self.model["Default"] = model
        elif isinstance(model, dict):
            self.multi_task = True
            for task_key in model:
                assert isinstance(model[task_key], torch.nn.Module), (
                    f"{task_key} in model_dict is not a torch.nn.Module!"
                )
                self.model[task_key] = model[task_key]
        # Loss — dpmodel losses are not nn.Module, so store in a plain dict.
        self.loss: dict[str, Any] | None = None
        if loss is not None:
            if isinstance(loss, dict):
                self.loss = dict(loss)
            else:
                self.loss = {"Default": loss}
        self.inference_only = self.loss is None

    def share_params(
        self,
        shared_links: dict[str, Any],
        model_key_prob_map: dict,
        data_stat_protect: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """Share parameters between models following rules in shared_links.

        Parameters
        ----------
        shared_links : dict
            Sharing rules from ``preprocess_shared_params``.
        model_key_prob_map : dict
            Probability map for each model key (for fitting_net stat weighting).
        data_stat_protect : float
            Protection value for standard deviation computation.
        resume : bool
            Whether resuming from checkpoint.
        """
        apply_shared_links(
            self.model,
            shared_links,
            share_descriptor=_share_descriptor,
            share_fitting=_share_fitting,
            model_key_prob_map=model_key_prob_map,
            data_stat_protect=data_stat_protect,
            resume=resume,
            logger=log,
        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        cur_lr: float | torch.Tensor | None = None,
        label: dict[str, torch.Tensor] | None = None,
        task_key: str | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict | None]:
        if not self.multi_task:
            task_key = "Default"
        else:
            assert task_key is not None, (
                f"Multitask model must specify the inference task! "
                f"Supported tasks are {list(self.model.keys())}."
            )
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": box,
            "do_atomic_virial": do_atomic_virial,
            "fparam": fparam,
            "aparam": aparam,
            "charge_spin": charge_spin,
        }
        # ``spin`` (native or virtual-atom magnetic moment) is only accepted
        # by spin-capable model forward()s; mirrors
        # ``deepmd.pt.train.wrapper.ModelWrapper.forward``'s ``has_spin`` gate
        # so non-spin models (whose forward() has no ``spin`` parameter) are
        # never called with an unexpected keyword argument.
        if self.model[task_key].has_spin():
            input_dict["spin"] = spin

        if self.inference_only:
            with self._frozen_parameter_context():
                model_pred = self._forward_without_loss(task_key, input_dict)
            return model_pred, None, None

        model_pred = self._forward_without_loss(task_key, input_dict)
        if label is None:
            return model_pred, None, None

        natoms = atype.shape[-1]
        loss, more_loss = self.loss[task_key](
            cur_lr,
            natoms,
            model_pred,
            label,
        )
        return model_pred, loss, more_loss

    @contextmanager
    def _frozen_parameter_context(self) -> Generator[None, None, None]:
        """
        Freeze model parameters during pure inference.

        Inference still differentiates model outputs with respect to
        coordinates for force and virial evaluation. Parameter gradients are not
        needed in that path, so disabling them keeps the autograd graph smaller
        without changing coordinate derivatives.
        """
        params = tuple(self.parameters())
        requires_grad = tuple(param.requires_grad for param in params)
        if not any(requires_grad):
            yield
            return
        for param in params:
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param, flag in zip(params, requires_grad, strict=True):
                param.requires_grad_(flag)

    def _forward_without_loss(
        self,
        task_key: str,
        input_dict: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return model predictions without constructing a loss."""
        return self.model[task_key](**input_dict)

    def set_extra_state(self, state: dict) -> None:
        self.model_params = state.get("model_params", {})
        self.train_infos = state.get("train_infos", {"lr": 0, "step": 0})

    def get_extra_state(self) -> dict:
        return {
            "model_params": self.model_params,
            "train_infos": self.train_infos,
        }
