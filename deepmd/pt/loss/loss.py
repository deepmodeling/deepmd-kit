# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    NoReturn,
)

import torch

from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)


class TaskLoss(torch.nn.Module, ABC, make_plugin_registry("loss")):
    def __init__(self, **kwargs: Any) -> None:
        """Construct loss."""
        super().__init__()

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float | torch.Tensor,
    ) -> NoReturn:
        """Return loss ."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        pass

    @staticmethod
    def _inject_atom_mask(
        model_pred: dict[str, torch.Tensor],
        input_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Recover the per-atom mask from atype for mixed_type batches.

        The exported forward drops the model's per-atom mask, so reconstruct it
        here from ``atype`` (training-only).  Ghost atoms have ``atype < 0``.
        An all-ones mask is produced for non-mixed batches, keeping the loss
        bit-identical to the pre-fix behavior.

        Parameters
        ----------
        model_pred : dict[str, torch.Tensor]
            Model predictions (modified in-place).
        input_dict : dict[str, torch.Tensor]
            Model inputs; must contain ``"atype"`` for injection to occur.

        Returns
        -------
        dict[str, torch.Tensor]
            ``model_pred`` with ``"mask"`` added if not already present.
        """
        if "mask" not in model_pred and "atype" in input_dict:
            atype = input_dict["atype"]
            ref = model_pred.get("energy", input_dict.get("coord", atype))
            model_pred["mask"] = (atype >= 0).to(ref.dtype)
        return model_pred

    @staticmethod
    def display_if_exist(loss: torch.Tensor, find_property: float) -> torch.Tensor:
        """Display NaN if labeled property is not found.

        Parameters
        ----------
        loss : torch.Tensor
            the loss tensor
        find_property : float
            whether the property is found
        """
        return loss if bool(find_property) else torch.nan

    @classmethod
    def get_loss(cls, loss_params: dict) -> "TaskLoss":
        """Get the loss module by the parameters.

        By default, all the parameters are directly passed to the constructor.
        If not, override this method.

        Parameters
        ----------
        loss_params : dict
            The loss parameters

        Returns
        -------
        TaskLoss
            The loss module
        """
        loss = cls(**loss_params)
        return loss

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        raise NotImplementedError

    @classmethod
    def deserialize(cls, data: dict) -> "TaskLoss":
        """Deserialize the loss module.

        Parameters
        ----------
        data : dict
            The serialized loss module

        Returns
        -------
        TaskLoss
            The deserialized loss module
        """
        raise NotImplementedError
