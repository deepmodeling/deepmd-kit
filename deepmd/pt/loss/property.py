# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    List,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)

log = logging.getLogger(__name__)


class PropertyLoss(TaskLoss):
    def __init__(
        self,
        task_num,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        **kwargs,
    ):
        r"""Construct a layer to compute loss on property.

        Parameters
        ----------
        task_num : float
            The learning rate at the start of the training.
        loss_func : str
            The loss function, such as "smooth_mae", "mae", "rmse"
        metric : list
            The metric such as mae,rmse which will be printed.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.loss_func = loss_func
        self.metric = metric
        self.task_num = task_num
        self.mean = kwargs.get("mean", 0)
        self.std = kwargs.get("std", 1)
        self.beta = kwargs.get("beta", 1.00)

    def forward(self, model_pred, label, natoms, learning_rate, mae=False):
        """Return loss on properties .

        Parameters:
        ----------
        model_pred : dict[str, torch.Tensor]
            Model predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        assert label["property"].shape[-1] == self.task_num
        assert model_pred["property"].shape[-1] == self.task_num
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        label_mean = torch.tensor(
            self.mean, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )
        label_std = torch.tensor(
            self.std, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
        )

        # loss
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(
                (label["property"] - label_mean) / label_std,
                model_pred["property"],
                reduction="sum",
                beta=self.beta,
            )
        elif self.func == "mae":
            loss += F.l1_loss(
                (label["property"] - label_mean) / label_std,
                model_pred["property"],
                reduction="sum",
            )
        elif self.func == "mse":
            loss += F.mse_loss(
                (label["property"] - label_mean) / label_std,
                model_pred["property"],
                reduction="sum",
            )
        elif self.func == "rmse":
            loss += torch.sqrt(
                F.mse_loss(
                    (label["property"] - label_mean) / label_std,
                    model_pred["property"],
                    reduction="mean",
                )
            )
        else:
            raise RuntimeError(f"Unknown loss function : {self.func}")

        # more loss
        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(
                label["property"],
                (model_pred["property"] * label_std) + label_mean,
                reduction="mean",
                beta=self.beta,
            ).detach()
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(
                label["property"],
                (model_pred["property"] * label_std) + label_mean,
                reduction="mean",
            ).detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(
                label["property"],
                (model_pred["property"] * label_std) + label_mean,
                reduction="mean",
            ).detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(
                    label["property"],
                    (model_pred["property"] * label_std) + label_mean,
                    reduction="mean",
                )
            ).detach()

        return loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "property",
                ndof=self.task_num,
                atomic=False,
                must=False,
                high_prec=True,
            )
        )
        return label_requirement
