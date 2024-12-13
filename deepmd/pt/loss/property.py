# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Union,
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
        task_dim,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        beta: float = 1.00,
        property_name: Union[str, list] = "property",
        property_dim: Union[int, list] = 1,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on property.

        Parameters
        ----------
        task_dim : float
            The output dimension of property fitting net.
        loss_func : str
            The loss function, such as "smooth_mae", "mae", "rmse".
        metric : list
            The metric such as mae, rmse which will be printed.
        beta:
            The 'beta' parameter in 'smooth_mae' loss.
        """
        super().__init__()
        self.task_dim = task_dim
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        if isinstance(property_name, str):
            property_name = [property_name]
        if isinstance(property_dim, int):
            property_dim = [property_dim]
        self.property_name = property_name
        assert self.task_dim == sum(property_dim)
        self.property_name_dim_mapping = dict(zip(property_name, property_dim))
        self.out_bias = kwargs.get("out_bias", None)
        self.out_std = kwargs.get("out_std", None)

    def forward(self, input_dict, model, label, natoms, learning_rate=0.0, mae=False):
        """Return loss on properties .

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Model inputs.
        model : torch.nn.Module
            Model to be used to output the predictions.
        label : dict[str, torch.Tensor]
            Labels.
        natoms : int
            The local atom number.

        Returns
        -------
        model_pred: dict[str, torch.Tensor]
            Model predictions.
        loss: torch.Tensor
            Loss for model to minimize.
        more_loss: dict[str, torch.Tensor]
            Other losses for display.
        """
        model_pred = model(**input_dict)
        nbz = model_pred["property"].shape[0]
        assert model_pred["property"].shape == (nbz, self.task_dim)

        concat_property = []
        for property_name in self.property_name:
            assert label[property_name].shape == (
                nbz,
                self.property_name_dim_mapping[property_name],
            )
            concat_property.append(label[property_name])
        label["property"] = torch.cat(concat_property, dim=1)
        assert label["property"].shape == (nbz, self.task_dim)

        out_std = model.atomic_model.out_std[0][0] if self.out_std is None else torch.tensor(self.out_std,device=env.DEVICE)
        out_bias = model.atomic_model.out_bias[0][0] if self.out_bias is None else torch.tensor(self.out_bias,device=env.DEVICE)
        assert len(out_std.shape) == 1
        assert out_std.shape[0] == self.task_dim

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        # loss
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(
                (label["property"] - out_bias) / out_std,
                (model_pred["property"] - out_bias) / out_std,
                reduction="sum",
                beta=self.beta,
            )
        elif self.loss_func == "mae":
            loss += F.l1_loss(
                (label["property"] - out_bias) / out_std,
                (model_pred["property"] - out_bias) / out_std,
                reduction="sum",
            )
        elif self.loss_func == "mse":
            loss += F.mse_loss(
                (label["property"] - out_bias) / out_std,
                (model_pred["property"] - out_bias) / out_std,
                reduction="sum",
            )
        elif self.loss_func == "rmse":
            loss += torch.sqrt(
                F.mse_loss(
                    (label["property"] - out_bias) / out_std,
                    (model_pred["property"] - out_bias) / out_std,
                    reduction="mean",
                )
            )
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        # more loss
        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(
                label["property"],
                model_pred["property"],
                reduction="mean",
                beta=self.beta,
            ).detach()
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(
                label["property"],
                model_pred["property"],
                reduction="mean",
            ).detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(
                label["property"],
                model_pred["property"],
                reduction="mean",
            ).detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(
                    label["property"],
                    model_pred["property"],
                    reduction="mean",
                )
            ).detach()

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        for property_name in self.property_name:
            label_requirement.append(
                DataRequirementItem(
                    property_name,
                    ndof=self.property_name_dim_mapping[property_name],
                    atomic=False,
                    must=True,
                    high_prec=True,
                )
            )
        return label_requirement
