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
        var_name: str,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        beta: float = 1.00,
        out_bias: Union[list, None] = None,
        out_std: Union[list, None] = None,
        intensive: bool = False,
        **kwargs,
    ) -> None:
        r"""Construct a layer to compute loss on property.

        Parameters
        ----------
        task_dim : float
            The output dimension of property fitting net.
        var_name : str
            The atomic property to fit, 'energy', 'dipole', and 'polar'.
        loss_func : str
            The loss function, such as "smooth_mae", "mae", "rmse".
        metric : list
            The metric such as mae, rmse which will be printed.
        beta : float
            The 'beta' parameter in 'smooth_mae' loss.
        out_bias : Union[list, None]
            It is the average value of the label. The shape is nkeys * ntypes * task_dim.
            In property fitting, nkeys = 1, so the shape is 1 * ntypes * task_dim.
        out_std : Union[list, None]
            It is the standard deviation of the label. The shape is nkeys * ntypes * task_dim.
            In property fitting, nkeys = 1, so the shape is 1 * ntypes * task_dim.
        intensive : bool
            Whether the property is intensive.
        """
        super().__init__()
        self.task_dim = task_dim
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.out_bias = out_bias
        self.out_std = out_std
        self.intensive = intensive
        self.var_name = var_name

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
        var_name = self.var_name
        nbz = model_pred[var_name].shape[0]
        assert model_pred[var_name].shape == (nbz, self.task_dim)
        assert label[var_name].shape == (nbz, self.task_dim)
        if not self.intensive:
            model_pred[var_name] = model_pred[var_name] / natoms
            label[var_name] = label[var_name] / natoms

        if self.out_std is None:
            out_std = model.atomic_model.out_std[0][0]
        else:
            out_std = torch.tensor(
                self.out_std, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
            )
        if out_std.shape != (self.task_dim,):
            raise ValueError(
                f"Expected out_std to have shape ({self.task_dim},), but got {out_std.shape}"
            )

        if self.out_bias is None:
            out_bias = model.atomic_model.out_bias[0][0]
        else:
            out_bias = torch.tensor(
                self.out_bias, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
            )
        if out_bias.shape != (self.task_dim,):
            raise ValueError(
                f"Expected out_bias to have shape ({self.task_dim},), but got {out_bias.shape}"
            )

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        # loss
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(
                (label[var_name] - out_bias) / out_std,
                (model_pred[var_name] - out_bias) / out_std,
                reduction="sum",
                beta=self.beta,
            )
        elif self.loss_func == "mae":
            loss += F.l1_loss(
                (label[var_name] - out_bias) / out_std,
                (model_pred[var_name] - out_bias) / out_std,
                reduction="sum",
            )
        elif self.loss_func == "mse":
            loss += F.mse_loss(
                (label[var_name] - out_bias) / out_std,
                (model_pred[var_name] - out_bias) / out_std,
                reduction="sum",
            )
        elif self.loss_func == "rmse":
            loss += torch.sqrt(
                F.mse_loss(
                    (label[var_name] - out_bias) / out_std,
                    (model_pred[var_name] - out_bias) / out_std,
                    reduction="mean",
                )
            )
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        # more loss
        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(
                label[var_name],
                model_pred[var_name],
                reduction="mean",
                beta=self.beta,
            ).detach()
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(
                label[var_name],
                model_pred[var_name],
                reduction="mean",
            ).detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(
                label[var_name],
                model_pred[var_name],
                reduction="mean",
            ).detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(
                    label[var_name],
                    model_pred[var_name],
                    reduction="mean",
                )
            ).detach()

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                self.var_name,
                ndof=self.task_dim,
                atomic=False,
                must=True,
                high_prec=True,
            )
        )
        return label_requirement
