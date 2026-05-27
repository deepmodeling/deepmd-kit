# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_FLOAT_PRECISION,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class GridDensityLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate=1.0,
        start_pref_d=0.0,
        limit_pref_d=0.0,
        inference=False,
        **kwargs,
    ):
        r"""Construct a layer to compute loss on grid density.

        Parameters
        ----------
        starter_learning_rate : float
            The learning rate at the start of the training.
        start_pref_d : float
            The prefactor of charge density loss at the start of the training.
        limit_pref_d : float
            The prefactor of charge density loss at the end of the training.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_d = (start_pref_d != 0.0 and limit_pref_d != 0.0) or inference

        self.start_pref_d = start_pref_d
        self.limit_pref_d = limit_pref_d
        self.inference = inference

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        """Return loss on energy and force.

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
        coef = learning_rate / self.starter_learning_rate
        pref_d = self.limit_pref_d + (self.start_pref_d - self.limit_pref_d) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_d and "density" in model_pred and "density" in label:
            density_pred = model_pred["density"]
            density_label = label["density"]
            find_density = label.get("find_density", 0.0)
            pref_d = pref_d * find_density
            density_pred_reshape = density_pred.reshape(-1)
            density_label_reshape = density_label.reshape(-1)
            l2_density_loss = torch.square(
                density_label_reshape - density_pred_reshape
            ).mean()
            rmse_d = l2_density_loss.sqrt()
            more_loss["rmse_d"] = self.display_if_exist(rmse_d.detach(), find_density)
            l1_density_loss = torch.abs(
                density_label_reshape - density_pred_reshape
            ).mean()
            loss += (pref_d * l1_density_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            mae_d = l1_density_loss
            more_loss["mae_d"] = self.display_if_exist(mae_d.detach(), find_density)
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "grid",
                ndof=3,
                atomic=True,  # the grid is defined for each atom, so it is atomic
                must=True,
                high_prec=True,
            )
        )
        if self.has_d:
            label_requirement.append(
                DataRequirementItem(
                    "density",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=True,
                )
            )
        return label_requirement
