# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from functools import (
    partial,
)
from typing import (
    Any,
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


class PopulationLoss(TaskLoss):
    def __init__(
        self,
        loss_func: str = "smooth_mae",
        metric: list = ["mae"],
        starter_learning_rate: float = 1.0,
        start_pref_spin: float = 1.00,
        limit_pref_spin: float = 1.00,
        start_pref_spin_total: float = 1.00,
        limit_pref_spin_total: float = 1.00,
        start_pref_pop: float = 1.00,
        limit_pref_pop: float = 1.00,
        start_pref_pop_alpha_total: float = 1.00,
        limit_pref_pop_alpha_total: float = 1.00,
        start_pref_pop_beta_total: float = 1.00,
        limit_pref_pop_beta_total: float = 1.00,
        beta: float = 1.00,
        **kwargs: Any,
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
        starter_learning_rate : float
            The learning rate for the model.
        start_pref_m : float
            The starting value for pref_m.
        limit_pref_m : float
            The limit value for pref_m.
        start_pref_t : float
            The starting value for pref_t.
        limit_pref_t : float
            The limit value for pref_t.
        beta : float
            The 'beta' parameter in 'smooth_mae' loss.
        """
        super().__init__()
        self.task_dim = 2
        self.var_name = "atom_population"
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.starter_learning_rate = starter_learning_rate

        self.start_pref_spin = start_pref_spin
        self.limit_pref_spin = limit_pref_spin
        self.start_pref_spin_total = start_pref_spin_total
        self.limit_pref_spin_total = limit_pref_spin_total
        self.start_pref_pop = start_pref_pop
        self.limit_pref_pop = limit_pref_pop
        self.start_pref_pop_alpha_total = start_pref_pop_alpha_total
        self.limit_pref_pop_alpha_total = limit_pref_pop_alpha_total
        self.start_pref_pop_beta_total = start_pref_pop_beta_total
        self.limit_pref_pop_beta_total = limit_pref_pop_beta_total
        assert (
            self.start_pref_spin >= 0.0
            and self.limit_pref_spin >= 0.0
            and self.start_pref_spin_total >= 0.0
            and self.limit_pref_spin_total >= 0.0
            and self.start_pref_pop >= 0.0
            and self.limit_pref_pop >= 0.0
            and self.start_pref_pop_alpha_total >= 0.0
            and self.limit_pref_pop_alpha_total >= 0.0
            and self.start_pref_pop_beta_total >= 0.0
            and self.limit_pref_pop_beta_total >= 0.0
        ), "Can not assign negative weight to `pref` and `pref_atomic`"

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
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

        coef = learning_rate / self.starter_learning_rate
        pref_spin = (
            self.limit_pref_spin + (self.start_pref_spin - self.limit_pref_spin) * coef
        )
        pref_spin_total = (
            self.limit_pref_spin_total
            + (self.start_pref_spin_total - self.limit_pref_spin_total) * coef
        )
        pref_pop = (
            self.limit_pref_pop + (self.start_pref_pop - self.limit_pref_pop) * coef
        )
        pref_pop_alpha_total = (
            self.limit_pref_pop_alpha_total
            + (self.start_pref_pop_alpha_total - self.limit_pref_pop_alpha_total) * coef
        )
        pref_pop_beta_total = (
            self.limit_pref_pop_beta_total
            + (self.start_pref_pop_beta_total - self.limit_pref_pop_beta_total) * coef
        )

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}

        pop_pred = model_pred["population"].reshape([-1, self.task_dim])
        pop_label = label["atom_population"].reshape([-1, self.task_dim])

        spin_pred = torch.sub(pop_pred[:, 0], pop_pred[:, 1])
        spin_label = torch.sub(pop_label[:, 0], pop_label[:, 1])

        spin_total_pred = torch.sum(spin_pred)
        spin_total_label = torch.sum(spin_label)
        pop_alpha_total_pred = torch.sum(pop_pred[:, 0])
        pop_beta_total_pred = torch.sum(pop_pred[:, 1])
        pop_alpha_total_label = torch.sum(pop_label[:, 0])
        pop_beta_total_label = torch.sum(pop_label[:, 1])

        loss_func = partial(F.l1_loss, reduction="sum")

        spin_loss = loss_func(input=spin_pred, target=spin_label)
        spin_total_loss = loss_func(input=spin_total_pred, target=spin_total_label)
        pop_loss = loss_func(input=pop_pred, target=pop_label)
        pop_alpha_total_loss = loss_func(
            input=pop_alpha_total_pred, target=pop_alpha_total_label
        )
        pop_beta_total_loss = loss_func(
            input=pop_beta_total_pred, target=pop_beta_total_label
        )

        loss += (
            pref_spin * spin_loss
            + pref_spin_total * spin_total_loss
            + pref_pop * pop_loss
            + pref_pop_alpha_total * pop_alpha_total_loss
            + pref_pop_beta_total * pop_beta_total_loss
        )

        more_loss["spin_total"] = spin_total_pred
        more_loss["spin_loss"] = spin_loss
        more_loss["spin_total_loss"] = spin_total_loss
        more_loss["pop_loss"] = pop_loss
        more_loss["pop_alpha_total_loss"] = pop_alpha_total_loss
        more_loss["pop_beta_total_loss"] = pop_beta_total_loss

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "atomic_population",
                ndof=self.task_dim,
                atomic=True,
                must=True,
                high_prec=True,
            )
        )
        return label_requirement
