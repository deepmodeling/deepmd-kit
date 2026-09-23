# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

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
from deepmd.utils.version import (
    check_version_compatibility,
)


class GridDensityLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_d: float = 0.0,
        limit_pref_d: float = 0.0,
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""Construct a layer to compute loss on grid density.

        The residual is masked by the grid-point mask (excluded points do
        not contribute) and reduced per frame before averaging over the
        batch, consistent with the other losses in the package.

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
        self.has_d = (start_pref_d != 0.0 or limit_pref_d != 0.0) or inference

        self.start_pref_d = start_pref_d
        self.limit_pref_d = limit_pref_d
        self.inference = inference

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
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
        if self.has_d and "density" in model_pred and "density" in label:
            find_density = label.get("find_density", 1.0)
            density_pred = model_pred["density"]
            density_label = label["density"]
            pref_d = pref_d * find_density
            # mask out excluded grid points (the atomic model zeroes them
            # and returns the mask) and reduce per frame, consistent with
            # the other losses in the package, so that frames with larger
            # grids do not dominate the batch
            mask = model_pred.get("mask", None)
            nframes = density_pred.shape[0]
            if mask is None:
                mask_f = torch.ones(
                    density_pred.shape[:2],
                    dtype=density_pred.dtype,
                    device=density_pred.device,
                )
            else:
                mask_f = mask.reshape(density_pred.shape[:2]).to(density_pred.dtype)
            mask_sum = mask_f.sum(dim=-1).clamp(min=1.0)
            residual = (
                density_label.reshape(nframes, -1) - density_pred.reshape(nframes, -1)
            ) * mask_f
            l2_density_loss = torch.square(residual).sum(dim=-1).div(mask_sum).mean()
            rmse_d = l2_density_loss.sqrt()
            more_loss["rmse_d"] = self.display_if_exist(rmse_d.detach(), find_density)
            l1_density_loss = residual.abs().sum(dim=-1).div(mask_sum).mean()
            mae_d = l1_density_loss
            # minimise the squared error, consistent with every other loss
            # in the package; the absolute error is only for display
            loss += (pref_d * l2_density_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            more_loss["mae_d"] = self.display_if_exist(mae_d.detach(), find_density)
        elif not self.inference and "density" in model_pred:
            # the density term is disabled (zero prefactors), but the graph
            # must stay connected so that backward() does not fail
            loss = loss + (model_pred["density"].sum() * 0.0).to(
                GLOBAL_PT_FLOAT_PRECISION
            )
        return model_pred, loss, more_loss

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        return {
            "@class": "GridDensityLoss",
            "@version": 1,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_d": self.start_pref_d,
            "limit_pref_d": self.limit_pref_d,
        }

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
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation.

        Only the density label is declared here; the grid is a model input
        and is declared via ``get_additional_data_requirement``.
        """
        label_requirement = []
        if self.has_d:
            # the density label is the only supervision signal of this model:
            # a missing file must abort training, not silently optimise nothing
            label_requirement.append(
                DataRequirementItem(
                    "density",
                    ndof=1,
                    atomic=False,
                    must=True,
                    high_prec=True,
                    special_shape="frame_major",
                )
            )
        return label_requirement
