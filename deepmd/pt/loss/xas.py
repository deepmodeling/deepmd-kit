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
from deepmd.utils.data import (
    DataRequirementItem,
)


class XASLoss(TaskLoss):
    """Loss for XAS spectrum fitting.

    Computes L2 loss on the reduced XAS spectrum (mean over absorbing atoms).
    An optional CDF (cumulative) loss can be added to improve spectral shape.

    The labels expected in the dataset are:

    * ``xas.npy`` : shape ``[nframes, numb_xas]`` — the mean XAS spectrum over
      absorbing atoms, in a relative energy (ΔE) grid.

    Parameters
    ----------
    starter_learning_rate : float
        Initial learning rate, used for prefactor scheduling.
    numb_xas : int
        Number of XAS energy grid points.
    start_pref_xas : float
        Starting prefactor for the XAS L2 loss.
    limit_pref_xas : float
        Limiting prefactor for the XAS L2 loss.
    start_pref_cdf : float
        Starting prefactor for the CDF L2 loss.
    limit_pref_cdf : float
        Limiting prefactor for the CDF L2 loss.
    inference : bool
        If True, output all losses regardless of prefactors.
    """

    def __init__(
        self,
        starter_learning_rate: float,
        numb_xas: int,
        start_pref_xas: float = 1.0,
        limit_pref_xas: float = 1.0,
        start_pref_cdf: float = 0.0,
        limit_pref_cdf: float = 0.0,
        inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.numb_xas = numb_xas
        self.inference = inference

        self.start_pref_xas = start_pref_xas
        self.limit_pref_xas = limit_pref_xas
        self.start_pref_cdf = start_pref_cdf
        self.limit_pref_cdf = limit_pref_cdf

        assert (
            self.start_pref_xas >= 0.0
            and self.limit_pref_xas >= 0.0
            and self.start_pref_cdf >= 0.0
            and self.limit_pref_cdf >= 0.0
        ), "Loss prefactors must be non-negative"

        self.has_xas = (start_pref_xas != 0.0 and limit_pref_xas != 0.0) or inference
        self.has_cdf = (start_pref_cdf != 0.0 and limit_pref_cdf != 0.0) or inference

        assert self.has_xas or self.has_cdf, (
            "At least one of start_pref_xas or start_pref_cdf must be non-zero"
        )

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Compute XAS loss.

        Parameters
        ----------
        input_dict : dict
            Model inputs.
        model : torch.nn.Module
            The model to evaluate.
        label : dict
            Label dict containing ``"xas"`` key.
        natoms : int
            Number of local atoms.
        learning_rate : float
            Current learning rate for prefactor scheduling.
        mae : bool
            Unused (kept for API compatibility).

        Returns
        -------
        model_pred : dict
        loss : torch.Tensor
        more_loss : dict
        """
        model_pred = model(**input_dict)

        coef = learning_rate / self.starter_learning_rate
        pref_xas = (
            self.limit_pref_xas + (self.start_pref_xas - self.limit_pref_xas) * coef
        )
        pref_cdf = (
            self.limit_pref_cdf + (self.start_pref_cdf - self.limit_pref_cdf) * coef
        )

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss: dict[str, torch.Tensor] = {}

        if self.has_xas and "xas" in model_pred and "xas" in label:
            find_xas = label.get("find_xas", 0.0)
            pref_xas = pref_xas * find_xas
            pred = model_pred["xas"].reshape([-1, self.numb_xas])
            ref = label["xas"].reshape([-1, self.numb_xas])
            diff = pred - ref
            l2_loss = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss["l2_xas_loss"] = self.display_if_exist(
                    l2_loss.detach(), find_xas
                )
            loss += pref_xas * l2_loss
            more_loss["rmse_xas"] = self.display_if_exist(
                l2_loss.sqrt().detach(), find_xas
            )

        if self.has_cdf and "xas" in model_pred and "xas" in label:
            find_xas = label.get("find_xas", 0.0)
            pref_cdf = pref_cdf * find_xas
            pred_cdf = torch.cumsum(
                model_pred["xas"].reshape([-1, self.numb_xas]), dim=-1
            )
            ref_cdf = torch.cumsum(label["xas"].reshape([-1, self.numb_xas]), dim=-1)
            diff_cdf = pred_cdf - ref_cdf
            l2_cdf_loss = torch.mean(torch.square(diff_cdf))
            if not self.inference:
                more_loss["l2_cdf_loss"] = self.display_if_exist(
                    l2_cdf_loss.detach(), find_xas
                )
            loss += pref_cdf * l2_cdf_loss
            more_loss["rmse_cdf"] = self.display_if_exist(
                l2_cdf_loss.sqrt().detach(), find_xas
            )

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements for XAS training."""
        return [
            DataRequirementItem(
                "xas",
                ndof=self.numb_xas,
                atomic=False,
                must=False,
                high_prec=False,
            )
        ]
