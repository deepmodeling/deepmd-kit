# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import Any

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import TaskLoss
from deepmd.pt.utils import env
from deepmd.utils.data import DataRequirementItem

log = logging.getLogger(__name__)


class XASLoss(TaskLoss):
    """Loss for XAS spectrum fitting via property fitting + sel_type reduction.

    The model outputs per-atom property vectors (atom_xas).  For each frame
    this loss selects the atoms of type ``sel_type`` (read from ``sel_type.npy``
    in each training system) and takes their mean, then computes a loss against
    the per-frame XAS label.

    Parameters
    ----------
    task_dim : int
        Output dimension of the fitting net (e.g. 102 = E_min + E_max + 100 pts).
    var_name : str
        Property name, must match ``property_name`` in the fitting config.
    loss_func : str
        One of ``smooth_mae``, ``mae``, ``mse``, ``rmse``.
    metric : list[str]
        Metrics to display during training.
    beta : float
        Beta parameter for smooth_l1 loss.
    """

    def __init__(
        self,
        task_dim: int,
        var_name: str = "xas",
        loss_func: str = "smooth_mae",
        metric: list[str] = ["mae"],
        beta: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.task_dim = task_dim
        self.var_name = var_name
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred = model(**input_dict)

        # per-atom outputs: [nf, nloc, task_dim]
        atom_prop = model_pred[f"atom_{self.var_name}"]
        atype = input_dict["atype"]  # [nf, nloc]

        # sel_type from label: [nf, 1] float → [nf] int
        sel_type = label["sel_type"][:, 0].long()

        # element-wise mean: for each frame average over atoms of sel_type
        nf, nloc, td = atom_prop.shape
        pred = torch.zeros(
            nf, td, dtype=atom_prop.dtype, device=atom_prop.device
        )
        for i in range(nf):
            t = int(sel_type[i].item())
            mask = (atype[i] == t).unsqueeze(-1)  # [nloc, 1]
            count = mask.sum().clamp(min=1)
            pred[i] = (atom_prop[i] * mask).sum(dim=0) / count

        label_xas = label[self.var_name]  # [nf, task_dim]

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(pred, label_xas, reduction="sum", beta=self.beta)
        elif self.loss_func == "mae":
            loss += F.l1_loss(pred, label_xas, reduction="sum")
        elif self.loss_func == "mse":
            loss += F.mse_loss(pred, label_xas, reduction="sum")
        elif self.loss_func == "rmse":
            loss += torch.sqrt(F.mse_loss(pred, label_xas, reduction="mean"))
        else:
            raise RuntimeError(f"Unknown loss function: {self.loss_func}")

        more_loss: dict[str, torch.Tensor] = {}
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(pred, label_xas, reduction="mean").detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(pred, label_xas, reduction="mean")
            ).detach()

        model_pred[self.var_name] = pred
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Declare required data files: xas label + sel_type."""
        return [
            DataRequirementItem(
                self.var_name,
                ndof=self.task_dim,
                atomic=False,
                must=True,
                high_prec=True,
            ),
            DataRequirementItem(
                "sel_type",
                ndof=1,
                atomic=False,
                must=True,
                high_prec=False,
            ),
        ]
