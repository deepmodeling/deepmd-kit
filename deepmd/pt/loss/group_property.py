# SPDX-License-Identifier: LGPL-3.0-or-later
"""Loss for grouped frame-level properties."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.grouped import (
    GROUP_ID_KEY,
    GROUP_WEIGHT_KEY,
    POOL_MASK_KEY,
    group_data_requirements,
    normalize_group_id_tensor,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


class GroupPropertyLoss(TaskLoss):
    """Compute property loss after frame embeddings are aggregated by group."""

    def __init__(
        self,
        task_dim: int,
        var_name: str,
        loss_func: str = "mse",
        metric: list[str] | None = None,
        beta: float = 1.0,
        label_tol: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.task_dim = task_dim
        self.var_name = var_name
        self.loss_func = loss_func
        self.metric = metric or ["mae"]
        self.beta = beta
        self.label_tol = label_tol

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        del natoms, learning_rate, mae
        var_name = self.var_name
        frame_label = label[var_name]
        nframes = frame_label.shape[0]
        if frame_label.shape != (nframes, self.task_dim):
            raise ValueError(
                f"{var_name} label must have shape (nframes, {self.task_dim}); "
                f"got {frame_label.shape}."
            )

        group_id = None
        if GROUP_ID_KEY in label and label.get(f"find_{GROUP_ID_KEY}", 1) is not None:
            find_group = label.get(f"find_{GROUP_ID_KEY}", 1)
            if not torch.as_tensor(find_group).eq(0).all():
                group_id = label[GROUP_ID_KEY]
        if group_id is None:
            group_id = torch.arange(nframes, dtype=torch.long, device=frame_label.device)
        else:
            group_id = normalize_group_id_tensor(group_id, nframes).to(frame_label.device)

        model_pred = model(
            **input_dict,
            group_id=group_id,
            weight=label.get(GROUP_WEIGHT_KEY),
            pool_mask=label.get(POOL_MASK_KEY),
        )
        pred = model_pred[var_name]
        group_label = self._group_labels(frame_label, group_id, model_pred["group_id"])
        if pred.shape != group_label.shape:
            raise ValueError(
                f"Prediction shape {pred.shape} does not match grouped labels "
                f"{group_label.shape}."
            )

        loss = self._loss(pred, group_label)
        more_loss = {}
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(pred, group_label, reduction="mean").detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(pred, group_label, reduction="mean").detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(pred, group_label, reduction="mean")
            ).detach()
        return model_pred, loss, more_loss

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_func == "smooth_mae":
            return F.smooth_l1_loss(pred, target, reduction="sum", beta=self.beta)
        if self.loss_func == "mae":
            return F.l1_loss(pred, target, reduction="sum")
        if self.loss_func == "mse":
            return F.mse_loss(pred, target, reduction="sum")
        if self.loss_func == "rmse":
            return torch.sqrt(F.mse_loss(pred, target, reduction="mean"))
        raise RuntimeError(f"Unknown loss function : {self.loss_func}")

    def _group_labels(
        self,
        frame_label: torch.Tensor,
        group_id: torch.Tensor,
        group_order: torch.Tensor,
    ) -> torch.Tensor:
        grouped: list[torch.Tensor] = []
        for gid in group_order:
            values = frame_label[group_id == gid]
            first = values[0]
            if not torch.allclose(values, first.expand_as(values), atol=self.label_tol):
                raise ValueError(
                    f"Inconsistent {self.var_name} labels within group {int(gid)}."
                )
            grouped.append(first)
        return torch.stack(grouped, dim=0).to(env.GLOBAL_PT_FLOAT_PRECISION)

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        return [
            DataRequirementItem(
                self.var_name,
                ndof=self.task_dim,
                atomic=False,
                must=True,
                high_prec=True,
            ),
            *group_data_requirements(),
        ]
