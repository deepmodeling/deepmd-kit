# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
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


class TensorLoss(TaskLoss):
    def __init__(
        self,
        tensor_name: str,
        tensor_size: int,
        label_name: str,
        pref_atomic: float = 0.0,
        pref: float = 0.0,
        inference=False,
        **kwargs,
    ):
        r"""Construct a loss for local and global tensors.

        Parameters
        ----------
        tensor_name : str
            The name of the tensor in the model predictions to compute the loss.
        tensor_size : int
            The size (dimension) of the tensor.
        label_name : str
            The name of the tensor in the labels to compute the loss.
        pref_atomic : float
            The prefactor of the weight of atomic loss. It should be larger than or equal to 0.
        pref : float
            The prefactor of the weight of global loss. It should be larger than or equal to 0.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.tensor_name = tensor_name
        self.tensor_size = tensor_size
        self.label_name = label_name
        self.local_weight = pref_atomic
        self.global_weight = pref
        self.inference = inference

        assert (
            self.local_weight >= 0.0 and self.global_weight >= 0.0
        ), "Can not assign negative weight to `pref` and `pref_atomic`"
        self.has_local_weight = self.local_weight > 0.0 or inference
        self.has_global_weight = self.global_weight > 0.0 or inference
        assert self.has_local_weight or self.has_global_weight, AssertionError(
            "Can not assian zero weight both to `pref` and `pref_atomic`"
        )

    def forward(self, input_dict, model, label, natoms, learning_rate=0.0, mae=False):
        """Return loss on local and global tensors.

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
        del learning_rate, mae
        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        if (
            self.has_local_weight
            and self.tensor_name in model_pred
            and "atom_" + self.label_name in label
        ):
            find_local = label.get("find_" + "atom_" + self.label_name, 0.0)
            local_weight = self.local_weight * find_local
            local_tensor_pred = model_pred[self.tensor_name].reshape(
                [-1, natoms, self.tensor_size]
            )
            local_tensor_label = label["atom_" + self.label_name].reshape(
                [-1, natoms, self.tensor_size]
            )
            diff = (local_tensor_pred - local_tensor_label).reshape(
                [-1, self.tensor_size]
            )
            if "mask" in model_pred:
                diff = diff[model_pred["mask"].reshape([-1]).bool()]
            l2_local_loss = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss[f"l2_local_{self.tensor_name}_loss"] = self.display_if_exist(
                    l2_local_loss.detach(), find_local
                )
            loss += local_weight * l2_local_loss
            rmse_local = l2_local_loss.sqrt()
            more_loss[f"rmse_local_{self.tensor_name}"] = self.display_if_exist(
                rmse_local.detach(), find_local
            )
        if (
            self.has_global_weight
            and "global_" + self.tensor_name in model_pred
            and self.label_name in label
        ):
            find_global = label.get("find_" + self.label_name, 0.0)
            global_weight = self.global_weight * find_global
            global_tensor_pred = model_pred["global_" + self.tensor_name].reshape(
                [-1, self.tensor_size]
            )
            global_tensor_label = label[self.label_name].reshape([-1, self.tensor_size])
            diff = global_tensor_pred - global_tensor_label
            if "mask" in model_pred:
                atom_num = model_pred["mask"].sum(-1, keepdim=True)
                l2_global_loss = torch.mean(
                    torch.sum(torch.square(diff) * atom_num, dim=0) / atom_num.sum()
                )
                atom_num = torch.mean(atom_num.float())
            else:
                atom_num = natoms
                l2_global_loss = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss[f"l2_global_{self.tensor_name}_loss"] = self.display_if_exist(
                    l2_global_loss.detach(), find_global
                )
            loss += global_weight * l2_global_loss
            rmse_global = l2_global_loss.sqrt() / atom_num
            more_loss[f"rmse_global_{self.tensor_name}"] = self.display_if_exist(
                rmse_global.detach(), find_global
            )
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_local_weight:
            label_requirement.append(
                DataRequirementItem(
                    "atomic_" + self.label_name,
                    ndof=self.tensor_size,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_global_weight:
            label_requirement.append(
                DataRequirementItem(
                    self.label_name,
                    ndof=self.tensor_size,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
