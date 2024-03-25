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


class DOSLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate: float,
        numb_dos: int,
        start_pref_dos: float = 1.00,
        limit_pref_dos: float = 1.00,
        start_pref_cdf: float = 1000,
        limit_pref_cdf: float = 1.00,
        start_pref_ados: float = 0.0,
        limit_pref_ados: float = 0.0,
        start_pref_acdf: float = 0.0,
        limit_pref_acdf: float = 0.0,
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
        self.starter_learning_rate = starter_learning_rate
        self.numb_dos = numb_dos
        self.inference = inference

        self.start_pref_dos = start_pref_dos
        self.limit_pref_dos = limit_pref_dos
        self.start_pref_cdf = start_pref_cdf
        self.limit_pref_cdf = limit_pref_cdf

        self.start_pref_ados = start_pref_ados
        self.limit_pref_ados = limit_pref_ados
        self.start_pref_acdf = start_pref_acdf
        self.limit_pref_acdf = limit_pref_acdf

        assert (
            self.start_pref_dos >= 0.0
            and self.limit_pref_dos >= 0.0
            and self.start_pref_cdf >= 0.0
            and self.limit_pref_cdf >= 0.0
            and self.start_pref_ados >= 0.0
            and self.limit_pref_ados >= 0.0
            and self.start_pref_acdf >= 0.0
            and self.limit_pref_acdf >= 0.0
        ), "Can not assign negative weight to `pref` and `pref_atomic`"

        self.has_dos = (start_pref_dos != 0.0 and limit_pref_dos != 0.0) or inference
        self.has_cdf = (start_pref_cdf != 0.0 and limit_pref_cdf != 0.0) or inference
        self.has_ados = (start_pref_ados != 0.0 and limit_pref_ados != 0.0) or inference
        self.has_acdf = (start_pref_acdf != 0.0 and limit_pref_acdf != 0.0) or inference

        assert (
            self.has_dos or self.has_cdf or self.has_ados or self.has_acdf
        ), AssertionError("Can not assian zero weight both to `pref` and `pref_atomic`")

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

        coef = learning_rate / self.starter_learning_rate
        pref_dos = (
            self.limit_pref_dos + (self.start_pref_dos - self.limit_pref_dos) * coef
        )
        pref_cdf = (
            self.limit_pref_cdf + (self.start_pref_cdf - self.limit_pref_cdf) * coef
        )
        pref_ados = (
            self.limit_pref_ados + (self.start_pref_ados - self.limit_pref_ados) * coef
        )
        pref_acdf = (
            self.limit_pref_acdf + (self.start_pref_acdf - self.limit_pref_acdf) * coef
        )

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        if self.has_ados and "atom_dos" in model_pred and "atom_dos" in label:
            find_local = label.get("find_atom_dos", 0.0)
            pref_ados = pref_ados * find_local
            local_tensor_pred_dos = model_pred["atom_dos"].reshape(
                [-1, natoms, self.numb_dos]
            )
            local_tensor_label_dos = label["atom_dos"].reshape(
                [-1, natoms, self.numb_dos]
            )
            diff = (local_tensor_pred_dos - local_tensor_label_dos).reshape(
                [-1, self.numb_dos]
            )
            if "mask" in model_pred:
                diff = diff[model_pred["mask"].reshape([-1]).bool()]
            l2_local_loss_dos = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss["l2_local_dos_loss"] = self.display_if_exist(
                    l2_local_loss_dos.detach(), find_local
                )
            loss += pref_ados * l2_local_loss_dos
            rmse_local_dos = l2_local_loss_dos.sqrt()
            more_loss["rmse_local_dos"] = self.display_if_exist(
                rmse_local_dos.detach(), find_local
            )
        if self.has_acdf and "atom_dos" in model_pred and "atom_dos" in label:
            find_local = label.get("find_atom_dos", 0.0)
            pref_acdf = pref_acdf * find_local
            local_tensor_pred_cdf = torch.cusum(
                model_pred["atom_dos"].reshape([-1, natoms, self.numb_dos]), dim=-1
            )
            local_tensor_label_cdf = torch.cusum(
                label["atom_dos"].reshape([-1, natoms, self.numb_dos]), dim=-1
            )
            diff = (local_tensor_pred_cdf - local_tensor_label_cdf).reshape(
                [-1, self.numb_dos]
            )
            if "mask" in model_pred:
                diff = diff[model_pred["mask"].reshape([-1]).bool()]
            l2_local_loss_cdf = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss["l2_local_cdf_loss"] = self.display_if_exist(
                    l2_local_loss_cdf.detach(), find_local
                )
            loss += pref_acdf * l2_local_loss_cdf
            rmse_local_cdf = l2_local_loss_cdf.sqrt()
            more_loss["rmse_local_cdf"] = self.display_if_exist(
                rmse_local_cdf.detach(), find_local
            )
        if self.has_dos and "dos" in model_pred and "dos" in label:
            find_global = label.get("find_dos", 0.0)
            pref_dos = pref_dos * find_global
            global_tensor_pred_dos = model_pred["dos"].reshape([-1, self.numb_dos])
            global_tensor_label_dos = label["dos"].reshape([-1, self.numb_dos])
            diff = global_tensor_pred_dos - global_tensor_label_dos
            if "mask" in model_pred:
                atom_num = model_pred["mask"].sum(-1, keepdim=True)
                l2_global_loss_dos = torch.mean(
                    torch.sum(torch.square(diff) * atom_num, dim=0) / atom_num.sum()
                )
                atom_num = torch.mean(atom_num.float())
            else:
                atom_num = natoms
                l2_global_loss_dos = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss["l2_global_dos_loss"] = self.display_if_exist(
                    l2_global_loss_dos.detach(), find_global
                )
            loss += pref_dos * l2_global_loss_dos
            rmse_global_dos = l2_global_loss_dos.sqrt() / atom_num
            more_loss["rmse_global_dos"] = self.display_if_exist(
                rmse_global_dos.detach(), find_global
            )
        if self.has_cdf and "dos" in model_pred and "dos" in label:
            find_global = label.get("find_dos", 0.0)
            pref_cdf = pref_cdf * find_global
            global_tensor_pred_cdf = torch.cusum(
                model_pred["dos"].reshape([-1, self.numb_dos]), dim=-1
            )
            global_tensor_label_cdf = torch.cusum(
                label["dos"].reshape([-1, self.numb_dos]), dim=-1
            )
            diff = global_tensor_pred_cdf - global_tensor_label_cdf
            if "mask" in model_pred:
                atom_num = model_pred["mask"].sum(-1, keepdim=True)
                l2_global_loss_cdf = torch.mean(
                    torch.sum(torch.square(diff) * atom_num, dim=0) / atom_num.sum()
                )
                atom_num = torch.mean(atom_num.float())
            else:
                atom_num = natoms
                l2_global_loss_cdf = torch.mean(torch.square(diff))
            if not self.inference:
                more_loss["l2_global_cdf_loss"] = self.display_if_exist(
                    l2_global_loss_cdf.detach(), find_global
                )
            loss += pref_cdf * l2_global_loss_cdf
            rmse_global_dos = l2_global_loss_cdf.sqrt() / atom_num
            more_loss["rmse_global_cdf"] = self.display_if_exist(
                rmse_global_dos.detach(), find_global
            )
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_ados or self.has_acdf:
            label_requirement.append(
                DataRequirementItem(
                    "atom_dos",
                    ndof=self.numb_dos,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_dos or self.has_cdf:
            label_requirement.append(
                DataRequirementItem(
                    "dos",
                    ndof=self.numb_dos,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
