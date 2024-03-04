# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

import torch
import torch.nn.functional as F

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


class EnergyStdLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate=1.0,
        start_pref_e=0.0,
        limit_pref_e=0.0,
        start_pref_f=0.0,
        limit_pref_f=0.0,
        start_pref_v=0.0,
        limit_pref_v=0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        use_l1_all: bool = False,
        inference=False,
        **kwargs,
    ):
        r"""Construct a layer to compute loss on energy, force and virial.

        Parameters
        ----------
        starter_learning_rate : float
            The learning rate at the start of the training.
        start_pref_e : float
            The prefactor of energy loss at the start of the training.
        limit_pref_e : float
            The prefactor of energy loss at the end of the training.
        start_pref_f : float
            The prefactor of force loss at the start of the training.
        limit_pref_f : float
            The prefactor of force loss at the end of the training.
        start_pref_v : float
            The prefactor of virial loss at the start of the training.
        limit_pref_v : float
            The prefactor of virial loss at the end of the training.
        start_pref_ae : float
            The prefactor of atomic energy loss at the start of the training.
        limit_pref_ae : float
            The prefactor of atomic energy loss at the end of the training.
        start_pref_pf : float
            The prefactor of atomic prefactor force loss at the start of the training.
        limit_pref_pf : float
            The prefactor of atomic prefactor force loss at the end of the training.
        use_l1_all : bool
            Whether to use L1 loss, if False (default), it will use L2 loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        **kwargs
            Other keyword arguments.
        """
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_f = (start_pref_f != 0.0 and limit_pref_f != 0.0) or inference
        self.has_v = (start_pref_v != 0.0 and limit_pref_v != 0.0) or inference

        # TODO need support for atomic energy and atomic pref
        self.has_ae = (start_pref_ae != 0.0 and limit_pref_ae != 0.0) or inference
        self.has_pf = (start_pref_pf != 0.0 and limit_pref_pf != 0.0) or inference

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.use_l1_all = use_l1_all
        self.inference = inference

    def forward(self, model_pred, label, natoms, learning_rate, mae=False):
        """Return loss on loss and force.

        Args:
        - natoms: Tell atom count.
        - p_energy: Predicted energy of all atoms.
        - p_force: Predicted force per atom.
        - l_energy: Actual energy of all atoms.
        - l_force: Actual force per atom.

        Returns
        -------
        - loss: Loss to minimize.
        """
        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        loss = torch.tensor(0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            if not self.use_l1_all:
                l2_ener_loss = torch.mean(
                    torch.square(model_pred["energy"] - label["energy"])
                )
                if not self.inference:
                    more_loss["l2_ener_loss"] = l2_ener_loss.detach()
                loss += atom_norm * (pref_e * l2_ener_loss)
                rmse_e = l2_ener_loss.sqrt() * atom_norm
                more_loss["rmse_e"] = rmse_e.detach()
                # more_loss['log_keys'].append('rmse_e')
            else:  # use l1 and for all atoms
                l1_ener_loss = F.l1_loss(
                    model_pred["energy"].reshape(-1),
                    label["energy"].reshape(-1),
                    reduction="sum",
                )
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = F.l1_loss(
                    model_pred["energy"].reshape(-1),
                    label["energy"].reshape(-1),
                    reduction="mean",
                ).detach()
                # more_loss['log_keys'].append('rmse_e')
            if mae:
                mae_e = (
                    torch.mean(torch.abs(model_pred["energy"] - label["energy"]))
                    * atom_norm
                )
                more_loss["mae_e"] = mae_e.detach()
                mae_e_all = torch.mean(
                    torch.abs(model_pred["energy"] - label["energy"])
                )
                more_loss["mae_e_all"] = mae_e_all.detach()

        if self.has_f and "force" in model_pred and "force" in label:
            if "force_target_mask" in model_pred:
                force_target_mask = model_pred["force_target_mask"]
            else:
                force_target_mask = None
            if not self.use_l1_all:
                if force_target_mask is not None:
                    diff_f = (label["force"] - model_pred["force"]) * force_target_mask
                    force_cnt = force_target_mask.squeeze(-1).sum(-1)
                    l2_force_loss = torch.mean(
                        torch.square(diff_f).mean(-1).sum(-1) / force_cnt
                    )
                else:
                    diff_f = label["force"] - model_pred["force"]
                    l2_force_loss = torch.mean(torch.square(diff_f))
                if not self.inference:
                    more_loss["l2_force_loss"] = l2_force_loss.detach()
                loss += (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_f = l2_force_loss.sqrt()
                more_loss["rmse_f"] = rmse_f.detach()
            else:
                l1_force_loss = F.l1_loss(
                    label["force"], model_pred["force"], reduction="none"
                )
                if force_target_mask is not None:
                    l1_force_loss *= force_target_mask
                    force_cnt = force_target_mask.squeeze(-1).sum(-1)
                    more_loss["mae_f"] = (
                        l1_force_loss.mean(-1).sum(-1) / force_cnt
                    ).mean()
                    l1_force_loss = (l1_force_loss.sum(-1).sum(-1) / force_cnt).sum()
                else:
                    more_loss["mae_f"] = l1_force_loss.mean().detach()
                    l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
                loss += (pref_f * l1_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            if mae:
                mae_f = torch.mean(torch.abs(diff_f))
                more_loss["mae_f"] = mae_f.detach()

        if self.has_v and "virial" in model_pred and "virial" in label:
            diff_v = label["virial"] - model_pred["virial"].reshape(-1, 9)
            l2_virial_loss = torch.mean(torch.square(diff_v))
            if not self.inference:
                more_loss["l2_virial_loss"] = l2_virial_loss.detach()
            loss += atom_norm * (pref_v * l2_virial_loss)
            rmse_v = l2_virial_loss.sqrt() * atom_norm
            more_loss["rmse_v"] = rmse_v.detach()
            if mae:
                mae_v = torch.mean(torch.abs(diff_v)) * atom_norm
                more_loss["mae_v"] = mae_v.detach()
        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return loss, more_loss

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        if self.has_e:
            label_requirement.append(
                DataRequirementItem(
                    "energy",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=True,
                )
            )
        if self.has_f:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_v:
            label_requirement.append(
                DataRequirementItem(
                    "virial",
                    ndof=9,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_ae:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_pf:
            label_requirement.append(
                DataRequirementItem(
                    "atom_pref",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                )
            )
        return label_requirement
