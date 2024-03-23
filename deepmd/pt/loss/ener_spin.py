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


class EnergySpinLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate=1.0,
        start_pref_e=0.0,
        limit_pref_e=0.0,
        start_pref_fr=0.0,
        limit_pref_fr=0.0,
        start_pref_fm=0.0,
        limit_pref_fm=0.0,
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
        """Construct a layer to compute loss on energy, real force, magnetic force and virial."""
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_fr = (start_pref_fr != 0.0 and limit_pref_fr != 0.0) or inference
        self.has_fm = (start_pref_fm != 0.0 and limit_pref_fm != 0.0) or inference

        # TODO EnergySpinLoss needs support for virial, atomic energy and atomic pref
        self.has_v = (start_pref_v != 0.0 and limit_pref_v != 0.0) or inference
        self.has_ae = (start_pref_ae != 0.0 and limit_pref_ae != 0.0) or inference
        self.has_pf = (start_pref_pf != 0.0 and limit_pref_pf != 0.0) or inference

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_fr = start_pref_fr
        self.limit_pref_fr = limit_pref_fr
        self.start_pref_fm = start_pref_fm
        self.limit_pref_fm = limit_pref_fm
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.use_l1_all = use_l1_all
        self.inference = inference

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
        """Return energy loss with magnetic labels.

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
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_fr = self.limit_pref_fr + (self.start_pref_fr - self.limit_pref_fr) * coef
        pref_fm = self.limit_pref_fm + (self.start_pref_fm - self.limit_pref_fm) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        loss = torch.tensor(0.0, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            if not self.use_l1_all:
                l2_ener_loss = torch.mean(
                    torch.square(model_pred["energy"] - label["energy"])
                )
                if not self.inference:
                    more_loss["l2_ener_loss"] = self.display_if_exist(
                        l2_ener_loss.detach(), find_energy
                    )
                loss += atom_norm * (pref_e * l2_ener_loss)
                rmse_e = l2_ener_loss.sqrt() * atom_norm
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e.detach(), find_energy
                )
                # more_loss['log_keys'].append('rmse_e')
            else:  # use l1 and for all atoms
                l1_ener_loss = F.l1_loss(
                    model_pred["energy"].reshape(-1),
                    label["energy"].reshape(-1),
                    reduction="sum",
                )
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    F.l1_loss(
                        model_pred["energy"].reshape(-1),
                        label["energy"].reshape(-1),
                        reduction="mean",
                    ).detach(),
                    find_energy,
                )
                # more_loss['log_keys'].append('rmse_e')
            if mae:
                mae_e = (
                    torch.mean(torch.abs(model_pred["energy"] - label["energy"]))
                    * atom_norm
                )
                more_loss["mae_e"] = self.display_if_exist(mae_e.detach(), find_energy)
                mae_e_all = torch.mean(
                    torch.abs(model_pred["energy"] - label["energy"])
                )
                more_loss["mae_e_all"] = self.display_if_exist(
                    mae_e_all.detach(), find_energy
                )

        if self.has_fr and "force" in model_pred and "force" in label:
            find_force_r = label.get("find_force", 0.0)
            pref_fr = pref_fr * find_force_r
            if not self.use_l1_all:
                diff_fr = label["force"] - model_pred["force"]
                l2_force_real_loss = torch.mean(torch.square(diff_fr))
                if not self.inference:
                    more_loss["l2_force_r_loss"] = self.display_if_exist(
                        l2_force_real_loss.detach(), find_force_r
                    )
                loss += (pref_fr * l2_force_real_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_fr = l2_force_real_loss.sqrt()
                more_loss["rmse_fr"] = self.display_if_exist(
                    rmse_fr.detach(), find_force_r
                )
                if mae:
                    mae_fr = torch.mean(torch.abs(diff_fr))
                    more_loss["mae_fr"] = self.display_if_exist(
                        mae_fr.detach(), find_force_r
                    )
            else:
                l1_force_real_loss = F.l1_loss(
                    label["force"], model_pred["force"], reduction="none"
                )
                more_loss["mae_fr"] = self.display_if_exist(
                    l1_force_real_loss.mean().detach(), find_force_r
                )
                l1_force_real_loss = l1_force_real_loss.sum(-1).mean(-1).sum()
                loss += (pref_fr * l1_force_real_loss).to(GLOBAL_PT_FLOAT_PRECISION)

        if self.has_fm and "force_mag" in model_pred and "force_mag" in label:
            find_force_m = label.get("find_force_mag", 0.0)
            pref_fm = pref_fm * find_force_m
            nframes = model_pred["force_mag"].shape[0]
            atomic_mask = model_pred["mask_mag"].expand([-1, -1, 3])
            label_force_mag = label["force_mag"][atomic_mask].view(nframes, -1, 3)
            model_pred_force_mag = model_pred["force_mag"][atomic_mask].view(
                nframes, -1, 3
            )
            if not self.use_l1_all:
                diff_fm = label_force_mag - model_pred_force_mag
                l2_force_mag_loss = torch.mean(torch.square(diff_fm))
                if not self.inference:
                    more_loss["l2_force_m_loss"] = self.display_if_exist(
                        l2_force_mag_loss.detach(), find_force_m
                    )
                loss += (pref_fm * l2_force_mag_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_fm = l2_force_mag_loss.sqrt()
                more_loss["rmse_fm"] = self.display_if_exist(
                    rmse_fm.detach(), find_force_m
                )
                if mae:
                    mae_fm = torch.mean(torch.abs(diff_fm))
                    more_loss["mae_fm"] = self.display_if_exist(
                        mae_fm.detach(), find_force_m
                    )
            else:
                l1_force_mag_loss = F.l1_loss(
                    label_force_mag, model_pred_force_mag, reduction="none"
                )
                more_loss["mae_fm"] = self.display_if_exist(
                    l1_force_mag_loss.mean().detach(), find_force_m
                )
                l1_force_mag_loss = l1_force_mag_loss.sum(-1).mean(-1).sum()
                loss += (pref_fm * l1_force_mag_loss).to(GLOBAL_PT_FLOAT_PRECISION)

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

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
        if self.has_fr:
            label_requirement.append(
                DataRequirementItem(
                    "force",
                    ndof=3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.has_fm:
            label_requirement.append(
                DataRequirementItem(
                    "force_mag",
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
