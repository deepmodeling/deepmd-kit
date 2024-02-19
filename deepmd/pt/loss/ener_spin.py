# SPDX-License-Identifier: LGPL-3.0-or-later
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
        use_l1_all: bool = False,
        inference=False,
        **kwargs,
    ):
        """Construct a layer to compute loss on energy, force and virial."""
        super().__init__()
        self.starter_learning_rate = starter_learning_rate
        self.has_e = (start_pref_e != 0.0 and limit_pref_e != 0.0) or inference
        self.has_fr = (start_pref_fr != 0.0 and limit_pref_fr != 0.0) or inference
        self.has_fm = (start_pref_fm != 0.0 and limit_pref_fm != 0.0) or inference
        self.has_v = (start_pref_v != 0.0 and limit_pref_v != 0.0) or inference
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
        pref_fr = self.limit_pref_fr + (self.start_pref_fr - self.limit_pref_fr) * coef
        pref_fm = self.limit_pref_fm + (self.start_pref_fm - self.limit_pref_fm) * coef
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

        if self.has_fr and "force_real" in model_pred and "force" in label:
            if not self.use_l1_all:
                diff_fr = label["force"] - model_pred["force_real"]
                l2_force_real_loss = torch.mean(torch.square(diff_fr))
                if not self.inference:
                    more_loss["l2_force_r_loss"] = l2_force_real_loss.detach()
                loss += (pref_fr * l2_force_real_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_fr = l2_force_real_loss.sqrt()
                more_loss["rmse_fr"] = rmse_fr.detach()
            else:
                l1_force_real_loss = F.l1_loss(
                    label["force"], model_pred["force_real"], reduction="none"
                )
                more_loss["mae_fr"] = l1_force_real_loss.mean().detach()
                l1_force_real_loss = l1_force_real_loss.sum(-1).mean(-1).sum()
                loss += (pref_fr * l1_force_real_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            if mae:
                mae_fr = torch.mean(torch.abs(diff_fr))
                more_loss["mae_fr"] = mae_fr.detach()

        if self.has_fm and "force_mag" in model_pred and "force_mag" in label:
            nframes = model_pred["force_mag"].shape[0]
            atmoic_mask = model_pred["atmoic_mask"].expand([-1, -1, 3])
            label_force_mag = label["force_mag"][atmoic_mask].view(nframes, -1, 3)
            model_pred_force_mag = model_pred["force_mag"][atmoic_mask].view(
                nframes, -1, 3
            )
            if not self.use_l1_all:
                diff_fm = label_force_mag - model_pred_force_mag
                l2_force_mag_loss = torch.mean(torch.square(diff_fm))
                if not self.inference:
                    more_loss["l2_force_m_loss"] = l2_force_mag_loss.detach()
                loss += (pref_fm * l2_force_mag_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_fm = l2_force_mag_loss.sqrt()
                more_loss["rmse_fm"] = rmse_fm.detach()
            else:
                l1_force_mag_loss = F.l1_loss(
                    label_force_mag, model_pred_force_mag, reduction="none"
                )
                more_loss["mae_fm"] = l1_force_mag_loss.mean().detach()
                l1_force_mag_loss = l1_force_mag_loss.sum(-1).mean(-1).sum()
                loss += (pref_fm * l1_force_mag_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            if mae:
                mae_fm = torch.mean(torch.abs(diff_fm))
                more_loss["mae_fm"] = mae_fm.detach()

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
