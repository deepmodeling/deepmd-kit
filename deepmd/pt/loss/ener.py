# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
    Optional,
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
        relative_f: Optional[float] = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
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
        relative_f : float
            If provided, relative force error will be used in the loss. The difference
            of force will be normalized by the magnitude of the force in the label with
            a shift given by relative_f
        enable_atom_ener_coeff : bool
            if true, the energy will be computed as \sum_i c_i E_i
        start_pref_gf : float
            The prefactor of generalized force loss at the start of the training.
        limit_pref_gf : float
            The prefactor of generalized force loss at the end of the training.
        numb_generalized_coord : int
            The dimension of generalized coordinates.
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
        self.has_ae = (start_pref_ae != 0.0 and limit_pref_ae != 0.0) or inference
        self.has_pf = (start_pref_pf != 0.0 and limit_pref_pf != 0.0) or inference
        self.has_gf = start_pref_gf != 0.0 and limit_pref_gf != 0.0

        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_f = start_pref_f
        self.limit_pref_f = limit_pref_f
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.start_pref_pf = start_pref_pf
        self.limit_pref_pf = limit_pref_pf
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.numb_generalized_coord = numb_generalized_coord
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.use_l1_all = use_l1_all
        self.inference = inference

    def forward(self, input_dict, model, label, natoms, learning_rate, mae=False):
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
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_f = self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        pref_ae = self.limit_pref_ae + (self.start_pref_ae - self.limit_pref_ae) * coef
        pref_pf = self.limit_pref_pf + (self.start_pref_pf - self.limit_pref_pf) * coef
        pref_gf = self.limit_pref_gf + (self.start_pref_gf - self.limit_pref_gf) * coef

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss = {}
        # more_loss['log_keys'] = []  # showed when validation on the fly
        # more_loss['test_keys'] = []  # showed when doing dp test
        atom_norm = 1.0 / natoms
        if self.has_e and "energy" in model_pred and "energy" in label:
            energy_pred = model_pred["energy"]
            energy_label = label["energy"]
            if self.enable_atom_ener_coeff and "atom_energy" in model_pred:
                atom_ener_pred = model_pred["atom_energy"]
                # when ener_coeff (\nu) is defined, the energy is defined as
                # E = \sum_i \nu_i E_i
                # instead of the sum of atomic energies.
                #
                # A case is that we want to train reaction energy
                # A + B -> C + D
                # E = - E(A) - E(B) + E(C) + E(D)
                # A, B, C, D could be put far away from each other
                atom_ener_coeff = label["atom_ener_coeff"]
                atom_ener_coeff = atom_ener_coeff.reshape(atom_ener_pred.shape)
                energy_pred = torch.sum(atom_ener_coeff * atom_ener_pred, dim=1)
            find_energy = label.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            if not self.use_l1_all:
                l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label))
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
                    energy_pred.reshape(-1),
                    energy_label.reshape(-1),
                    reduction="sum",
                )
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    F.l1_loss(
                        energy_pred.reshape(-1),
                        energy_label.reshape(-1),
                        reduction="mean",
                    ).detach(),
                    find_energy,
                )
                # more_loss['log_keys'].append('rmse_e')
            if mae:
                mae_e = torch.mean(torch.abs(energy_pred - energy_label)) * atom_norm
                more_loss["mae_e"] = self.display_if_exist(mae_e.detach(), find_energy)
                mae_e_all = torch.mean(torch.abs(energy_pred - energy_label))
                more_loss["mae_e_all"] = self.display_if_exist(
                    mae_e_all.detach(), find_energy
                )

        if (
            (self.has_f or self.has_pf or self.relative_f or self.has_gf)
            and "force" in model_pred
            and "force" in label
        ):
            find_force = label.get("find_force", 0.0)
            pref_f = pref_f * find_force
            force_pred = model_pred["force"]
            force_label = label["force"]
            diff_f = (force_label - force_pred).reshape(-1)

            if self.relative_f is not None:
                force_label_3 = force_label.reshape(-1, 3)
                norm_f = force_label_3.norm(dim=1, keepdim=True) + self.relative_f
                diff_f_3 = diff_f.reshape(-1, 3)
                diff_f_3 = diff_f_3 / norm_f
                diff_f = diff_f_3.reshape(-1)

            if self.has_f:
                if not self.use_l1_all:
                    l2_force_loss = torch.mean(torch.square(diff_f))
                    if not self.inference:
                        more_loss["l2_force_loss"] = self.display_if_exist(
                            l2_force_loss.detach(), find_force
                        )
                    loss += (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                    rmse_f = l2_force_loss.sqrt()
                    more_loss["rmse_f"] = self.display_if_exist(
                        rmse_f.detach(), find_force
                    )
                else:
                    l1_force_loss = F.l1_loss(force_label, force_pred, reduction="none")
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_loss.mean().detach(), find_force
                    )
                    l1_force_loss = l1_force_loss.sum(-1).mean(-1).sum()
                    loss += (pref_f * l1_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                if mae:
                    mae_f = torch.mean(torch.abs(diff_f))
                    more_loss["mae_f"] = self.display_if_exist(
                        mae_f.detach(), find_force
                    )

            if self.has_pf and "atom_pref" in label:
                atom_pref = label["atom_pref"]
                find_atom_pref = label.get("find_atom_pref", 0.0)
                pref_pf = pref_pf * find_atom_pref
                atom_pref_reshape = atom_pref.reshape(-1)
                l2_pref_force_loss = (torch.square(diff_f) * atom_pref_reshape).mean()
                if not self.inference:
                    more_loss["l2_pref_force_loss"] = self.display_if_exist(
                        l2_pref_force_loss.detach(), find_atom_pref
                    )
                loss += (pref_pf * l2_pref_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_pf = l2_pref_force_loss.sqrt()
                more_loss["rmse_pf"] = self.display_if_exist(
                    rmse_pf.detach(), find_atom_pref
                )

            if self.has_gf and "drdq" in label:
                drdq = label["drdq"]
                find_drdq = label.get("find_drdq", 0.0)
                pref_gf = pref_gf * find_drdq
                force_reshape_nframes = force_pred.reshape(-1, natoms * 3)
                force_label_reshape_nframes = force_label.reshape(-1, natoms * 3)
                drdq_reshape = drdq.reshape(-1, natoms * 3, self.numb_generalized_coord)
                gen_force_label = torch.einsum(
                    "bij,bi->bj", drdq_reshape, force_label_reshape_nframes
                )
                gen_force = torch.einsum(
                    "bij,bi->bj", drdq_reshape, force_reshape_nframes
                )
                diff_gen_force = gen_force_label - gen_force
                l2_gen_force_loss = torch.square(diff_gen_force).mean()
                if not self.inference:
                    more_loss["l2_gen_force_loss"] = self.display_if_exist(
                        l2_gen_force_loss.detach(), find_drdq
                    )
                loss += (pref_gf * l2_gen_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                rmse_gf = l2_gen_force_loss.sqrt()
                more_loss["rmse_gf"] = self.display_if_exist(
                    rmse_gf.detach(), find_drdq
                )

        if self.has_v and "virial" in model_pred and "virial" in label:
            find_virial = label.get("find_virial", 0.0)
            pref_v = pref_v * find_virial
            diff_v = label["virial"] - model_pred["virial"].reshape(-1, 9)
            l2_virial_loss = torch.mean(torch.square(diff_v))
            if not self.inference:
                more_loss["l2_virial_loss"] = self.display_if_exist(
                    l2_virial_loss.detach(), find_virial
                )
            loss += atom_norm * (pref_v * l2_virial_loss)
            rmse_v = l2_virial_loss.sqrt() * atom_norm
            more_loss["rmse_v"] = self.display_if_exist(rmse_v.detach(), find_virial)
            if mae:
                mae_v = torch.mean(torch.abs(diff_v)) * atom_norm
                more_loss["mae_v"] = self.display_if_exist(mae_v.detach(), find_virial)

        if self.has_ae and "atom_energy" in model_pred and "atom_ener" in label:
            atom_ener = model_pred["atom_energy"]
            atom_ener_label = label["atom_ener"]
            find_atom_ener = label.get("find_atom_ener", 0.0)
            pref_ae = pref_ae * find_atom_ener
            atom_ener_reshape = atom_ener.reshape(-1)
            atom_ener_label_reshape = atom_ener_label.reshape(-1)
            l2_atom_ener_loss = torch.square(
                atom_ener_label_reshape - atom_ener_reshape
            ).mean()
            if not self.inference:
                more_loss["l2_atom_ener_loss"] = self.display_if_exist(
                    l2_atom_ener_loss.detach(), find_atom_ener
                )
            loss += (pref_ae * l2_atom_ener_loss).to(GLOBAL_PT_FLOAT_PRECISION)
            rmse_ae = l2_atom_ener_loss.sqrt()
            more_loss["rmse_ae"] = self.display_if_exist(
                rmse_ae.detach(), find_atom_ener
            )

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
        if self.has_gf > 0:
            label_requirement.append(
                DataRequirementItem(
                    "drdq",
                    ndof=self.numb_generalized_coord * 3,
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        if self.enable_atom_ener_coeff:
            label_requirement.append(
                DataRequirementItem(
                    "atom_ener_coeff",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    default=1.0,
                )
            )
        return label_requirement
