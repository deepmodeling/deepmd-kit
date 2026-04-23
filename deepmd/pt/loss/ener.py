# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
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
from deepmd.utils.loss import (
    resolve_huber_deltas,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def custom_huber_loss(
    predictions: torch.Tensor, targets: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    error = targets - predictions
    abs_error = torch.abs(error)
    quadratic_loss = 0.5 * torch.pow(error, 2)
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = torch.where(abs_error <= delta, quadratic_loss, linear_loss)
    return torch.mean(loss)


class EnergyStdLoss(TaskLoss):
    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 0.0,
        limit_pref_e: float = 0.0,
        start_pref_f: float = 0.0,
        limit_pref_f: float = 0.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: float | None = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        loss_func: str = "mse",
        inference: bool = False,
        use_huber: bool = False,
        use_default_pf: bool = False,
        f_use_norm: bool = False,
        huber_delta: float | list[float] = 0.01,
        **kwargs: Any,
    ) -> None:
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
        loss_func : str
            Loss function type. Options: 'mse' (Mean Squared Error, L2 loss, default) or 'mae' (Mean Absolute Error, L1 loss).
            MAE loss is less sensitive to outliers compared to MSE loss.
        inference : bool
            If true, it will output all losses found in output, ignoring the pre-factors.
        use_default_pf : bool
            If true, use default atom_pref of 1.0 for all atoms when atom_pref data is not provided.
            This allows using the prefactor force loss (pf) without requiring atom_pref.npy files.
        use_huber : bool
            Enables Huber loss calculation for energy/force/virial terms with user-defined threshold delta (D).
            The loss function smoothly transitions between L2 and L1 loss:
            - For absolute prediction errors within D: quadratic loss (0.5 * (error**2))
            - For absolute errors exceeding D: linear loss (D * |error| - 0.5 * D)
            Formula: loss = 0.5 * (error**2) if |error| <= D else D * (|error| - 0.5 * D).
        f_use_norm : bool
            If true, use L2 norm of force vectors for loss calculation when loss_func='mae' or use_huber is True.
            Instead of computing loss on force components, computes loss on ||F_pred - F_label||_2.
            This treats the force vector as a whole rather than three independent components.
        huber_delta : float | list[float]
            The threshold delta (D) used for Huber loss, controlling transition between
            L2 and L1 loss. It can be either one float shared by all terms or a list of
            three values ordered as [energy, force, virial].
        **kwargs
            Other keyword arguments.
        """
        super().__init__()

        # Validate loss_func
        valid_loss_funcs = ["mse", "mae"]
        if loss_func not in valid_loss_funcs:
            raise ValueError(
                f"Invalid loss_func '{loss_func}'. Must be one of {valid_loss_funcs}."
            )

        self.loss_func = loss_func
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
        self.use_default_pf = use_default_pf
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.numb_generalized_coord = numb_generalized_coord
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.inference = inference
        self.use_huber = use_huber
        self.f_use_norm = f_use_norm
        if self.f_use_norm and not (self.use_huber or self.loss_func == "mae"):
            raise RuntimeError(
                "f_use_norm can only be True when use_huber or loss_func='mae'."
            )
        self.huber_delta = huber_delta
        (
            self._huber_delta_energy,
            self._huber_delta_force,
            self._huber_delta_virial,
        ) = resolve_huber_deltas(huber_delta)
        if self.use_huber and (
            self.has_pf or self.has_gf or self.relative_f is not None
        ):
            raise RuntimeError(
                "Huber loss is not implemented for force with atom_pref, generalized force and relative force. "
            )

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
            if self.loss_func == "mse":
                l2_ener_loss = torch.mean(torch.square(energy_pred - energy_label))
                if not self.inference:
                    more_loss["l2_ener_loss"] = self.display_if_exist(
                        l2_ener_loss.detach(), find_energy
                    )
                if not self.use_huber:
                    loss += atom_norm * (pref_e * l2_ener_loss)
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_norm * energy_pred,
                        atom_norm * energy_label,
                        delta=self._huber_delta_energy,
                    )
                    loss += pref_e * l_huber_loss
                rmse_e = l2_ener_loss.sqrt() * atom_norm
                more_loss["rmse_e"] = self.display_if_exist(
                    rmse_e.detach(), find_energy
                )
                # more_loss['log_keys'].append('rmse_e')
            elif self.loss_func == "mae":
                l1_ener_loss = F.l1_loss(
                    energy_pred.reshape(-1),
                    energy_label.reshape(-1),
                    reduction="mean",
                )
                loss += atom_norm * (pref_e * l1_ener_loss)
                more_loss["mae_e"] = self.display_if_exist(
                    l1_ener_loss.detach() * atom_norm,
                    find_energy,
                )
                # more_loss['log_keys'].append('rmse_e')
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for energy loss."
                )
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
                if self.loss_func == "mse":
                    l2_force_loss = torch.mean(torch.square(diff_f))
                    if not self.inference:
                        more_loss["l2_force_loss"] = self.display_if_exist(
                            l2_force_loss.detach(), find_force
                        )
                    if not self.use_huber:
                        loss += (pref_f * l2_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                    else:
                        if not self.f_use_norm:
                            l_huber_loss = custom_huber_loss(
                                force_pred.reshape(-1),
                                force_label.reshape(-1),
                                delta=self._huber_delta_force,
                            )
                        else:
                            force_diff_norm = torch.linalg.vector_norm(
                                (force_label - force_pred).reshape(-1, 3),
                                ord=2,
                                dim=1,
                                keepdim=True,
                            )
                            l_huber_loss = custom_huber_loss(
                                force_diff_norm,
                                torch.zeros_like(force_diff_norm),
                                delta=self._huber_delta_force,
                            )
                        loss += pref_f * l_huber_loss
                    rmse_f = l2_force_loss.sqrt()
                    more_loss["rmse_f"] = self.display_if_exist(
                        rmse_f.detach(), find_force
                    )
                elif self.loss_func == "mae":
                    if not self.f_use_norm:
                        l1_force_loss = F.l1_loss(
                            force_label.reshape(-1),
                            force_pred.reshape(-1),
                            reduction="mean",
                        )
                    else:
                        l1_force_loss = torch.linalg.vector_norm(
                            (force_label - force_pred).reshape(-1, 3),
                            ord=2,
                            dim=1,
                            keepdim=True,
                        ).mean()
                    more_loss["mae_f"] = self.display_if_exist(
                        l1_force_loss.detach(), find_force
                    )
                    loss += (pref_f * l1_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                else:
                    raise NotImplementedError(
                        f"Loss type {self.loss_func} is not implemented for force loss."
                    )
                if mae:
                    mae_f = torch.mean(torch.abs(diff_f))
                    more_loss["mae_f"] = self.display_if_exist(
                        mae_f.detach(), find_force
                    )

            if self.has_pf and "atom_pref" in label:
                atom_pref = label["atom_pref"]
                find_atom_pref = (
                    label.get("find_atom_pref", 0.0) if not self.use_default_pf else 1.0
                )
                pref_pf = pref_pf * find_atom_pref
                atom_pref_reshape = atom_pref.reshape(-1)

                if self.loss_func == "mse":
                    l2_pref_force_loss = (
                        torch.square(diff_f) * atom_pref_reshape
                    ).mean()
                    if not self.inference:
                        more_loss["l2_pref_force_loss"] = self.display_if_exist(
                            l2_pref_force_loss.detach(), find_atom_pref
                        )
                    loss += (pref_pf * l2_pref_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                    rmse_pf = l2_pref_force_loss.sqrt()
                    more_loss["rmse_pf"] = self.display_if_exist(
                        rmse_pf.detach(), find_atom_pref
                    )
                elif self.loss_func == "mae":
                    l1_pref_force_loss = (torch.abs(diff_f) * atom_pref_reshape).mean()
                    loss += (pref_pf * l1_pref_force_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                    more_loss["mae_pf"] = self.display_if_exist(
                        l1_pref_force_loss.detach(), find_atom_pref
                    )
                else:
                    raise NotImplementedError(
                        f"Loss type {self.loss_func} is not implemented for atom prefactor force loss."
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
            if self.loss_func == "mse":
                l2_virial_loss = torch.mean(torch.square(diff_v))
                if not self.inference:
                    more_loss["l2_virial_loss"] = self.display_if_exist(
                        l2_virial_loss.detach(), find_virial
                    )
                if not self.use_huber:
                    loss += atom_norm * (pref_v * l2_virial_loss)
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_norm * model_pred["virial"].reshape(-1),
                        atom_norm * label["virial"].reshape(-1),
                        delta=self._huber_delta_virial,
                    )
                    loss += pref_v * l_huber_loss
                rmse_v = l2_virial_loss.sqrt() * atom_norm
                more_loss["rmse_v"] = self.display_if_exist(
                    rmse_v.detach(), find_virial
                )
            elif self.loss_func == "mae":
                l1_virial_loss = F.l1_loss(
                    label["virial"].reshape(-1),
                    model_pred["virial"].reshape(-1),
                    reduction="mean",
                )
                loss += atom_norm * (pref_v * l1_virial_loss)
                more_loss["mae_v"] = self.display_if_exist(
                    l1_virial_loss.detach() * atom_norm,
                    find_virial,
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for virial loss."
                )
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

            if self.loss_func == "mse":
                l2_atom_ener_loss = torch.square(
                    atom_ener_label_reshape - atom_ener_reshape
                ).mean()
                if not self.inference:
                    more_loss["l2_atom_ener_loss"] = self.display_if_exist(
                        l2_atom_ener_loss.detach(), find_atom_ener
                    )
                if not self.use_huber:
                    loss += (pref_ae * l2_atom_ener_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                else:
                    l_huber_loss = custom_huber_loss(
                        atom_ener_reshape,
                        atom_ener_label_reshape,
                        delta=self._huber_delta_energy,
                    )
                    loss += pref_ae * l_huber_loss
                rmse_ae = l2_atom_ener_loss.sqrt()
                more_loss["rmse_ae"] = self.display_if_exist(
                    rmse_ae.detach(), find_atom_ener
                )
            elif self.loss_func == "mae":
                l1_atom_ener_loss = F.l1_loss(
                    atom_ener_reshape,
                    atom_ener_label_reshape,
                    reduction="mean",
                )
                loss += (pref_ae * l1_atom_ener_loss).to(GLOBAL_PT_FLOAT_PRECISION)
                more_loss["mae_ae"] = self.display_if_exist(
                    l1_atom_ener_loss.detach(), find_atom_ener
                )
            else:
                raise NotImplementedError(
                    f"Loss type {self.loss_func} is not implemented for atomic energy loss."
                )

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
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
        if self.has_f or self.has_pf or self.relative_f is not None or self.has_gf:
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
                    default=1.0,
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

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        return {
            "@class": "EnergyLoss",
            "@version": 3,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_f": self.start_pref_f,
            "limit_pref_f": self.limit_pref_f,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "start_pref_pf": self.start_pref_pf,
            "limit_pref_pf": self.limit_pref_pf,
            "relative_f": self.relative_f,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "start_pref_gf": self.start_pref_gf,
            "limit_pref_gf": self.limit_pref_gf,
            "numb_generalized_coord": self.numb_generalized_coord,
            "use_huber": self.use_huber,
            "huber_delta": self.huber_delta,
            "loss_func": self.loss_func,
            "f_use_norm": self.f_use_norm,
            "use_default_pf": self.use_default_pf,
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
        Loss
            The deserialized loss module
        """
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 3, 1)
        data.pop("@class")
        return cls(**data)


class EnergyHessianStdLoss(EnergyStdLoss):
    def __init__(
        self,
        start_pref_h: float = 0.0,
        limit_pref_h: float = 0.0,
        **kwargs: Any,
    ) -> None:
        r"""Enable the layer to compute loss on hessian.

        Parameters
        ----------
        start_pref_h : float
            The prefactor of hessian loss at the start of the training.
        limit_pref_h : float
            The prefactor of hessian loss at the end of the training.
        **kwargs
            Other keyword arguments.
        """
        super().__init__(**kwargs)
        self.has_h = (start_pref_h != 0.0 and limit_pref_h != 0.0) or self.inference

        self.start_pref_h = start_pref_h
        self.limit_pref_h = limit_pref_h

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        model_pred, loss, more_loss = super().forward(
            input_dict, model, label, natoms, learning_rate, mae=mae
        )
        coef = learning_rate / self.starter_learning_rate
        pref_h = self.limit_pref_h + (self.start_pref_h - self.limit_pref_h) * coef

        if self.has_h and "hessian" in model_pred and "hessian" in label:
            find_hessian = label.get("find_hessian", 0.0)
            pref_h = pref_h * find_hessian
            diff_h = label["hessian"].reshape(
                -1,
            ) - model_pred["hessian"].reshape(
                -1,
            )
            l2_hessian_loss = torch.mean(torch.square(diff_h))
            if not self.inference:
                more_loss["l2_hessian_loss"] = self.display_if_exist(
                    l2_hessian_loss.detach(), find_hessian
                )
            loss += pref_h * l2_hessian_loss
            rmse_h = l2_hessian_loss.sqrt()
            more_loss["rmse_h"] = self.display_if_exist(rmse_h.detach(), find_hessian)
            if mae:
                mae_h = torch.mean(torch.abs(diff_h))
                more_loss["mae_h"] = self.display_if_exist(mae_h.detach(), find_hessian)

        if not self.inference:
            more_loss["rmse"] = torch.sqrt(loss.detach())
        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Add hessian label requirement needed for this loss calculation."""
        label_requirement = super().label_requirement
        if self.has_h:
            label_requirement.append(
                DataRequirementItem(
                    "hessian",
                    ndof=1,  # 9=3*3 --> 3N*3N=ndof*natoms*natoms
                    atomic=True,
                    must=False,
                    high_prec=False,
                )
            )
        return label_requirement
