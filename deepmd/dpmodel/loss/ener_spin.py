# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class EnergySpinLoss(Loss):
    r"""Loss on energy, real force, magnetic force and virial for spin models.

    Parameters
    ----------
    starter_learning_rate : float
        The learning rate at the start of the training.
    start_pref_e : float
        The prefactor of energy loss at the start of the training.
    limit_pref_e : float
        The prefactor of energy loss at the end of the training.
    start_pref_fr : float
        The prefactor of real force loss at the start of the training.
    limit_pref_fr : float
        The prefactor of real force loss at the end of the training.
    start_pref_fm : float
        The prefactor of magnetic force loss at the start of the training.
    limit_pref_fm : float
        The prefactor of magnetic force loss at the end of the training.
    start_pref_v : float
        The prefactor of virial loss at the start of the training.
    limit_pref_v : float
        The prefactor of virial loss at the end of the training.
    start_pref_ae : float
        The prefactor of atomic energy loss at the start of the training.
    limit_pref_ae : float
        The prefactor of atomic energy loss at the end of the training.
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \sum_i c_i E_i
    loss_func : str
        Loss function type: 'mse' or 'mae'.
    intensive : bool
        If true (default), energy and virial losses are computed as intensive quantities,
        normalized by the square of the number of atoms (1/N^2). This ensures the loss
        value is independent of system size and consistent with per-atom RMSE reporting.
        If false, uses the legacy normalization (1/N), which may cause the loss to scale
        with system size. Set to false for backward compatibility with models trained
        using deepmd-kit <= 3.0.1.
    **kwargs
        Other keyword arguments.
    """

    def __init__(
        self,
        starter_learning_rate: float = 1.0,
        start_pref_e: float = 0.0,
        limit_pref_e: float = 0.0,
        start_pref_fr: float = 0.0,
        limit_pref_fr: float = 0.0,
        start_pref_fm: float = 0.0,
        limit_pref_fm: float = 0.0,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        enable_atom_ener_coeff: bool = False,
        loss_func: str = "mse",
        intensive: bool = True,
        **kwargs: Any,
    ) -> None:
        valid_loss_funcs = ["mse", "mae"]
        if loss_func not in valid_loss_funcs:
            raise ValueError(
                f"Invalid loss_func '{loss_func}'. Must be one of {valid_loss_funcs}."
            )
        self.loss_func = loss_func
        self.starter_learning_rate = starter_learning_rate
        self.start_pref_e = start_pref_e
        self.limit_pref_e = limit_pref_e
        self.start_pref_fr = start_pref_fr
        self.limit_pref_fr = limit_pref_fr
        self.start_pref_fm = start_pref_fm
        self.limit_pref_fm = limit_pref_fm
        self.start_pref_v = start_pref_v
        self.limit_pref_v = limit_pref_v
        self.start_pref_ae = start_pref_ae
        self.limit_pref_ae = limit_pref_ae
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.intensive = intensive
        self.has_e = self.start_pref_e != 0.0 or self.limit_pref_e != 0.0
        self.has_fr = self.start_pref_fr != 0.0 or self.limit_pref_fr != 0.0
        self.has_fm = self.start_pref_fm != 0.0 or self.limit_pref_fm != 0.0
        self.has_v = self.start_pref_v != 0.0 or self.limit_pref_v != 0.0
        self.has_ae = self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate loss from model results and labeled results."""
        energy = model_dict["energy"]
        xp = array_api_compat.array_namespace(energy)

        coef = learning_rate / self.starter_learning_rate
        pref_e = self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * coef
        pref_fr = self.limit_pref_fr + (self.start_pref_fr - self.limit_pref_fr) * coef
        pref_fm = self.limit_pref_fm + (self.start_pref_fm - self.limit_pref_fm) * coef
        pref_v = self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * coef
        pref_ae = self.limit_pref_ae + (self.start_pref_ae - self.limit_pref_ae) * coef

        loss = 0
        more_loss = {}
        atom_norm = 1.0 / natoms
        # Normalization exponent controls loss scaling with system size:
        # - norm_exp=2 (intensive=True): loss uses 1/N² scaling, making it independent of system size
        # - norm_exp=1 (intensive=False, legacy): loss uses 1/N scaling, which varies with system size
        norm_exp = 2 if self.intensive else 1

        if self.has_e:
            energy_pred = model_dict["energy"]
            energy_label = label_dict["energy"]
            find_energy = label_dict.get("find_energy", 0.0)
            pref_e = pref_e * find_energy
            if self.enable_atom_ener_coeff and "atom_energy" in model_dict:
                atom_ener_pred = model_dict["atom_energy"]
                atom_ener_coeff = label_dict["atom_ener_coeff"]
                atom_ener_coeff = xp.reshape(atom_ener_coeff, atom_ener_pred.shape)
                energy_pred = xp.sum(atom_ener_coeff * atom_ener_pred, axis=1)
            if self.loss_func == "mse":
                l2_ener_loss = xp.mean(xp.square(energy_pred - energy_label))
                loss += atom_norm**norm_exp * (pref_e * l2_ener_loss)
                more_loss["rmse_e"] = self.display_if_exist(
                    xp.sqrt(l2_ener_loss) * atom_norm, find_energy
                )
            elif self.loss_func == "mae":
                abs_diff_e = xp.abs(energy_pred - energy_label)
                l1_ener_loss = xp.sum(abs_diff_e)
                loss += pref_e * l1_ener_loss
                more_loss["mae_e"] = self.display_if_exist(
                    xp.mean(abs_diff_e), find_energy
                )
            if mae:
                mae_e = xp.mean(xp.abs(energy_pred - energy_label)) * atom_norm
                more_loss["mae_e"] = self.display_if_exist(mae_e, find_energy)
                mae_e_all = xp.mean(xp.abs(energy_pred - energy_label))
                more_loss["mae_e_all"] = self.display_if_exist(mae_e_all, find_energy)

        if self.has_fr:
            find_force = label_dict.get("find_force", 0.0)
            pref_fr = pref_fr * find_force
            force_pred = model_dict["force"]
            force_label = label_dict["force"]
            if self.loss_func == "mse":
                diff_fr = force_label - force_pred
                l2_force_real_loss = xp.mean(xp.square(diff_fr))
                loss += pref_fr * l2_force_real_loss
                more_loss["rmse_fr"] = self.display_if_exist(
                    xp.sqrt(l2_force_real_loss), find_force
                )
                if mae:
                    mae_fr = xp.mean(xp.abs(force_label - force_pred))
                    more_loss["mae_fr"] = self.display_if_exist(mae_fr, find_force)
            elif self.loss_func == "mae":
                abs_diff_fr = xp.abs(force_label - force_pred)
                per_atom_fr = xp.sum(abs_diff_fr, axis=-1)  # [nf, na]
                per_frame_fr = xp.mean(per_atom_fr, axis=-1)  # [nf]
                l1_force_real_loss = xp.sum(per_frame_fr)  # scalar
                loss += pref_fr * l1_force_real_loss
                more_loss["mae_fr"] = self.display_if_exist(
                    xp.mean(abs_diff_fr), find_force
                )

        if self.has_fm:
            find_force_mag = label_dict.get("find_force_mag", 0.0)
            pref_fm = pref_fm * find_force_mag
            force_mag_pred = model_dict["force_mag"]
            force_mag_label = label_dict["force_mag"]
            mask_mag = model_dict["mask_mag"]
            # mask_mag: [nframes, natoms, 1], bool -> use mask multiplication
            mask_float = xp.astype(mask_mag, force_mag_pred.dtype)
            # zero out non-magnetic atoms
            diff_fm = (force_mag_label - force_mag_pred) * mask_float
            n_valid = xp.sum(mask_float)
            if self.loss_func == "mse":
                l2_force_mag_loss = xp.sum(xp.square(diff_fm)) / (n_valid * 3)
                loss += pref_fm * l2_force_mag_loss
                more_loss["rmse_fm"] = self.display_if_exist(
                    xp.sqrt(l2_force_mag_loss), find_force_mag
                )
                if mae:
                    mae_fm = xp.sum(xp.abs(diff_fm)) / (n_valid * 3)
                    more_loss["mae_fm"] = self.display_if_exist(mae_fm, find_force_mag)
            elif self.loss_func == "mae":
                abs_diff_fm = xp.abs(diff_fm)  # [nf, na, 3], zeros for non-magnetic
                per_atom_fm = xp.sum(abs_diff_fm, axis=-1)  # [nf, na]
                mask_2d = mask_float[:, :, 0]  # [nf, na]
                per_frame_sum_fm = xp.sum(per_atom_fm, axis=-1)  # [nf]
                per_frame_count_fm = xp.sum(mask_2d, axis=-1)  # [nf]
                l1_force_mag_loss = xp.sum(
                    per_frame_sum_fm / per_frame_count_fm
                )  # scalar
                loss += pref_fm * l1_force_mag_loss
                more_loss["mae_fm"] = self.display_if_exist(
                    xp.sum(abs_diff_fm) / (n_valid * 3), find_force_mag
                )

        if self.has_ae:
            find_atom_ener = label_dict.get("find_atom_ener", 0.0)
            pref_ae = pref_ae * find_atom_ener
            atom_ener = model_dict["atom_energy"]
            atom_ener_label = label_dict["atom_ener"]
            atom_ener_reshape = xp.reshape(atom_ener, (-1,))
            atom_ener_label_reshape = xp.reshape(atom_ener_label, (-1,))
            if self.loss_func == "mse":
                l2_atom_ener_loss = xp.mean(
                    xp.square(atom_ener_label_reshape - atom_ener_reshape)
                )
                loss += pref_ae * l2_atom_ener_loss
                more_loss["rmse_ae"] = self.display_if_exist(
                    xp.sqrt(l2_atom_ener_loss), find_atom_ener
                )
            elif self.loss_func == "mae":
                l1_atom_ener_loss = xp.mean(
                    xp.abs(atom_ener_label_reshape - atom_ener_reshape)
                )
                loss += pref_ae * l1_atom_ener_loss
                more_loss["mae_ae"] = self.display_if_exist(
                    l1_atom_ener_loss, find_atom_ener
                )

        if self.has_v:
            find_virial = label_dict.get("find_virial", 0.0)
            pref_v = pref_v * find_virial
            virial_pred = xp.reshape(model_dict["virial"], (-1, 9))
            virial_label = label_dict["virial"]
            diff_v = virial_label - virial_pred
            if self.loss_func == "mse":
                l2_virial_loss = xp.mean(xp.square(diff_v))
                loss += atom_norm**norm_exp * (pref_v * l2_virial_loss)
                more_loss["rmse_v"] = self.display_if_exist(
                    xp.sqrt(l2_virial_loss) * atom_norm, find_virial
                )
                if mae:
                    mae_v = xp.mean(xp.abs(diff_v)) * atom_norm
                    more_loss["mae_v"] = self.display_if_exist(mae_v, find_virial)
            elif self.loss_func == "mae":
                l1_virial_loss = xp.mean(xp.abs(diff_v))
                loss += atom_norm * (pref_v * l1_virial_loss)
                more_loss["mae_v"] = self.display_if_exist(
                    l1_virial_loss * atom_norm, find_virial
                )

        more_loss["rmse"] = xp.sqrt(loss)
        return loss, more_loss

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
        """Serialize the loss module."""
        return {
            "@class": "EnergySpinLoss",
            "@version": 2,
            "starter_learning_rate": self.starter_learning_rate,
            "start_pref_e": self.start_pref_e,
            "limit_pref_e": self.limit_pref_e,
            "start_pref_fr": self.start_pref_fr,
            "limit_pref_fr": self.limit_pref_fr,
            "start_pref_fm": self.start_pref_fm,
            "limit_pref_fm": self.limit_pref_fm,
            "start_pref_v": self.start_pref_v,
            "limit_pref_v": self.limit_pref_v,
            "start_pref_ae": self.start_pref_ae,
            "limit_pref_ae": self.limit_pref_ae,
            "enable_atom_ener_coeff": self.enable_atom_ener_coeff,
            "loss_func": self.loss_func,
            "intensive": self.intensive,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EnergySpinLoss":
        """Deserialize the loss module."""
        data = data.copy()
        version = data.pop("@version")
        check_version_compatibility(version, 2, 1)
        data.pop("@class")
        # Backward compatibility: version 1 used legacy normalization
        if version < 2:
            data.setdefault("intensive", False)
        return cls(**data)
