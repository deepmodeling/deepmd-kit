# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


def custom_huber_loss(predictions, targets, delta=1.0):
    xp = array_api_compat.array_namespace(predictions, targets)
    error = targets - predictions
    abs_error = xp.abs(error)
    quadratic_loss = 0.5 * error**2
    linear_loss = delta * (abs_error - 0.5 * delta)
    loss = xp.where(abs_error <= delta, quadratic_loss, linear_loss)
    return xp.mean(loss)


class EnergyLoss(Loss):
    def __init__(
        self,
        starter_learning_rate: float,
        start_pref_e: float = 0.02,
        limit_pref_e: float = 1.00,
        start_pref_f: float = 1000,
        limit_pref_f: float = 1.00,
        start_pref_v: float = 0.0,
        limit_pref_v: float = 0.0,
        start_pref_ae: float = 0.0,
        limit_pref_ae: float = 0.0,
        start_pref_pf: float = 0.0,
        limit_pref_pf: float = 0.0,
        relative_f: Optional[float] = None,
        enable_atom_ener_coeff: bool = False,
        start_pref_gf: float = 0.0,
        limit_pref_gf: float = 0.0,
        numb_generalized_coord: int = 0,
        use_huber=False,
        huber_delta=0.01,
        **kwargs,
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
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
        self.relative_f = relative_f
        self.enable_atom_ener_coeff = enable_atom_ener_coeff
        self.start_pref_gf = start_pref_gf
        self.limit_pref_gf = limit_pref_gf
        self.numb_generalized_coord = numb_generalized_coord
        self.has_e = self.start_pref_e != 0.0 or self.limit_pref_e != 0.0
        self.has_f = self.start_pref_f != 0.0 or self.limit_pref_f != 0.0
        self.has_v = self.start_pref_v != 0.0 or self.limit_pref_v != 0.0
        self.has_ae = self.start_pref_ae != 0.0 or self.limit_pref_ae != 0.0
        self.has_pf = self.start_pref_pf != 0.0 or self.limit_pref_pf != 0.0
        self.has_gf = self.start_pref_gf != 0.0 or self.limit_pref_gf != 0.0
        if self.has_gf and self.numb_generalized_coord < 1:
            raise RuntimeError(
                "When generalized force loss is used, the dimension of generalized coordinates should be larger than 0"
            )
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        if self.use_huber and (
            self.has_pf or self.has_gf or self.relative_f is not None
        ):
            raise RuntimeError(
                "Huber loss is not implemented for force with atom_pref, generalized force and relative force. "
            )

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, np.ndarray],
        label_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Calculate loss from model results and labeled results."""
        energy = model_dict["energy_redu"]
        force = model_dict["energy_derv_r"]
        virial = model_dict["energy_derv_c_redu"]
        atom_ener = model_dict["energy"]
        energy_hat = label_dict["energy"]
        force_hat = label_dict["force"]
        virial_hat = label_dict["virial"]
        atom_ener_hat = label_dict["atom_ener"]
        atom_pref = label_dict["atom_pref"]
        find_energy = label_dict["find_energy"]
        find_force = label_dict["find_force"]
        find_virial = label_dict["find_virial"]
        find_atom_ener = label_dict["find_atom_ener"]
        find_atom_pref = label_dict["find_atom_pref"]
        xp = array_api_compat.array_namespace(
            energy,
            force,
            virial,
            atom_ener,
            energy_hat,
            force_hat,
            virial_hat,
            atom_ener_hat,
            atom_pref,
        )

        if self.enable_atom_ener_coeff:
            # when ener_coeff (\nu) is defined, the energy is defined as
            # E = \sum_i \nu_i E_i
            # instead of the sum of atomic energies.
            #
            # A case is that we want to train reaction energy
            # A + B -> C + D
            # E = - E(A) - E(B) + E(C) + E(D)
            # A, B, C, D could be put far away from each other
            atom_ener_coeff = label_dict["atom_ener_coeff"]
            atom_ener_coeff = xp.reshape(atom_ener_coeff, xp.shape(atom_ener))
            energy = xp.sum(atom_ener_coeff * atom_ener, 1)
        if self.has_f or self.has_pf or self.relative_f or self.has_gf:
            force_reshape = xp.reshape(force, [-1])
            force_hat_reshape = xp.reshape(force_hat, [-1])
            diff_f = force_hat_reshape - force_reshape
        else:
            diff_f = None

        if self.relative_f is not None:
            force_hat_3 = xp.reshape(force_hat, [-1, 3])
            norm_f = xp.reshape(xp.norm(force_hat_3, axis=1), [-1, 1]) + self.relative_f
            diff_f_3 = xp.reshape(diff_f, [-1, 3])
            diff_f_3 = diff_f_3 / norm_f
            diff_f = xp.reshape(diff_f_3, [-1])

        atom_norm = 1.0 / natoms
        atom_norm_ener = 1.0 / natoms
        lr_ratio = learning_rate / self.starter_learning_rate
        pref_e = find_energy * (
            self.limit_pref_e + (self.start_pref_e - self.limit_pref_e) * lr_ratio
        )
        pref_f = find_force * (
            self.limit_pref_f + (self.start_pref_f - self.limit_pref_f) * lr_ratio
        )
        pref_v = find_virial * (
            self.limit_pref_v + (self.start_pref_v - self.limit_pref_v) * lr_ratio
        )
        pref_ae = find_atom_ener * (
            self.limit_pref_ae + (self.start_pref_ae - self.limit_pref_ae) * lr_ratio
        )
        pref_pf = find_atom_pref * (
            self.limit_pref_pf + (self.start_pref_pf - self.limit_pref_pf) * lr_ratio
        )

        loss = 0
        more_loss = {}
        if self.has_e:
            l2_ener_loss = xp.mean(xp.square(energy - energy_hat))
            if not self.use_huber:
                loss += atom_norm_ener * (pref_e * l2_ener_loss)
            else:
                l_huber_loss = custom_huber_loss(
                    atom_norm_ener * energy,
                    atom_norm_ener * energy_hat,
                    delta=self.huber_delta,
                )
                loss += pref_e * l_huber_loss
            more_loss["rmse_e"] = self.display_if_exist(l2_ener_loss, find_energy)
        if self.has_f:
            l2_force_loss = xp.mean(xp.square(diff_f))
            if not self.use_huber:
                loss += pref_f * l2_force_loss
            else:
                l_huber_loss = custom_huber_loss(
                    xp.reshape(force, [-1]),
                    xp.reshape(force_hat, [-1]),
                    delta=self.huber_delta,
                )
                loss += pref_f * l_huber_loss
            more_loss["rmse_f"] = self.display_if_exist(l2_force_loss, find_force)
        if self.has_v:
            virial_reshape = xp.reshape(virial, [-1])
            virial_hat_reshape = xp.reshape(virial_hat, [-1])
            l2_virial_loss = xp.mean(
                xp.square(virial_hat_reshape - virial_reshape),
            )
            if not self.use_huber:
                loss += atom_norm * (pref_v * l2_virial_loss)
            else:
                l_huber_loss = custom_huber_loss(
                    atom_norm * virial_reshape,
                    atom_norm * virial_hat_reshape,
                    delta=self.huber_delta,
                )
                loss += pref_v * l_huber_loss
            more_loss["rmse_v"] = self.display_if_exist(l2_virial_loss, find_virial)
        if self.has_ae:
            atom_ener_reshape = xp.reshape(atom_ener, [-1])
            atom_ener_hat_reshape = xp.reshape(atom_ener_hat, [-1])
            l2_atom_ener_loss = xp.mean(
                xp.square(atom_ener_hat_reshape - atom_ener_reshape),
            )
            if not self.use_huber:
                loss += pref_ae * l2_atom_ener_loss
            else:
                l_huber_loss = custom_huber_loss(
                    atom_ener_reshape,
                    atom_ener_hat_reshape,
                    delta=self.huber_delta,
                )
                loss += pref_ae * l_huber_loss
            more_loss["rmse_ae"] = self.display_if_exist(
                l2_atom_ener_loss, find_atom_ener
            )
        if self.has_pf:
            atom_pref_reshape = xp.reshape(atom_pref, [-1])
            l2_pref_force_loss = xp.mean(
                xp.multiply(xp.square(diff_f), atom_pref_reshape),
            )
            loss += pref_pf * l2_pref_force_loss
            more_loss["rmse_pf"] = self.display_if_exist(
                l2_pref_force_loss, find_atom_pref
            )
        if self.has_gf:
            find_drdq = label_dict["find_drdq"]
            drdq = label_dict["drdq"]
            force_reshape_nframes = xp.reshape(force, [-1, natoms[0] * 3])
            force_hat_reshape_nframes = xp.reshape(force_hat, [-1, natoms[0] * 3])
            drdq_reshape = xp.reshape(
                drdq, [-1, natoms[0] * 3, self.numb_generalized_coord]
            )
            gen_force_hat = xp.einsum(
                "bij,bi->bj", drdq_reshape, force_hat_reshape_nframes
            )
            gen_force = xp.einsum("bij,bi->bj", drdq_reshape, force_reshape_nframes)
            diff_gen_force = gen_force_hat - gen_force
            l2_gen_force_loss = xp.mean(xp.square(diff_gen_force))
            pref_gf = find_drdq * (
                self.limit_pref_gf
                + (self.start_pref_gf - self.limit_pref_gf) * lr_ratio
            )
            loss += pref_gf * l2_gen_force_loss
            more_loss["rmse_gf"] = self.display_if_exist(l2_gen_force_loss, find_drdq)

        self.l2_l = loss
        more_loss["rmse"] = xp.sqrt(loss)
        self.l2_more = more_loss
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        label_requirement = []
        label_requirement.append(
            DataRequirementItem(
                "energy",
                ndof=1,
                atomic=False,
                must=False,
                high_prec=True,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "force",
                ndof=3,
                atomic=True,
                must=False,
                high_prec=False,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "virial",
                ndof=9,
                atomic=False,
                must=False,
                high_prec=False,
            )
        )
        label_requirement.append(
            DataRequirementItem(
                "atom_ener",
                ndof=1,
                atomic=True,
                must=False,
                high_prec=False,
            )
        )
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

    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """
        return {
            "@class": "EnergyLoss",
            "@version": 2,
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
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Loss":
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
        check_version_compatibility(data.pop("@version"), 2, 1)
        data.pop("@class")
        return cls(**data)
