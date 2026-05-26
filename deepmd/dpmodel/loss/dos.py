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


class DOSLoss(Loss):
    r"""Loss on DOS (density of states) for both local and global predictions.

    Parameters
    ----------
    starter_learning_rate : float
        The learning rate at the start of the training.
    numb_dos : int
        The number of DOS components.
    start_pref_dos : float
        The prefactor of global DOS loss at the start of the training.
    limit_pref_dos : float
        The prefactor of global DOS loss at the end of the training.
    start_pref_cdf : float
        The prefactor of global CDF loss at the start of the training.
    limit_pref_cdf : float
        The prefactor of global CDF loss at the end of the training.
    start_pref_ados : float
        The prefactor of atomic DOS loss at the start of the training.
    limit_pref_ados : float
        The prefactor of atomic DOS loss at the end of the training.
    start_pref_acdf : float
        The prefactor of atomic CDF loss at the start of the training.
    limit_pref_acdf : float
        The prefactor of atomic CDF loss at the end of the training.
    **kwargs
        Other keyword arguments.
    """

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
        **kwargs: Any,
    ) -> None:
        self.starter_learning_rate = starter_learning_rate
        self.numb_dos = numb_dos
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

        self.has_dos = start_pref_dos != 0.0 or limit_pref_dos != 0.0
        self.has_cdf = start_pref_cdf != 0.0 or limit_pref_cdf != 0.0
        self.has_ados = start_pref_ados != 0.0 or limit_pref_ados != 0.0
        self.has_acdf = start_pref_acdf != 0.0 or limit_pref_acdf != 0.0

        assert self.has_dos or self.has_cdf or self.has_ados or self.has_acdf, (
            "Can not assign zero weight to all pref terms"
        )

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate loss from model results and labeled results."""
        # Get array namespace from any available tensor
        first_key = next(iter(model_dict))
        xp = array_api_compat.array_namespace(model_dict[first_key])

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

        loss = 0
        more_loss = {}

        if self.has_ados and "atom_dos" in model_dict and "atom_dos" in label_dict:
            find_local = label_dict.get("find_atom_dos", 0.0)
            pref_ados = pref_ados * find_local
            local_pred = xp.reshape(model_dict["atom_dos"], (-1, natoms, self.numb_dos))
            local_label = xp.reshape(
                label_dict["atom_dos"], (-1, natoms, self.numb_dos)
            )
            diff = xp.reshape(local_pred - local_label, (-1, self.numb_dos))
            if "mask" in model_dict:
                mask = xp.reshape(model_dict["mask"], (-1,))
                mask_float = xp.astype(mask, diff.dtype)
                diff = diff * mask_float[:, None]
                n_valid = xp.sum(mask_float)
                l2_local_loss_dos = xp.sum(xp.square(diff)) / (n_valid * self.numb_dos)
            else:
                l2_local_loss_dos = xp.mean(xp.square(diff))
            loss += pref_ados * l2_local_loss_dos
            more_loss["rmse_local_dos"] = self.display_if_exist(
                xp.sqrt(l2_local_loss_dos), find_local
            )

        if self.has_acdf and "atom_dos" in model_dict and "atom_dos" in label_dict:
            find_local = label_dict.get("find_atom_dos", 0.0)
            pref_acdf = pref_acdf * find_local
            local_pred_cdf = xp.cumulative_sum(
                xp.reshape(model_dict["atom_dos"], (-1, natoms, self.numb_dos)),
                axis=-1,
            )
            local_label_cdf = xp.cumulative_sum(
                xp.reshape(label_dict["atom_dos"], (-1, natoms, self.numb_dos)),
                axis=-1,
            )
            diff = xp.reshape(local_pred_cdf - local_label_cdf, (-1, self.numb_dos))
            if "mask" in model_dict:
                mask = xp.reshape(model_dict["mask"], (-1,))
                mask_float = xp.astype(mask, diff.dtype)
                diff = diff * mask_float[:, None]
                n_valid = xp.sum(mask_float)
                l2_local_loss_cdf = xp.sum(xp.square(diff)) / (n_valid * self.numb_dos)
            else:
                l2_local_loss_cdf = xp.mean(xp.square(diff))
            loss += pref_acdf * l2_local_loss_cdf
            more_loss["rmse_local_cdf"] = self.display_if_exist(
                xp.sqrt(l2_local_loss_cdf), find_local
            )

        if self.has_dos and "dos" in model_dict and "dos" in label_dict:
            find_global = label_dict.get("find_dos", 0.0)
            pref_dos = pref_dos * find_global
            global_pred = xp.reshape(model_dict["dos"], (-1, self.numb_dos))
            global_label = xp.reshape(label_dict["dos"], (-1, self.numb_dos))
            diff = global_pred - global_label
            if "mask" in model_dict:
                atom_num = xp.sum(model_dict["mask"], axis=-1, keepdims=True)
                l2_global_loss_dos = xp.mean(
                    xp.sum(xp.square(diff) * atom_num, axis=0) / xp.sum(atom_num)
                )
                atom_num = xp.mean(xp.astype(atom_num, diff.dtype))
            else:
                atom_num = natoms
                l2_global_loss_dos = xp.mean(xp.square(diff))
            loss += pref_dos * l2_global_loss_dos
            more_loss["rmse_global_dos"] = self.display_if_exist(
                xp.sqrt(l2_global_loss_dos) / atom_num, find_global
            )

        if self.has_cdf and "dos" in model_dict and "dos" in label_dict:
            find_global = label_dict.get("find_dos", 0.0)
            pref_cdf = pref_cdf * find_global
            global_pred_cdf = xp.cumulative_sum(
                xp.reshape(model_dict["dos"], (-1, self.numb_dos)), axis=-1
            )
            global_label_cdf = xp.cumulative_sum(
                xp.reshape(label_dict["dos"], (-1, self.numb_dos)), axis=-1
            )
            diff = global_pred_cdf - global_label_cdf
            if "mask" in model_dict:
                atom_num = xp.sum(model_dict["mask"], axis=-1, keepdims=True)
                l2_global_loss_cdf = xp.mean(
                    xp.sum(xp.square(diff) * atom_num, axis=0) / xp.sum(atom_num)
                )
                atom_num = xp.mean(xp.astype(atom_num, diff.dtype))
            else:
                atom_num = natoms
                l2_global_loss_cdf = xp.mean(xp.square(diff))
            loss += pref_cdf * l2_global_loss_cdf
            more_loss["rmse_global_cdf"] = self.display_if_exist(
                xp.sqrt(l2_global_loss_cdf) / atom_num, find_global
            )

        more_loss["rmse"] = xp.sqrt(loss)
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
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

    def serialize(self) -> dict:
        """Serialize the loss module."""
        return {
            "@class": "DOSLoss",
            "@version": 1,
            "starter_learning_rate": self.starter_learning_rate,
            "numb_dos": self.numb_dos,
            "start_pref_dos": self.start_pref_dos,
            "limit_pref_dos": self.limit_pref_dos,
            "start_pref_cdf": self.start_pref_cdf,
            "limit_pref_cdf": self.limit_pref_cdf,
            "start_pref_ados": self.start_pref_ados,
            "limit_pref_ados": self.limit_pref_ados,
            "start_pref_acdf": self.start_pref_acdf,
            "limit_pref_acdf": self.limit_pref_acdf,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "DOSLoss":
        """Deserialize the loss module."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
