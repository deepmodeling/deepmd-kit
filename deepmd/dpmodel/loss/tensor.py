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


class TensorLoss(Loss):
    r"""Loss on local and global tensors (e.g. dipole, polarizability).

    Parameters
    ----------
    tensor_name : str
        The name of the tensor in model predictions.
    tensor_size : int
        The size (dimension) of the tensor.
    label_name : str
        The name of the tensor in labels.
    pref_atomic : float
        The prefactor of the weight of atomic (local) loss.
    pref : float
        The prefactor of the weight of global loss.
    enable_atomic_weight : bool
        If true, atomic weight will be used in the loss calculation.
    **kwargs
        Other keyword arguments.
    """

    def __init__(
        self,
        tensor_name: str,
        tensor_size: int,
        label_name: str,
        pref_atomic: float = 0.0,
        pref: float = 0.0,
        enable_atomic_weight: bool = False,
        **kwargs: Any,
    ) -> None:
        self.tensor_name = tensor_name
        self.tensor_size = tensor_size
        self.label_name = label_name
        self.local_weight = pref_atomic
        self.global_weight = pref
        self.enable_atomic_weight = enable_atomic_weight

        assert self.local_weight >= 0.0 and self.global_weight >= 0.0, (
            "Can not assign negative weight to `pref` and `pref_atomic`"
        )
        self.has_local_weight = self.local_weight > 0.0
        self.has_global_weight = self.global_weight > 0.0
        assert self.has_local_weight or self.has_global_weight, (
            "Can not assign zero weight both to `pref` and `pref_atomic`"
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
        del learning_rate, mae
        first_key = next(iter(model_dict))
        xp = array_api_compat.array_namespace(model_dict[first_key])

        if self.enable_atomic_weight:
            atomic_weight = xp.reshape(label_dict["atom_weight"], (-1, 1))
        else:
            atomic_weight = 1.0

        loss = 0
        more_loss = {}

        if (
            self.has_local_weight
            and self.tensor_name in model_dict
            and "atom_" + self.label_name in label_dict
        ):
            find_local = label_dict.get("find_atom_" + self.label_name, 0.0)
            local_weight = self.local_weight * find_local
            local_pred = xp.reshape(
                model_dict[self.tensor_name], (-1, natoms, self.tensor_size)
            )
            local_label = xp.reshape(
                label_dict["atom_" + self.label_name], (-1, natoms, self.tensor_size)
            )
            diff = xp.reshape(local_pred - local_label, (-1, self.tensor_size))
            diff = diff * atomic_weight
            if "mask" in model_dict:
                mask = xp.reshape(model_dict["mask"], (-1,))
                mask_float = xp.astype(mask, diff.dtype)
                diff = diff * mask_float[:, None]
                n_valid = xp.sum(mask_float)
                l2_local_loss = xp.sum(xp.square(diff)) / (n_valid * self.tensor_size)
            else:
                l2_local_loss = xp.mean(xp.square(diff))
            loss += local_weight * l2_local_loss
            more_loss[f"rmse_local_{self.tensor_name}"] = self.display_if_exist(
                xp.sqrt(l2_local_loss), find_local
            )

        if (
            self.has_global_weight
            and "global_" + self.tensor_name in model_dict
            and self.label_name in label_dict
        ):
            find_global = label_dict.get("find_" + self.label_name, 0.0)
            global_weight = self.global_weight * find_global
            global_pred = xp.reshape(
                model_dict["global_" + self.tensor_name], (-1, self.tensor_size)
            )
            global_label = xp.reshape(
                label_dict[self.label_name], (-1, self.tensor_size)
            )
            diff = global_pred - global_label
            if "mask" in model_dict:
                atom_num = xp.sum(model_dict["mask"], axis=-1, keepdims=True)
                l2_global_loss = xp.mean(
                    xp.sum(xp.square(diff) * atom_num, axis=0) / xp.sum(atom_num)
                )
                atom_num = xp.mean(xp.astype(atom_num, diff.dtype))
            else:
                atom_num = natoms
                l2_global_loss = xp.mean(xp.square(diff))
            loss += global_weight * l2_global_loss
            more_loss[f"rmse_global_{self.tensor_name}"] = self.display_if_exist(
                xp.sqrt(l2_global_loss) / atom_num, find_global
            )

        more_loss["rmse"] = xp.sqrt(loss)
        return loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
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
        if self.enable_atomic_weight:
            label_requirement.append(
                DataRequirementItem(
                    "atomic_weight",
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
            "@class": "TensorLoss",
            "@version": 1,
            "tensor_name": self.tensor_name,
            "tensor_size": self.tensor_size,
            "label_name": self.label_name,
            "pref_atomic": self.local_weight,
            "pref": self.global_weight,
            "enable_atomic_weight": self.enable_atomic_weight,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "TensorLoss":
        """Deserialize the loss module."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
