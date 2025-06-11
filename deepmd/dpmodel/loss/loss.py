# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)


class Loss(NativeOP, ABC, make_plugin_registry("loss")):
    @abstractmethod
    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, np.ndarray],
        label_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Calculate loss from model results and labeled results."""

    @property
    @abstractmethod
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""

    @staticmethod
    def display_if_exist(loss: np.ndarray, find_property: float) -> np.ndarray:
        """Display NaN if labeled property is not found.

        Parameters
        ----------
        loss : np.ndarray
            the loss scalar
        find_property : float
            whether the property is found

        Returns
        -------
        np.ndarray
            the loss scalar or NaN
        """
        xp = array_api_compat.array_namespace(loss)
        return xp.where(
            xp.asarray(find_property, dtype=xp.bool), loss, xp.asarray(xp.nan)
        )

    @classmethod
    def get_loss(cls, loss_params: dict) -> "Loss":
        """Get the loss module by the parameters.

        By default, all the parameters are directly passed to the constructor.
        If not, override this method.

        Parameters
        ----------
        loss_params : dict
            The loss parameters

        Returns
        -------
        Loss
            The loss module
        """
        loss = cls(**loss_params)
        return loss

    @abstractmethod
    def serialize(self) -> dict:
        """Serialize the loss module.

        Returns
        -------
        dict
            The serialized loss module
        """

    @classmethod
    @abstractmethod
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
