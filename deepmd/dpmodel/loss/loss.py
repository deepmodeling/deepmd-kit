# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
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
    r"""Base interface for objectives :math:`L(\hat y,y)`.

    Concrete losses map model outputs and labels to a scalar objective and
    return diagnostic metrics alongside it.
    """

    @abstractmethod
    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate loss from model results and labeled results.

        Returns
        -------
        loss : Array
            The scalar loss to minimize.
        more_loss : dict[str, Array]
            Additional loss terms/metrics for logging.
        """

    @property
    @abstractmethod
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""

    @property
    def supports_ragged_batches(self) -> bool:
        """Whether this objective accepts a flat per-node batch axis."""
        return False

    @staticmethod
    def display_if_exist(loss: Array, find_property: float) -> Array:
        """Display NaN if labeled property is not found.

        Parameters
        ----------
        loss : Array
            the loss scalar
        find_property : float
            whether the property is found

        Returns
        -------
        np.ndarray
            the loss scalar or NaN
        """
        xp = array_api_compat.array_namespace(loss)
        dev = array_api_compat.device(loss)
        # ``full_like`` passes NaN as a scalar kernel argument, where
        # ``asarray(xp.nan, device=dev)`` would copy it from the host: a
        # synchronizing transfer, once per reported quantity per step, on a
        # value that never changes.
        return xp.where(
            xp.asarray(find_property, dtype=xp.bool, device=dev),
            loss,
            xp.full_like(loss, xp.nan),
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
