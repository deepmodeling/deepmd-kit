# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import paddle

from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)


class TaskLoss(paddle.nn.Layer, ABC, make_plugin_registry("loss")):
    def __init__(self, **kwargs):
        """Construct loss."""
        super().__init__()

    def forward(self, input_dict, model, label, natoms, learning_rate):
        """Return loss ."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        pass

    @staticmethod
    def display_if_exist(loss: paddle.Tensor, find_property: float) -> paddle.Tensor:
        """Display NaN if labeled property is not found.

        Parameters
        ----------
        loss : paddle.Tensor
            the loss tensor
        find_property : float
            whether the property is found
        """
        return loss if bool(find_property) else paddle.to_tensor(float("nan"))

    @classmethod
    def get_loss(cls, loss_params: dict) -> "TaskLoss":
        """Get the loss module by the parameters.

        By default, all the parameters are directly passed to the constructor.
        If not, override this method.

        Parameters
        ----------
        loss_params : dict
            The loss parameters

        Returns
        -------
        TaskLoss
            The loss module
        """
        loss = cls(**loss_params)
        return loss
