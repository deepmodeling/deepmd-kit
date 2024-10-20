# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import paddle

from deepmd.utils.data import (
    DataRequirementItem,
)


class TaskLoss(paddle.nn.Layer, ABC):
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
