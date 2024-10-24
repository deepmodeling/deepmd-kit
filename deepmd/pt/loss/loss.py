# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import torch

from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.plugin import (
    make_plugin_registry,
)


class TaskLoss(torch.nn.Module, ABC, make_plugin_registry("loss")):
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
    def display_if_exist(loss: torch.Tensor, find_property: float) -> torch.Tensor:
        """Display NaN if labeled property is not found.

        Parameters
        ----------
        loss : torch.Tensor
            the loss tensor
        find_property : float
            whether the property is found
        """
        return loss if bool(find_property) else torch.nan
