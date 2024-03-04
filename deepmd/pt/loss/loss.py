# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    List,
)

import torch

from deepmd.utils.data import (
    DataRequirementItem,
)


class TaskLoss(torch.nn.Module, ABC):
    def __init__(self, **kwargs):
        """Construct loss."""
        super().__init__()

    def forward(self, model_pred, label, natoms, learning_rate):
        """Return loss ."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        pass
