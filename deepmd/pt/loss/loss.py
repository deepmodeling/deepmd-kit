# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    List,
)

import torch

from deepmd.utils.data import (
    DataRequirementItem,
)


class TaskLoss(torch.nn.Module):
    def __init__(self, **kwargs):
        """Construct loss."""
        super().__init__()

    def forward(self, model_pred, label, natoms, learning_rate):
        """Return loss ."""
        raise NotImplementedError

    @property
    def label_requirement(self) -> List[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation."""
        raise NotImplementedError
