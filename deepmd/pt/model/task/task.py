# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)

import torch

from deepmd.model_format import (
    FittingOutputDef,
)


class TaskBaseMethod(torch.nn.Module, ABC):
    @abstractmethod
    def output_def(self) -> FittingOutputDef:
        """Definition for the task Output."""
        raise NotImplementedError
