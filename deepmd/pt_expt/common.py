# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
    overload,
)

import numpy as np
import torch

from deepmd.pt_expt.utils import (
    env,
)


@overload
def to_torch_array(array: np.ndarray) -> torch.Tensor: ...


@overload
def to_torch_array(array: None) -> None: ...


@overload
def to_torch_array(array: torch.Tensor) -> torch.Tensor: ...


def to_torch_array(array: Any) -> torch.Tensor | None:
    """Convert input to a torch tensor on the pt-expt device."""
    if array is None:
        return None
    if torch.is_tensor(array):
        return array.to(device=env.DEVICE)
    return torch.as_tensor(array, device=env.DEVICE)
