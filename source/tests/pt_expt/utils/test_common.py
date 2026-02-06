# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib

import numpy as np

from deepmd.pt_expt.common import (
    to_torch_array,
)
from deepmd.pt_expt.utils import (
    env,
)

torch = importlib.import_module("torch")


def test_to_torch_array_moves_device() -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = to_torch_array(arr)
    assert torch.is_tensor(tensor)
    assert tensor.device == env.DEVICE

    input_tensor = torch.as_tensor(arr, device=torch.device("cpu"))
    output_tensor = to_torch_array(input_tensor)
    assert torch.is_tensor(output_tensor)
    assert output_tensor.device == env.DEVICE
