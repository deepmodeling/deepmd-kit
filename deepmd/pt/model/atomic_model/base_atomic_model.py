# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.atomic_model import (
    make_base_atomic_model,
)

BaseAtomicModel = make_base_atomic_model(torch.Tensor)
