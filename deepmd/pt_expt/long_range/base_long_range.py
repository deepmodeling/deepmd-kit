# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.dpmodel.long_range import (
    make_base_long_range,
)

BaseLongRange = make_base_long_range(torch.Tensor)
