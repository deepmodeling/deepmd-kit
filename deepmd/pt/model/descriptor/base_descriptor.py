# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.descriptor import (
    make_base_descriptor,
)

BaseDescriptor = make_base_descriptor(torch.Tensor, "forward")
