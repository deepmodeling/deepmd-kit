# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.descriptor import (
    make_base_descriptor,
)

BaseDescriptor = make_base_descriptor(torch.Tensor, "forward")
