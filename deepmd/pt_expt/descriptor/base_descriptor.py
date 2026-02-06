# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib

from deepmd.dpmodel.descriptor import (
    make_base_descriptor,
)

torch = importlib.import_module("torch")

BaseDescriptor = make_base_descriptor(torch.Tensor, "forward")
