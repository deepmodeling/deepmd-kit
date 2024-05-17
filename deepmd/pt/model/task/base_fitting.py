# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.fitting import (
    make_base_fitting,
)

BaseFitting = make_base_fitting(torch.Tensor, fwd_method_name="forward")
