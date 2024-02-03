# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.fitting import (
    make_base_fitting,
)

BaseFitting = make_base_fitting(torch.Tensor, fwd_method_name="forward")
