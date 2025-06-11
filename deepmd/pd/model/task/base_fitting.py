# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle

from deepmd.dpmodel.fitting import (
    make_base_fitting,
)

BaseFitting = make_base_fitting(paddle.Tensor, fwd_method_name="forward")
