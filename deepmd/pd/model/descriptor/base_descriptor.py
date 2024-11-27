# SPDX-License-Identifier: LGPL-3.0-or-later
import paddle

from deepmd.dpmodel.descriptor import (
    make_base_descriptor,
)

BaseDescriptor = make_base_descriptor(paddle.Tensor, "forward")
