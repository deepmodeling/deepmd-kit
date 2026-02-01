#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
)


@ops.RegisterGradient("FltNvnmd")
def _FltNvnmdGrad(op: Any, grad: Any) -> list[Any]:
    dx = op_module.flt_nvnmd(grad)
    return [dx]
