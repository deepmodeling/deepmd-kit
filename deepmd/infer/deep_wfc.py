# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.infer.deep_tensor import (
    DeepTensor,
)


class DeepWFC(DeepTensor):
    @property
    def output_tensor_name(self) -> str:
        return "wfc"
