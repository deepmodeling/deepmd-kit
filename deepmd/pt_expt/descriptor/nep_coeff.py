# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.descriptor.nep import NepEmbeddingCoeff as NepEmbeddingCoeffDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    to_torch_array,
)


class NepEmbeddingCoeff(NepEmbeddingCoeffDP, torch.nn.Module):
    """PyTorch wrapper storing the NEP expansion coefficients as a parameter.

    The dense ``coeff`` tensor is registered as a trainable ``torch.nn.Parameter``
    (or a buffer when the table is frozen), mirroring how ``NativeLayer`` exposes
    its weights. The gathered contraction in the inherited ``call`` runs with
    plain ``array_api_compat`` torch ops.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        torch.nn.Module.__init__(self)
        NepEmbeddingCoeffDP.__init__(self, *args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return torch.nn.Module.__call__(self, *args, **kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "coeff" and "_parameters" in self.__dict__:
            val = to_torch_array(value)
            if getattr(self, "trainable", False):
                param = (
                    val
                    if isinstance(val, torch.nn.Parameter)
                    else torch.nn.Parameter(val, requires_grad=True)
                )
                if name in self._parameters:
                    self._parameters[name] = param
                    return
                return super().__setattr__(name, param)
            if name in self._buffers:
                self._buffers[name] = val
                return
            self.register_buffer(name, val)
            return
        return super().__setattr__(name, value)

    def forward(self, fn: torch.Tensor, pair_index: torch.Tensor) -> torch.Tensor:
        return self.call(fn, pair_index)


register_dpmodel_mapping(
    NepEmbeddingCoeffDP,
    lambda v: NepEmbeddingCoeff.deserialize(v.serialize()),
)
