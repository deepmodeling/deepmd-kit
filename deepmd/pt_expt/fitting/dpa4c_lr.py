# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch-exportable wrapper for the DPA4C long-range fitting."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.fitting.dpa4c_lr import DPA4CLRFitting as DPA4CLRFittingDP
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("dpa4c_lr")
@torch_module
class DPA4CLRFitting(DPA4CLRFittingDP):
    """DPA4C long-range fitting net for the pt_expt backend."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # In the dpmodel these are plain numpy arrays, which the wrapper stores
        # as non-trainable buffers. Promote only the parameters used by the
        # selected kernel so every registered Parameter participates in loss
        # backpropagation.
        if self.lr_kernel == "les":
            self._promote_to_parameter("les_alpha")
        elif self.lr_kernel == "sog":
            self._promote_to_parameter("amp")
            self._promote_to_parameter("bandwidth")
        self._promote_to_parameter("bias_atom_q")

    def _promote_to_parameter(self, name: str) -> None:
        value = getattr(self, name, None)
        if value is None or isinstance(value, torch.nn.Parameter):
            return
        tensor = (
            value.detach().clone()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
        if name in self._buffers:
            del self._buffers[name]
        self.register_parameter(
            name, torch.nn.Parameter(tensor, requires_grad=bool(self.trainable))
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self.lr_kernel == "les":
            # A short-lived SOG implementation registered these unused buffers
            # on LES models. Ignore them when reading such a checkpoint, while
            # older LES checkpoints naturally have neither key.
            state_dict.pop(prefix + "amp", None)
            state_dict.pop(prefix + "bandwidth", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


register_dpmodel_mapping(
    DPA4CLRFittingDP,
    lambda v: DPA4CLRFitting.deserialize(v.serialize()),
)
