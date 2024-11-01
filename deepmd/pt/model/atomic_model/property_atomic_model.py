# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, PropertyFittingNet):
            raise TypeError("fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel")
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        """Apply the stat to each atomic output.
        This function defines how the bias is applied to the atomic output of the model.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc

        """
        if self.fitting_net.get_bias_method() == "normal":
            out_bias, out_std = self._fetch_out_stat(self.bias_keys)
            for kk in self.bias_keys:
                # nf x nloc x odims, out_bias: ntypes x odims
                ret[kk] = ret[kk] + out_bias[kk][atype]
            return ret
        elif self.fitting_net.get_bias_method() == "no_bias":
            return ret
        else:
            raise NotImplementedError(
                "Only 'normal' and 'no_bias' is supported for parameter 'bias_method'."
            )
