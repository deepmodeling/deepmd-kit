# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
)

import torch

from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPolarAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        assert isinstance(fitting, PolarFittingNet)
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: Dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        """Apply the stat to each atomic output.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc

        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)

        if self.fitting_net.shift_diag:
            nframes, nloc = atype.shape
            device = out_bias[self.bias_keys[0]].device
            dtype = out_bias[self.bias_keys[0]].dtype
            for kk in self.bias_keys:
                ntypes = out_bias[kk].shape[0]
                temp = torch.zeros(ntypes, dtype=dtype, device=device)
                for i in range(ntypes):
                    temp[i] = torch.mean(torch.diagonal(out_bias[kk][i].reshape(3, 3)))
                modified_bias = temp[atype]

                # (nframes, nloc, 1)
                modified_bias = (
                    modified_bias.unsqueeze(-1)
                    * (self.fitting_net.scale.to(atype.device))[atype]
                )

                eye = torch.eye(3, dtype=dtype, device=device)
                eye = eye.repeat(nframes, nloc, 1, 1)
                # (nframes, nloc, 3, 3)
                modified_bias = modified_bias.unsqueeze(-1) * eye

                # nf x nloc x odims, out_bias: ntypes x odims
                ret[kk] = ret[kk] + modified_bias
        return ret
