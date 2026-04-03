# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import torch

from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        if not isinstance(fitting, PropertyFittingNet):
            raise TypeError(
                "fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def get_compute_stats_distinguish_types(self) -> bool:
        """Get whether the fitting net computes stats which are not distinguished between different types of atoms."""
        return self.fitting_net.get_distinguish_types()

    def get_intensive(self) -> bool:
        """Whether the fitting property is intensive."""
        return self.fitting_net.get_intensive()

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply the stat to each atomic output.
        In property fitting, each output will be multiplied by label std and then plus the label average value.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc. It is useless in property fitting.

        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        if self.get_compute_stats_distinguish_types():
            for kk in self.bias_keys:
                ret[kk] = ret[kk] * out_std[kk][atype] + out_bias[kk][atype]
        else:
            for kk in self.bias_keys:
                ret[kk] = ret[kk] * out_std[kk][0] + out_bias[kk][0]
        return ret


class DPXASAtomicModel(DPPropertyAtomicModel):
    """Atomic model for XAS spectrum fitting.

    Extends :class:`DPPropertyAtomicModel` with a per-(absorbing_type, edge)
    energy reference buffer ``xas_e_ref`` [ntypes, nfparam, 2].  The buffer is
    populated by :meth:`deepmd.pt.loss.xas.XASLoss.compute_output_stats` before
    training starts and is saved in the model checkpoint so that absolute edge
    energies can be reconstructed at inference time without any external files.
    """

    def __init__(
        self, descriptor: Any, fitting: Any, type_map: Any, **kwargs: Any
    ) -> None:
        super().__init__(descriptor, fitting, type_map, **kwargs)
        nfparam: int = getattr(fitting, "numb_fparam", 0)
        if nfparam > 0:
            ntypes: int = len(type_map)
            self.register_buffer(
                "xas_e_ref",
                torch.zeros(ntypes, nfparam, 2, dtype=torch.float64),
            )
        else:
            self.xas_e_ref: torch.Tensor | None = None
