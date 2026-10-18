# SPDX-License-Identifier: LGPL-3.0-or-later
import torch

from deepmd.dpmodel.long_range.latent_charge import (
    LatentChargeLongRange as LatentChargeLongRangeDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)

# The charge network and the kernels built by the dpmodel class are converted
# to their pt_expt modules on assignment; importing them registers the
# conversions.
from deepmd.pt_expt.fitting import latent_charge as _latent_charge_fitting  # noqa: F401

from . import kernel as _kernel  # noqa: F401
from .base_long_range import (
    BaseLongRange,
)


@BaseLongRange.register("latent_charge")
@torch_module
class LatentChargeLongRange(LatentChargeLongRangeDP):
    """Energy of interacting latent charges in complete non-periodic systems (pt_expt)."""

    def _pair_kernel(
        self, r_sq: torch.Tensor, src: torch.Tensor, n: int
    ) -> torch.Tensor:
        # A graph traced for a symbolic pair count plans the reduction of the
        # coefficient gradients over the pairs before it sees a batch. In
        # training the coefficients therefore reach the pairs through the node
        # axis, which turns that reduction into a scatter over the pairs and a
        # sum over the nodes, parallel in the pair count of every batch. A
        # concrete pair count is reduced directly.
        if not (self.training and isinstance(r_sq.shape[0], torch.SymInt)):
            return self.kernel(r_sq)
        coefficients = {
            name: value.expand(n, *value.shape)[src]
            for name, value in self.kernel.coefficients(r_sq).items()
        }
        return self.kernel.evaluate(r_sq, coefficients)
