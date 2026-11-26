# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.dpmodel.fitting.latent_charge import (
    LatentChargeFitting as LatentChargeFittingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)


@torch_module
class LatentChargeFitting(LatentChargeFittingDP):
    """Per-atom latent charges predicted from the invariant descriptor (pt_expt)."""
