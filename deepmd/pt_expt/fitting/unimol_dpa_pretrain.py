# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.unimol_dpa_pretrain import (
    UniMolDPAPretrainFitting as UniMolDPAPretrainFittingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("unimol_dpa_pretrain")
@torch_module
class UniMolDPAPretrainFitting(UniMolDPAPretrainFittingDP):
    """Uni-Mol's heads on a DPA backbone, PyTorch-Exportable backend."""
