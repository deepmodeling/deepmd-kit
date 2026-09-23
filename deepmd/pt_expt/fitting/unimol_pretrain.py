# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.fitting.unimol_pretrain import (
    UniMolPretrainFitting as UniMolPretrainFittingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)

from .base_fitting import (
    BaseFitting,
)


@BaseFitting.register("unimol_pretrain")
@torch_module
class UniMolPretrainFitting(UniMolPretrainFittingDP):
    """The Uni-Mol pretraining heads on the PyTorch-Exportable backend."""
