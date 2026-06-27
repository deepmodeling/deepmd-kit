# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.nep import (
    DescrptNep as DescrptNepDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("nep")
@torch_module
class DescrptNep(DescrptNepDP):
    """PyTorch (pt_expt) wrapper around the array-API NEP descriptor.

    The radial/angular coefficient collections and the exclusion mask are
    standard dpmodel sub-components that ``torch_module`` converts to trainable
    PyTorch modules automatically; the descriptor statistics become buffers. No
    bespoke attribute handling or forward implementation is required.
    """

    _update_sel_cls = UpdateSel

    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        resume: bool = False,
    ) -> None:
        """Share parameters with ``base_class`` for multi-task training.

        Level 0 shares all coefficient modules and statistic buffers.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
            for item in self._buffers:
                self._buffers[item] = base_class._buffers[item]
        else:
            raise NotImplementedError
