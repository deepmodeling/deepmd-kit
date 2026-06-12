# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DescrptDPA4DP
from deepmd.dpmodel.descriptor.dpa4_nn.activation import SwiGLU as SwiGLUDP
from deepmd.dpmodel.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator as WignerDCalculatorDP,
)
from deepmd.pt_expt.common import (
    register_dpmodel_mapping,
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@torch_module
class WignerDCalculator(WignerDCalculatorDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# WignerDCalculator.deserialize raises NotImplementedError by design (its
# tables are derived constants); rebuild from the stored constructor args.
register_dpmodel_mapping(
    WignerDCalculatorDP,
    lambda v: WignerDCalculator(v.lmax, eps=v.eps, precision=v.precision),
)


@torch_module
class SwiGLU(SwiGLUDP):
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)


# SwiGLU is parameter-free (no serialize); rebuild fresh.
register_dpmodel_mapping(SwiGLUDP, lambda v: SwiGLU())


@BaseDescriptor.register("SeZM")
@BaseDescriptor.register("sezm")
@BaseDescriptor.register("DPA4")
@BaseDescriptor.register("dpa4")
@torch_module
class DescrptDPA4(DescrptDPA4DP):
    _update_sel_cls = UpdateSel

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    def share_params(
        self,
        base_class: "DescrptDPA4",
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        # Multi-task parameter sharing for DPA4 is out of scope for this PR.
        raise NotImplementedError("share_params is not yet implemented for DescrptDPA4")
