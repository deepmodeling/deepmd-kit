# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)


@BaseDescriptor.register("hybrid")
@torch_module
class DescrptHybrid(DescrptHybridDP):
    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share parameters with base_class for multi-task training.

        Level 0: share all sub-descriptors.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            for ii, des in enumerate(self.descrpt_list):
                self.descrpt_list[ii].share_params(
                    base_class.descrpt_list[ii],
                    shared_level,
                    model_prob=model_prob,
                    resume=resume,
                )
        else:
            raise NotImplementedError
