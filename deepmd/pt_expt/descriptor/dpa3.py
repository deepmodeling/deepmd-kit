# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DescrptDPA3DP
from deepmd.dpmodel.utils.env_mat_stat import (
    merge_env_stat,
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


@BaseDescriptor.register("dpa3")
@torch_module
class DescrptDPA3(DescrptDPA3DP):
    _update_sel_cls = UpdateSel

    def share_params(
        self,
        base_class: "DescrptDPA3",
        shared_level: int,
        model_prob: float = 1.0,
        resume: bool = False,
    ) -> None:
        """Share parameters with base_class for multi-task training.

        Level 0: share type_embedding and repflows.
        Level 1: share type_embedding only.
        """
        assert self.__class__ == base_class.__class__, (
            "Only descriptors of the same type can share params!"
        )
        if shared_level == 0:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
            if not resume:
                merge_env_stat(base_class.repflows, self.repflows, model_prob)
            self._modules["repflows"] = base_class._modules["repflows"]
        elif shared_level == 1:
            self._modules["type_embedding"] = base_class._modules["type_embedding"]
        else:
            raise NotImplementedError
