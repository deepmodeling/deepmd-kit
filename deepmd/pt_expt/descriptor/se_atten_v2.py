# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.se_atten_v2 import DescrptSeAttenV2 as DescrptSeAttenV2DP
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt_expt.utils.update_sel import (
    UpdateSel,
)


@BaseDescriptor.register("se_atten_v2")
@torch_module
class DescrptSeAttenV2(DescrptSeAttenV2DP):
    """se_atten_v2 inherits from DPA1 in dpmodel, so compression reuses DPA1 methods."""

    _update_sel_cls = UpdateSel

    def enable_compression(self, *args: Any, **kwargs: Any) -> None:
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        return DescrptDPA1.enable_compression(self, *args, **kwargs)

    def _store_compress_data(self) -> None:
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        return DescrptDPA1._store_compress_data(self)

    def _store_type_embd_data(self) -> None:
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        return DescrptDPA1._store_type_embd_data(self)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        return DescrptDPA1.call(self, *args, **kwargs)

    def _call_compressed(self, *args: Any, **kwargs: Any) -> Any:
        from deepmd.pt_expt.descriptor.dpa1 import (
            DescrptDPA1,
        )

        return DescrptDPA1._call_compressed(self, *args, **kwargs)
