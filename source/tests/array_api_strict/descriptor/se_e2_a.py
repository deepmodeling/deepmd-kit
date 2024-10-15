# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeAArrayAPI as DescrptSeADP

from ..common import (
    to_array_api_strict_array,
)
from ..utils.exclude_mask import (
    PairExcludeMask,
)
from ..utils.network import (
    NetworkCollection,
)


class DescrptSeA(DescrptSeADP):
    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"dstd", "davg"}:
            value = to_array_api_strict_array(value)
        elif name in {"embeddings"}:
            if value is not None:
                value = NetworkCollection.deserialize(value.serialize())
        elif name == "env_mat":
            # env_mat doesn't store any value
            pass
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)

        return super().__setattr__(name, value)
