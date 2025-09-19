# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.fitting.property_fitting import (
    PropertyFittingNet,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPropertyAtomicModel(DPAtomicModel):
    def __init__(
        self, descriptor: Any, fitting: Any, type_map: list[str], **kwargs: Any
    ) -> None:
        if not isinstance(fitting, PropertyFittingNet):
            raise TypeError(
                "fitting must be an instance of PropertyFittingNet for DPPropertyAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, Array],
        atype: Array,
    ) -> dict[str, Array]:
        """Apply the stat to each atomic output.

        In property fitting, each output will be multiplied by label std and then plus the label average value.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc. It is useless in property fitting.

        """
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        for kk in self.bias_keys:
            ret[kk] = ret[kk] * out_std[kk][0] + out_bias[kk][0]
        return ret
