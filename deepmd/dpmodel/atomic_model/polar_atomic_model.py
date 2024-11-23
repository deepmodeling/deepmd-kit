# SPDX-License-Identifier: LGPL-3.0-or-later

import array_api_compat
import numpy as np

from deepmd.dpmodel.fitting.polarizability_fitting import (
    PolarFitting,
)

from .dp_atomic_model import (
    DPAtomicModel,
)


class DPPolarAtomicModel(DPAtomicModel):
    def __init__(self, descriptor, fitting, type_map, **kwargs):
        if not isinstance(fitting, PolarFitting):
            raise TypeError(
                "fitting must be an instance of PolarFitting for DPPolarAtomicModel"
            )
        super().__init__(descriptor, fitting, type_map, **kwargs)

    def apply_out_stat(
        self,
        ret: dict[str, np.ndarray],
        atype: np.ndarray,
    ):
        """Apply the stat to each atomic output.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc

        """
        xp = array_api_compat.array_namespace(atype)
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)

        if self.fitting.shift_diag:
            nframes, nloc = atype.shape
            dtype = out_bias[self.bias_keys[0]].dtype
            for kk in self.bias_keys:
                ntypes = out_bias[kk].shape[0]
                temp = xp.mean(
                    xp.diagonal(out_bias[kk].reshape(ntypes, 3, 3), axis1=1, axis2=2),
                    axis=1,
                )
                modified_bias = temp[atype]

                # (nframes, nloc, 1)
                modified_bias = (
                    modified_bias[..., xp.newaxis] * (self.fitting.scale[atype])
                )

                eye = xp.eye(3, dtype=dtype)
                eye = xp.tile(eye, (nframes, nloc, 1, 1))
                # (nframes, nloc, 3, 3)
                modified_bias = modified_bias[..., xp.newaxis] * eye

                # nf x nloc x odims, out_bias: ntypes x odims
                ret[kk] = ret[kk] + modified_bias
        return ret
