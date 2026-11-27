# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.pt.utils.auto_batch_size import AutoBatchSize as AutoBatchSizeBase
from deepmd.pt_expt.utils import (
    env,
)


class AutoBatchSize(AutoBatchSizeBase):
    """Auto batch size following the device pt_expt runs on.

    ``env.DEVICE`` is CPU whenever ``DEVICE=cpu`` is set, even on a CUDA host.
    Growing the batch there risks a host OOM, which the CUDA-OOM handler
    cannot recover from, so the growth policy follows the selected device
    rather than CUDA availability. The module is held rather than the value
    so that the check reads the device in effect at call time.
    """

    def is_gpu_available(self) -> bool:
        """Check if the selected device is a GPU.

        Returns
        -------
        bool
            True if pt_expt runs on a CUDA device
        """
        return env.DEVICE.type == "cuda"
