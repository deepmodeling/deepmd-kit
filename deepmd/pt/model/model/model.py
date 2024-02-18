# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.utils.path import (
    DPPath,
)


class BaseModel(torch.nn.Module):
    def __init__(self):
        """Construct a basic model for different tasks."""
        super().__init__()

    def compute_or_load_stat(
        self,
        sampled,
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute or load the statistics parameters of the model,
        such as mean and standard deviation of descriptors or the energy bias of the fitting net.
        When `sampled` is provided, all the statistics parameters will be calculated (or re-calculated for update),
        and saved in the `stat_file_path`(s).
        When `sampled` is not provided, it will check the existence of `stat_file_path`(s)
        and load the calculated statistics parameters.

        Parameters
        ----------
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        raise NotImplementedError
