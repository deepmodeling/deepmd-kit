# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import torch

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)

if TYPE_CHECKING:
    from deepmd.utils.path import (
        DPPath,
    )


class BaseModel(torch.nn.Module, make_base_model()):
    def __init__(self, *args, **kwargs):
        """Construct a basic model for different tasks."""
        torch.nn.Module.__init__(self)
        self.model_def_script = ""

    def compute_or_load_stat(
        self,
        sampled_func,
        stat_file_path: DPPath | None = None,
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
        sampled_func
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        raise NotImplementedError

    @torch.jit.export
    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.model_def_script

    @torch.jit.export
    def get_ntypes(self):
        """Returns the number of element types."""
        return len(self.get_type_map())
