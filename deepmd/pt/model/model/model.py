# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    NoReturn,
    Optional,
)

import torch

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)


class BaseModel(torch.nn.Module, make_base_model()):
    def __init__(self, *args, **kwargs) -> None:
        """Construct a basic model for different tasks."""
        torch.nn.Module.__init__(self)
        self.model_def_script = ""
        self.register_buffer(
            "min_nbor_dist", torch.tensor(-1.0, dtype=torch.float64, device=env.DEVICE)
        )

    def compute_or_load_stat(
        self,
        sampled_func,
        stat_file_path: Optional[DPPath] = None,
    ) -> NoReturn:
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
    def get_min_nbor_dist(self) -> Optional[float]:
        """Get the minimum distance between two atoms."""
        if self.min_nbor_dist.item() == -1.0:
            return None
        return self.min_nbor_dist.item()

    @torch.jit.export
    def get_ntypes(self):
        """Returns the number of element types."""
        return len(self.get_type_map())
