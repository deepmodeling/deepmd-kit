# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    NoReturn,
    Optional,
)

import paddle

from deepmd.dpmodel.model.base_model import (
    make_base_model,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.utils.path import (
    DPPath,
)


class BaseModel(paddle.nn.Layer, make_base_model()):
    def __init__(self, *args, **kwargs):
        """Construct a basic model for different tasks."""
        paddle.nn.Layer.__init__(self)
        self.model_def_script = ""
        self.register_buffer(
            "min_nbor_dist",
            paddle.to_tensor(-1.0, dtype=paddle.float64, place=env.DEVICE),
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

    def get_observed_type_list(self) -> list[str]:
        """Get observed types (elements) of the model during data statistics.

        Returns
        -------
        observed_type_list: a list of the observed types in this model.
        """
        raise NotImplementedError

    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        return self.model_def_script

    def get_min_nbor_dist(self) -> Optional[float]:
        """Get the minimum distance between two atoms."""
        if self.min_nbor_dist.item() == -1.0:
            return None
        return self.min_nbor_dist.item()

    def get_ntypes(self):
        """Returns the number of element types."""
        return len(self.get_type_map())
