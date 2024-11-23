# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    DPZBLLinearEnergyAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

from .make_model import (
    make_model,
)

DPZBLModel_ = make_model(DPZBLLinearEnergyAtomicModel)


@BaseModel.register("zbl")
class DPZBLModel(DPZBLModel_):
    model_type = "zbl"

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = local_jdata.copy()
        local_jdata_cpy["dpmodel"], min_nbor_dist = DPModelCommon.update_sel(
            train_data, type_map, local_jdata["dpmodel"]
        )
        return local_jdata_cpy, min_nbor_dist
