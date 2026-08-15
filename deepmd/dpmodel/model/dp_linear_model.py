# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel linear energy model: a make_model CM over the linear atomic-model
composition (twin of ``deepmd.pt_expt.model.dp_linear_model``).
"""

from typing import (
    Any,
)

from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_model import (
    DPModelCommon,
)
from deepmd.dpmodel.model.make_model import (
    make_model,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

DPLinearModel_ = make_model(LinearEnergyAtomicModel, T_Bases=(NativeOP, BaseModel))


@BaseModel.register("linear_ener")  # config type
@BaseModel.register("linear")  # wire type emitted by the flat serialize
class LinearEnergyModel(DPModelCommon, DPLinearModel_):
    r"""Energy model over a linear combination of atomic models.

    The atomic energy is the weighted sum of the children's atomic
    energies; on the NeighborGraph route every child consumes the same
    graph, so the summed energy differentiates through one shared edge
    backward. Used e.g. for analytical bridging compositions
    (learned model + :class:`~deepmd.dpmodel.atomic_model.inner_potential.InnerPotentialAtomicModel`).
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPLinearModel_.__init__(self, *args, **kwargs)

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the selection and perform neighbor statistics.

        Updates each learned child in place, skipping analytical
        (``inner_potential``) and pair-table children, and aggregates the
        minimum neighbor distance (twin of the pt_expt implementation).

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
        type_map = local_jdata_cpy["type_map"]
        min_nbor_dist = None
        for idx, sub_model in enumerate(local_jdata_cpy["models"]):
            if sub_model.get("type") == "inner_potential":
                # analytical child: no descriptor, no selection to update
                continue
            if "tab_file" not in sub_model:
                sub_model, temp_min = DPModelCommon.update_sel(
                    train_data, type_map, local_jdata_cpy["models"][idx]
                )
                local_jdata_cpy["models"][idx] = sub_model
                if min_nbor_dist is None or temp_min <= min_nbor_dist:
                    min_nbor_dist = temp_min
        return local_jdata_cpy, min_nbor_dist
