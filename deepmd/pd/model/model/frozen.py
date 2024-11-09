# SPDX-License-Identifier: LGPL-3.0-or-later
import json
from typing import (
    Optional,
)

import paddle

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)
from deepmd.pd.model.model.model import (
    BaseModel,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


@BaseModel.register("frozen")
class FrozenModel(BaseModel):
    """Load model from a frozen model, which cannot be trained.

    Parameters
    ----------
    model_file : str
        The path to the frozen model
    """

    def __init__(self, model_file: str, **kwargs):
        super().__init__(**kwargs)
        self.model_file = model_file
        if model_file.endswith(".json"):
            self.model = paddle.jit.load(model_file.split(".json")[0])
        else:
            raise NotImplementedError(
                f"Only support .json file, but received {model_file}"
            )

    def fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of developer implemented atomic models."""
        return self.model.fitting_output_def()

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.model.get_rcut()

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.model.get_type_map()

    def get_sel(self) -> list[int]:
        """Returns the number of selected atoms for each type."""
        return self.model.get_sel()

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.model.get_dim_fparam()

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.model.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.model.get_sel_type()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return self.model.is_aparam_nall()

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return self.model.mixed_types()

    def has_message_passing(self) -> bool:
        """Returns whether the descriptor has message passing."""
        return self.model.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the model needs sorted nlist when using `forward_lower`."""
        return self.model.need_sorted_nlist_for_lower()

    def forward(
        self,
        coord,
        atype,
        box: Optional[paddle.Tensor] = None,
        fparam: Optional[paddle.Tensor] = None,
        aparam: Optional[paddle.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, paddle.Tensor]:
        return self.model.forward(
            coord,
            atype,
            box=box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

    def get_model_def_script(self) -> str:
        """Get the model definition script."""
        # try to use the original script instead of "frozen model"
        # Note: this cannot change the script of the parent model
        # it may still try to load hard-coded filename, which might
        # be a problem
        return self.model.get_model_def_script()

    def get_min_nbor_dist(self) -> Optional[float]:
        """Get the minimum neighbor distance."""
        return self.model.get_min_nbor_dist()

    def serialize(self) -> dict:
        from deepmd.pd.model.model import (
            get_model,
        )

        # try to recover the original model
        model_def_script = json.loads(self.get_model_def_script())
        model = get_model(model_def_script)
        model.set_state_dict(self.model.state_dict())
        return model.serialize()

    @classmethod
    def deserialize(cls, data: dict):
        raise RuntimeError("Should not touch here.")

    def get_nnei(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.model.get_nnei()

    def get_nsel(self) -> int:
        """Returns the total number of selected neighboring atoms in the cut-off radius."""
        return self.model.get_nsel()

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
        return local_jdata, None

    def model_output_type(self) -> str:
        """Get the output type for the model."""
        return self.model.model_output_type()
