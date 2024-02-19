# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.dpmodel.utils import (
    Spin,
)

from .base_atomic_model import (
    BaseAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)


class WrapperAtomicModel(torch.nn.Module, BaseAtomicModel):
    """Wrapper model that has an existing model inside
    with additionally transformation on the input and output for the model.

    Parameters
    ----------
    model : BaseAtomicModel
        A model to be wrapped inside.
    """

    def __init__(
        self,
        model: BaseAtomicModel,
        **kwargs,
    ):
        super().__init__()
        self.model = model

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return self.model.distinguish_types()

    @torch.jit.export
    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.model.get_rcut()

    @torch.jit.export
    def get_type_map(self) -> List[str]:
        """Get the type map."""
        raise self.model.get_type_map()

    def get_sel(self) -> List[int]:
        return self.model.get_sel()

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coodinates in extended region, (nframes, nall * 3)
        extended_atype
            atomic type in extended region, (nframes, nall)
        nlist
            neighbor list, (nframes, nloc, nsel).
        mapping
            mapps the extended indices to local indices.
        fparam
            frame parameter. (nframes, ndf)
        aparam
            atomic parameter. (nframes, nloc, nda)

        Returns
        -------
        result_dict
            the result dict, defined by the fitting net output def.
        """
        return self.model.forward_atomic(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam,
            aparam,
        )

    def fitting_output_def(self) -> FittingOutputDef:
        return self.model.fitting_output_def()

    @staticmethod
    def serialize(model) -> dict:
        return {
            "model": model.serialize(),
            "model_name": model.__class__.__name__,
        }

    @staticmethod
    def deserialize(data) -> "BaseAtomicModel":
        model = getattr(sys.modules[__name__], data["model_name"]).deserialize(
            data["model"]
        )
        return model

    @torch.jit.export
    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return self.model.get_dim_fparam()

    @torch.jit.export
    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return self.model.get_dim_aparam()

    @torch.jit.export
    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return self.model.get_sel_type()

    @torch.jit.export
    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False


class DPSpinWrapperAtomicModel(WrapperAtomicModel):
    """Spin model wrapper with an AtomicModel.

    Parameters
    ----------
    backbone_model
            The backbone model wrapped inside.
    spin
            The object containing spin settings.
    """

    def __init__(
        self,
        backbone_model: DPAtomicModel,
        spin: Spin,
        **kwargs,
    ):
        super().__init__(backbone_model, **kwargs)
        self.spin = spin

    def serialize(self) -> dict:
        return {
            "wrapper_model": WrapperAtomicModel.serialize(self.model),
            "spin": self.spin.serialize(),
        }

    @classmethod
    def deserialize(cls, data) -> "DPSpinWrapperAtomicModel":
        spin = Spin.deserialize(data["spin"])
        backbone_model = WrapperAtomicModel.deserialize(data["wrapper_model"])
        return cls(
            backbone_model=backbone_model,
            spin=spin,
        )
