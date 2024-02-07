# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    abstractmethod,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import sys
import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
    nlist_distinguish_types,
)

from .base_atomic_model import (
    BaseAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)
from .model import (
    BaseModel,
)
from .pairtab_atomic_model import (
    PairTabModel,
)


class LinearAtomicModel(BaseModel, BaseAtomicModel):
    """Linear model make linear combinations of several existing models.

    Parameters
    ----------
    models : list[DPAtomicModel or PairTabModel]
        A list of models to be combined. PairTabModel must be used together with a DPAtomicModel.
    """

    def __init__(
        self,
        models: List[BaseAtomicModel],
        **kwargs,
    ):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.distinguish_type_list = [
            model.distinguish_types() for model in self.models
        ]

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return False

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return max(self.get_model_rcuts())

    def get_model_rcuts(self) -> List[float]:
        """Get the cut-off radius for each individual models."""
        return [model.get_rcut() for model in self.models]

    def get_sel(self) -> List[int]:
        return [max([model.get_nsel() for model in self.models])]
    
    def get_model_nsels(self) -> List[int]:
        """Get the processed sels for each individual models. Not distinguishing types."""
        return [model.get_nsel() for model in self.models]

    def get_model_sels(self) -> List[List[int]]:
        """Get the sels for each individual models."""
        return [model.get_sel() for model in self.models]

    def _sort_rcuts_sels(self) -> Tuple[List[float], List[int]]:
        # sort the pair of rcut and sels in ascending order, first based on sel, then on rcut.
        rcuts = torch.tensor(self.get_model_rcuts(), dtype=torch.float64)
        nsels = torch.tensor(self.get_model_nsels())
        zipped = torch.stack([torch.tensor(rcuts), torch.tensor(nsels)], dim=0).T
        inner_sorting = torch.argsort(zipped[:, 1],dim=0)
        inner_sorted = zipped[inner_sorting]
        outer_sorting = torch.argsort(inner_sorted[:, 0],stable=True)
        outer_sorted = inner_sorted[outer_sorting]
        sorted_rcuts: List[float] = outer_sorted[:,0].tolist()
        sorted_sels: List[int] = outer_sorted[:,1].to(torch.int64).tolist()
        return  sorted_rcuts, sorted_sels

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
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
        nframes, nloc, nnei = nlist.shape
        if self.do_grad():
            extended_coord.requires_grad_(True)
        self.extended_coord = extended_coord.view(nframes, -1, 3)
        sorted_rcuts, sorted_sels = self._sort_rcuts_sels()
        nlists = build_multiple_neighbor_list(
            extended_coord,
            nlist,
            sorted_rcuts,
            sorted_sels,
        )
        raw_nlists = [
            nlists[get_multiple_nlist_key(rcut, sel)]
            for rcut, sel in zip(self.get_model_rcuts(), self.get_model_nsels())
        ]
        self.nlists_ = [
            nl if not dt else nlist_distinguish_types(nl, extended_atype, sel)
            for dt, nl, sel in zip(
                self.distinguish_type_list, raw_nlists, self.get_model_sels()
            )
        ]
        ener_list = []
        
        for i, model in enumerate(self.models):
            ener_list.append(
                model.forward_atomic(
                    self.extended_coord,
                    extended_atype,
                    self.nlists_[i],
                    mapping,
                    fparam,
                    aparam,
                )["energy"]
            )

        self.weights = self._compute_weight()
        self.atomic_bias = None
        if self.atomic_bias is not None:
            raise NotImplementedError("Need to add bias in a future PR.")
        else:
            fit_ret = {
                "energy": torch.sum(
                    torch.stack(ener_list) * torch.stack(self.weights), dim=0
                ),
            }  # (nframes, nloc, 1)
        return fit_ret

    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy", shape=[1], reduciable=True, r_differentiable=True, c_differentiable=True
                )
            ]
        )

    @staticmethod
    def serialize(models) -> dict:
        return {
            "models": [model.serialize() for model in models],
            "model_name": [model.__class__.__name__ for model in models]
        }

    @staticmethod
    def deserialize(data) -> List[BaseAtomicModel]:
        
        model_names = data["model_name"]
        models = [getattr(sys.modules[__name__], name).deserialize(model) for name, model in zip(model_names, data["models"])]
        return models

    @abstractmethod
    def _compute_weight(self) -> List[torch.Tensor]:
        """This should be a list of user defined weights that matches the number of models to be combined."""
        raise NotImplementedError


class DPZBLLinearAtomicModel(LinearAtomicModel):
    """Model linearly combine a list of AtomicModels.

    Parameters
    ----------
    models
            This linear model should take a DPAtomicModel and a PairTable model.
    """

    def __init__(
        self,
        dp_model: DPAtomicModel,
        zbl_model: PairTabModel,
        sw_rmin: float,
        sw_rmax: float,
        smin_alpha: Optional[float] = 0.1,
        **kwargs,
    ):
        models = [dp_model, zbl_model]
        super().__init__(models, **kwargs)
        self.dp_model = dp_model
        self.zbl_model = zbl_model

        self.sw_rmin = sw_rmin
        self.sw_rmax = sw_rmax
        self.smin_alpha = smin_alpha

    def serialize(self) -> dict:
        return {
            "models": LinearAtomicModel.serialize([self.dp_model, self.zbl_model]),
            "sw_rmin": self.sw_rmin,
            "sw_rmax": self.sw_rmax,
            "smin_alpha": self.smin_alpha,
        }

    @classmethod
    def deserialize(cls, data) -> "DPZBLLinearAtomicModel":
        sw_rmin = data["sw_rmin"]
        sw_rmax = data["sw_rmax"]
        smin_alpha = data["smin_alpha"]

        dp_model, zbl_model = LinearAtomicModel.deserialize(data["models"])

        return cls(
            dp_model=dp_model,
            zbl_model=zbl_model,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            smin_alpha=smin_alpha,
        )

    def _compute_weight(self) -> List[torch.Tensor]:
        """ZBL weight.

        Returns
        -------
        List[torch.Tensor]
            the atomic ZBL weight for interpolation. (nframes, nloc, 1)
        """
        assert (
            self.sw_rmax > self.sw_rmin
        ), "The upper boundary `sw_rmax` must be greater than the lower boundary `sw_rmin`."

        dp_nlist = self.nlists_[0]
        zbl_nlist = self.nlists_[1]

        zbl_nnei = zbl_nlist.shape[-1]
        dp_nnei = dp_nlist.shape[-1]

        # use the larger rr based on nlist
        nlist_larger = zbl_nlist if zbl_nnei >= dp_nnei else dp_nlist
        nloc = nlist_larger.shape[1]
        masked_nlist = torch.clamp(nlist_larger, 0)
        pairwise_rr = (
            torch.clamp((self.extended_coord.unsqueeze(2) - self.extended_coord.unsqueeze(1))
            .square()
            .sum(-1), 1E-19)
            .sqrt()
        )
        rr = torch.gather(
            pairwise_rr[:, :nloc, :], 2, masked_nlist
        )  # nframes, nloc, nnei

        numerator = torch.sum(
            rr * torch.exp(-rr / self.smin_alpha), dim=-1
        )  # masked nnei will be zero, no need to handle
        denominator = torch.sum(
            torch.where(
                nlist_larger != -1,
                torch.exp(-rr / self.smin_alpha),
                torch.zeros_like(nlist_larger),
            ),
            dim=-1,
        )  # handle masked nnei.

        sigma = numerator / denominator  # nfrmes, nloc
        u = (sigma - self.sw_rmin) / (self.sw_rmax - self.sw_rmin)
        coef = torch.zeros_like(u)
        left_mask = sigma < self.sw_rmin
        mid_mask = (self.sw_rmin <= sigma) & (sigma < self.sw_rmax)
        right_mask = sigma >= self.sw_rmax
        coef[left_mask] = 1
        smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef[mid_mask] = smooth[mid_mask]
        coef[right_mask] = 0
        self.zbl_weight = coef  # nframes, nloc
        return [1 - coef.unsqueeze(-1), coef.unsqueeze(-1)]  # to match the model order.
