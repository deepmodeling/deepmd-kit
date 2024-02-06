# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

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
from .pair_tab_model import (
    PairTabModel,
)


class LinearModel(BaseModel, BaseAtomicModel):
    """Linear model make linear combinations of several existing models.

    Parameters
    ----------
    models : list[DPAtomicModel or PairTabModel]
        A list of models to be combined. PairTabModel must be used together with a DPAtomicModel.
    weights : list[float] or str
        If the type is list[float], a list of weights for each model.
        If "mean", the weights are set to be 1 / len(models).
        If "sum", the weights are set to be 1.
        If "switch_by_softmin_pair_distance", the weights are calculated internally. This only allows the combination of a PairTabModel and a DPAtomicModel.
    """

    def __init__(
        self,
        models: List[BaseAtomicModel],
        weights: Union[str, List[float]],
        **kwargs,
    ):
        super().__init__()
        self.models = models
        self.weights = weights
        self.distinguish_type_list = [
            model.distinguish_types() for model in self.models
        ]

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return False

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return max(self.get_rcuts())

    def get_rcuts(self) -> List[float]:
        """Get the cut-off radius for each individual models."""
        return [model.get_rcut() for model in self.models]

    def get_sel(self) -> List[int]:
        """Get the processed sels for each individual models. Not distinguishing types."""
        return [model.get_nsel() for model in self.models]

    def get_original_sels(self) -> List[Union[int, List[int]]]:
        """Get the sels for each individual models."""
        return [model.get_sel() for model in self.models]

    def _sort_rcuts_sels(self) -> Tuple[List[int], List[float]]:
        # sort the pair of rcut and sels in ascending order, first based on sel, then on rcut.
        zipped = sorted(
            zip(self.get_rcuts(), self.get_sel()), key=lambda x: (x[1], x[0])
        )
        return [p[0] for p in zipped], [p[1] for p in zipped]

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
            for rcut, sel in zip(self.get_rcuts(), self.get_sel())
        ]
        self.nlists_ = [
            nl if not dt else nlist_distinguish_types(nl, extended_atype, sel)
            for dt, nl, sel in zip(
                self.distinguish_type_list, raw_nlists, self.get_original_sels()
            )
        ]

        ener_list = [
            model.forward_atomic(
                self.extended_coord,
                extended_atype,
                nl,
                mapping,
            )["energy"]
            for model, nl in zip(self.models, self.nlists_)
        ]

        weights = self._compute_weight()
        self.atomic_bias = None
        if self.atomic_bias is not None:
            raise NotImplementedError("Need to add bias in a future PR.")
        else:
            fit_ret = {
                "energy": sum(torch.stack(ener_list) * torch.stack(weights)),
            }  # (nframes, nloc, 1)
        return fit_ret

    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy", shape=[1], reduciable=True, differentiable=True
                )
            ]
        )

    def serialize(self) -> dict:
        return {
            "models": [model.serialize() for model in self.models],
            "weights": self.weights,
        }

    @classmethod
    def deserialize(cls, data) -> "LinearModel":
        weights = data["weights"]

        if weights == "switch_by_softmin_pair_distance":
            raise NotImplementedError("Use ZBLAtomicModel instead of LinearModel.")
        else:
            models = [DPAtomicModel.deserialize(model) for model in data["models"]]
        return cls(models, weights)

    def _compute_weight(self) -> torch.Tensor:
        if isinstance(self.weights, list):
            if len(self.weights) != len(self.models):
                raise ValueError(
                    "The length of weights is not equal to the number of models"
                )
            return self.weights
        elif self.weights == "mean":
            return [1 / len(self.models) for _ in range(len(self.models))]
        elif self.weights == "sum":
            return [1 for _ in range(len(self.models))]
        # TODO: add more weights, for example, so-called committee models
        elif self.weights == "switch_by_softmin_pair_distance":
            raise NotImplementedError("Use ZBLAtomicModel instead of LinearModel.")
        else:
            raise ValueError(f"Invalid weights {self.weights}")


class ZBLAtomicModel(LinearModel):
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
        weights="switch_by_softmin_pair_distance",
        smin_alpha: Optional[float] = 0.1,
        **kwargs,
    ):
        models = [dp_model, zbl_model]
        super().__init__(models, weights, **kwargs)
        self.dp_model = dp_model
        self.zbl_model = zbl_model
        if weights != "switch_by_softmin_pair_distance":
            raise ValueError(
                "ZBLAtomicModel only supports weights 'switch_by_softmin_pair_distance'."
            )

        self.sw_rmin = sw_rmin
        self.sw_rmax = sw_rmax
        self.smin_alpha = smin_alpha

    def serialize(self) -> dict:
        return {
            "dp_model": self.dp_model.serialize(),
            "zbl_model": self.zbl_model.serialize(),
            "weights": self.weights,
            "sw_rmin": self.sw_rmin,
            "sw_rmax": self.sw_rmax,
            "smin_alpha": self.smin_alpha,
        }

    @classmethod
    def deserialize(cls, data) -> "ZBLAtomicModel":
        weights = data["weights"]
        sw_rmin = data["sw_rmin"]
        sw_rmax = data["sw_rmax"]
        smin_alpha = data["smin_alpha"]

        if weights == "switch_by_softmin_pair_distance":
            dp_model = DPAtomicModel.deserialize(data["dp_model"])
            zbl_model = PairTabModel.deserialize(data["zbl_model"])
        else:
            raise ValueError(
                "ZBLAtomicModel only supports weights 'switch_by_softmin_pair_distance'."
            )
        return cls(
            dp_model=dp_model,
            zbl_model=zbl_model,
            weights=weights,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            smin_alpha=smin_alpha,
        )

    def _compute_weight(self) -> torch.Tensor:
        """ZBL weight.

        Returns
        -------
        torch.Tensor
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
            (self.extended_coord.unsqueeze(2) - self.extended_coord.unsqueeze(1))
            .pow(2)
            .sum(-1)
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
