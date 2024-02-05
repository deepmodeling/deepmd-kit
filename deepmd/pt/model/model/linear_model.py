# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
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
        If "zbl", the weights are calculated internally. This only allows the combination of a PairTabModel and a DPAtomicModel.
    """

    def __init__(
        self,
        models: List[Union[DPAtomicModel, PairTabModel]],
        weights: Union[str, List[float]],
        **kwargs,
    ):
        super().__init__()
        self.models = models
        self.weights = weights
        if self.weights == "zbl":
            if len(models) != 2:
                raise ValueError("ZBL only supports two models.")
            if not isinstance(models[1], PairTabModel):
                raise ValueError(
                    "The PairTabModel must be placed after the DPAtomicModel in the input lists."
                )

        if isinstance(weights, list):
            if len(weights) != len(models):
                raise ValueError(
                    "The length of weights is not equal to the number of models"
                )
            self.weights = weights
        elif weights == "mean":
            self.weights = [1 / len(models) for _ in range(len(models))]
        elif weights == "sum":
            self.weights = [1 for _ in range(len(models))]
        # TODO: add more weights, for example, so-called committee models
        elif weights == "zbl":
            pass
        else:
            raise ValueError(f"Invalid weights {weights}")

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return all(model.distinguish_types() for model in self.models)

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return max(self.get_rcuts)

    def get_rcuts(self) -> List[float]:
        """Get the cut-off radius for each individual models in ascending order."""
        return [model.get_rcut() for model in self.models]

    def get_sel(self) -> List[int]:
        return [max(self.get_sels)]

    def get_sels(self) -> List[int]:
        """Get the cut-off radius for each individual models in ascending order."""
        return [
            sum(model.get_sel())
            if isinstance(model.get_sel(), list)
            else model.get_sel()
            for model in self.models
        ]

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
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

        Returns
        -------
        result_dict
            the result dict, defined by the fitting net output def.
        """
        nframes, nloc, nnei = nlist.shape
        extended_coord = extended_coord.view(nframes, -1, 3)
        nlists = build_multiple_neighbor_list(
            extended_coord,
            nlist,
            self.get_rcuts(),
            self.get_sels(),
        )
        nlists_ = [
            nlists[str(rcut) + "_" + str(sel)]
            for rcut, sel in zip(self.get_rcuts(), self.get_sels())
        ]
        ener_list = [
            model.forward_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
            )["energy"]
            for model, nlist in zip(self.models, nlists_)
        ]

        fit_ret = {
            "energy": sum([w * e for w, e in zip(self.weights, ener_list)]),
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

        if weights == "zbl":
            if len(data["models"]) != 2:
                raise ValueError("ZBL only supports two models.")
            try:
                models = [
                    DPAtomicModel.deserialize(data["models"][0]),
                    PairTabModel.deserialize(data["models"][1]),
                ]
            except KeyError:
                raise ValueError(
                    "The PairTabModel must be placed after the DPAtomicModel in the input lists."
                )

        else:
            models = [DPAtomicModel.deserialize(model) for model in data["models"]]
        return cls(models, weights)


class ZBLModel(LinearModel):
    """Model linearly combine a list of AtomicModels.

    Parameters
    ----------
    models
            This linear model should take a DPAtomicModel and a PairTable model.
    """

    def __init__(
        self,
        models: List[Union[DPAtomicModel, PairTabModel]],
        sw_rmin: float,
        sw_rmax: float,
        weights="zbl",
        smin_alpha: Optional[float] = 0.1,
        **kwargs,
    ):
        super().__init__(models, weights)
        self.dp_model = models[0]
        self.zbl_model = models[1]
        if weights != "zbl":
            raise ValueError("ZBLModel only supports weights 'zbl'.")
        if not (
            isinstance(self.dp_model, DPAtomicModel)
            and isinstance(self.zbl_model, PairTabModel)
        ):
            raise ValueError(
                "The input models for ZBLModel must be a DPAtomicModel and a PairTabModel in the exact order."
            )
        self.sw_rmin = sw_rmin
        self.sw_rmax = sw_rmax
        self.smin_alpha = smin_alpha

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return self.dp_model.distinguish_types() and self.zbl_model.distinguish_types()

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return atomic prediction.

        Note: currently only support the linear combination of a ZBL model and a DP model,
        the weight is calculated based on this paper:
        Appl. Phys. Lett. 114, 244101 (2019); https://doi.org/10.1063/1.5098061

        Parameters
        ----------
        extended_coord
            coodinates in extended region, (nframes, nall * 3)
        extended_atype
            atomic type in extended region, (nframes, nall)
        nlist
            neighbor list, (nframes, nloc, nsel).
        mapping
            mapps the extended indices to local indices

        Returns
        -------
        result_dict
            the result dict, defined by the fitting net output def.
        """
        nframes, nloc, nnei = nlist.shape
        extended_coord = extended_coord.view(nframes, -1, 3)
        nlists = build_multiple_neighbor_list(
            extended_coord,
            nlist,
            [self.zbl_model.rcut, self.dp_model.rcut],
            [self.zbl_model.sel, sum(self.dp_model.sel)],
        )
        zbl_nlist = nlists[str(self.zbl_model.rcut) + "_" + str(self.zbl_model.sel)]
        dp_nlist = nlists[str(self.dp_model.rcut) + "_" + str(sum(self.dp_model.sel))]

        zbl_nnei = zbl_nlist.shape[-1]
        dp_nnei = dp_nlist.shape[-1]

        # use the larger rr based on nlist
        nlist_ = zbl_nlist if zbl_nnei >= dp_nnei else dp_nlist
        masked_nlist = torch.clamp(nlist_, 0)
        pairwise_rr = (
            (extended_coord.unsqueeze(2) - extended_coord.unsqueeze(1))
            .pow(2)
            .sum(-1)
            .sqrt()
        )
        rr = torch.gather(pairwise_rr[:, :nloc, :], 2, masked_nlist)
        # (nframes, nloc, 1)
        self.zbl_weight = self._compute_weight(
            nlist_, rr, self.sw_rmin, self.sw_rmax, self.smin_alpha
        )
        # (nframes, nloc, 1)
        dp_energy = self.dp_model.forward_atomic(
            extended_coord, extended_atype, dp_nlist
        )["energy"]
        # (nframes, nloc, 1)
        zbl_energy = self.zbl_model.forward_atomic(
            extended_coord, extended_atype, zbl_nlist
        )["energy"]
        assert zbl_energy.shape == (nframes, nloc, 1)
        fit_ret = {
            "energy": (self.zbl_weight * zbl_energy + (1 - self.zbl_weight) * dp_energy)
        }  # (nframes, nloc, 1)
        return fit_ret

    def serialize(self) -> dict:
        return {
            "models": [model.serialize() for model in self.models],
            "weights": self.weights,
            "sw_rmin": self.sw_rmin,
            "sw_rmax": self.sw_rmax,
            "smin_alpha": self.smin_alpha,
        }

    @classmethod
    def deserialize(cls, data) -> "ZBLModel":
        weights = data["weights"]
        sw_rmin = data["sw_rmin"]
        sw_rmax = data["sw_rmax"]
        smin_alpha = data["smin_alpha"]

        if weights == "zbl":
            if len(data["models"]) != 2:
                raise ValueError("ZBL only supports two models.")
            try:
                models = [
                    DPAtomicModel.deserialize(data["models"][0]),
                    PairTabModel.deserialize(data["models"][1]),
                ]
            except KeyError:
                raise ValueError(
                    "The PairTabModel must be placed after the DPAtomicModel in the input lists."
                )

        else:
            raise ValueError("ZBLModel only supports weights 'zbl'.")
        return cls(
            models=models,
            weights=weights,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            smin_alpha=smin_alpha,
        )

    @staticmethod
    def _compute_weight(
        nlist: torch.Tensor,
        rr: torch.Tensor,
        sw_rmin: float,
        sw_rmax: float,
        smin_alpha: Optional[float] = 0.1,
    ) -> torch.Tensor:
        """ZBL weight.

        Parameters
        ----------
        nlist : torch.Tensor
            the neighbour list, (nframes, nloc, nnei).
        rr : torch.Tensor
            pairwise distance between atom i and atom j, (nframes, nloc, nnei).
        sw_rmin : float
            inclusive lower boundary of the range in which the ZBL potential and the deep potential are interpolated.
        sw_rmax : float
            exclusive upper boundary of the range in which the ZBL potential and the deep potential are interpolated.
        smin_alpha : float
            a tunable scale of the distances between atoms.

        Returns
        -------
        torch.Tensor
            the atomic ZBL weight for interpolation. (nframes, nloc, 1)
        """
        assert (
            sw_rmax > sw_rmin
        ), "The upper boundary `sw_rmax` must be greater than the lower boundary `sw_rmin`."

        numerator = torch.sum(
            rr * torch.exp(-rr / smin_alpha), dim=-1
        )  # masked nnei will be zero, no need to handle
        denominator = torch.sum(
            torch.where(
                nlist != -1, torch.exp(-rr / smin_alpha), torch.zeros_like(nlist)
            ),
            dim=-1,
        )  # handle masked nnei.
        sigma = numerator / denominator
        u = (sigma - sw_rmin) / (sw_rmax - sw_rmin)
        coef = torch.zeros_like(u)
        left_mask = sigma < sw_rmin
        mid_mask = (sw_rmin <= sigma) & (sigma < sw_rmax)
        right_mask = sigma >= sw_rmax
        coef[left_mask] = 1
        smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef[mid_mask] = smooth[mid_mask]
        coef[right_mask] = 0
        return coef.unsqueeze(-1)
