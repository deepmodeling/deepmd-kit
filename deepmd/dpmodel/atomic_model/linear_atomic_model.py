# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import sys
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

import numpy as np

from deepmd.dpmodel.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
    nlist_distinguish_types,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from ..output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from .base_atomic_model import (
    BaseAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)
from .pairtab_atomic_model import (
    PairTabAtomicModel,
)


class LinearAtomicModel(BaseAtomicModel):
    """Linear model make linear combinations of several existing models.

    Parameters
    ----------
    models : list[DPAtomicModel or PairTabAtomicModel]
        A list of models to be combined. PairTabAtomicModel must be used together with a DPAtomicModel.
    """

    def __init__(
        self,
        models: List[BaseAtomicModel],
        **kwargs,
    ):
        self.models = models
        self.mixed_types_list = [model.mixed_types() for model in self.models]
        super().__init__(**kwargs)

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return True

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return max(self.get_model_rcuts())

    def get_type_map(self) -> Optional[List[str]]:
        """Get the type map."""
        raise NotImplementedError("TODO: get_type_map should be implemented")

    def get_model_rcuts(self) -> List[float]:
        """Get the cut-off radius for each individual models."""
        return [model.get_rcut() for model in self.models]

    def get_sel(self) -> List[int]:
        return [max([model.get_nsel() for model in self.models])]

    def get_model_nsels(self) -> List[int]:
        """Get the processed sels for each individual models. Not distinguishing types."""
        return [model.get_nsel() for model in self.models]

    def get_model_sels(self) -> List[Union[int, List[int]]]:
        """Get the sels for each individual models."""
        return [model.get_sel() for model in self.models]

    def _sort_rcuts_sels(self) -> Tuple[List[float], List[int]]:
        # sort the pair of rcut and sels in ascending order, first based on sel, then on rcut.
        zipped = sorted(
            zip(self.get_model_rcuts(), self.get_model_nsels()),
            key=lambda x: (x[1], x[0]),
        )
        return [p[0] for p in zipped], [p[1] for p in zipped]

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
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
        extended_coord = extended_coord.reshape(nframes, -1, 3)
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
        nlists_ = [
            nl if mt else nlist_distinguish_types(nl, extended_atype, sel)
            for mt, nl, sel in zip(
                self.mixed_types_list, raw_nlists, self.get_model_sels()
            )
        ]
        ener_list = [
            model.forward_atomic(
                extended_coord,
                extended_atype,
                nl,
                mapping,
                fparam,
                aparam,
            )["energy"]
            for model, nl in zip(self.models, nlists_)
        ]
        self.weights = self._compute_weight(extended_coord, extended_atype, nlists_)
        self.atomic_bias = None
        if self.atomic_bias is not None:
            raise NotImplementedError("Need to add bias in a future PR.")
        else:
            fit_ret = {
                "energy": np.sum(np.stack(ener_list) * np.stack(self.weights), axis=0),
            }  # (nframes, nloc, 1)
        return fit_ret

    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reduciable=True,
                    r_differentiable=True,
                    c_differentiable=True,
                )
            ]
        )

    @staticmethod
    def serialize(models) -> dict:
        return {
            "@class": "Model",
            "type": "linear",
            "@version": 1,
            "models": [model.serialize() for model in models],
            "model_name": [model.__class__.__name__ for model in models],
        }

    @staticmethod
    def deserialize(data) -> List[BaseAtomicModel]:
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        data.pop("type")
        model_names = data["model_name"]
        models = [
            getattr(sys.modules[__name__], name).deserialize(model)
            for name, model in zip(model_names, data["models"])
        ]
        return models

    @abstractmethod
    def _compute_weight(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlists_: List[np.ndarray],
    ) -> np.ndarray:
        """This should be a list of user defined weights that matches the number of models to be combined."""
        raise NotImplementedError

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        # tricky...
        return max([model.get_dim_fparam() for model in self.models])

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return max([model.get_dim_aparam() for model in self.models])

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        if any(model.get_sel_type() == [] for model in self.models):
            return []
        # join all the selected types
        return list(set().union(*[model.get_sel_type() for model in self.models]))

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False


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
        zbl_model: PairTabAtomicModel,
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
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "type": "zbl",
                "@version": 1,
                "models": LinearAtomicModel.serialize([self.dp_model, self.zbl_model]),
                "sw_rmin": self.sw_rmin,
                "sw_rmax": self.sw_rmax,
                "smin_alpha": self.smin_alpha,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "DPZBLLinearAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        data.pop("type")
        sw_rmin = data.pop("sw_rmin")
        sw_rmax = data.pop("sw_rmax")
        smin_alpha = data.pop("smin_alpha")

        dp_model, zbl_model = LinearAtomicModel.deserialize(data.pop("models"))

        return cls(
            dp_model=dp_model,
            zbl_model=zbl_model,
            sw_rmin=sw_rmin,
            sw_rmax=sw_rmax,
            smin_alpha=smin_alpha,
            **data,
        )

    def _compute_weight(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlists_: List[np.ndarray],
    ) -> List[np.ndarray]:
        """ZBL weight.

        Returns
        -------
        List[np.ndarray]
            the atomic ZBL weight for interpolation. (nframes, nloc, 1)
        """
        assert (
            self.sw_rmax > self.sw_rmin
        ), "The upper boundary `sw_rmax` must be greater than the lower boundary `sw_rmin`."

        dp_nlist = nlists_[0]
        zbl_nlist = nlists_[1]

        zbl_nnei = zbl_nlist.shape[-1]
        dp_nnei = dp_nlist.shape[-1]

        # use the larger rr based on nlist
        nlist_larger = zbl_nlist if zbl_nnei >= dp_nnei else dp_nlist
        masked_nlist = np.clip(nlist_larger, 0, None)
        pairwise_rr = PairTabAtomicModel._get_pairwise_dist(
            extended_coord, masked_nlist
        )

        numerator = np.sum(
            pairwise_rr * np.exp(-pairwise_rr / self.smin_alpha), axis=-1
        )  # masked nnei will be zero, no need to handle
        denominator = np.sum(
            np.where(
                nlist_larger != -1,
                np.exp(-pairwise_rr / self.smin_alpha),
                np.zeros_like(nlist_larger),
            ),
            axis=-1,
        )  # handle masked nnei.
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = numerator / denominator
        u = (sigma - self.sw_rmin) / (self.sw_rmax - self.sw_rmin)
        coef = np.zeros_like(u)
        left_mask = sigma < self.sw_rmin
        mid_mask = (self.sw_rmin <= sigma) & (sigma < self.sw_rmax)
        right_mask = sigma >= self.sw_rmax
        coef[left_mask] = 1
        with np.errstate(invalid="ignore"):
            smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef[mid_mask] = smooth[mid_mask]
        coef[right_mask] = 0
        self.zbl_weight = coef
        return [1 - np.expand_dims(coef, -1), np.expand_dims(coef, -1)]
