# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
    nlist_distinguish_types,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
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


@BaseAtomicModel.register("linear")
class LinearEnergyAtomicModel(BaseAtomicModel):
    """Linear model make linear combinations of several existing models.

    Parameters
    ----------
    models : list[DPAtomicModel or PairTabAtomicModel]
        A list of models to be combined. PairTabAtomicModel must be used together with a DPAtomicModel.
    type_map : list[str]
        Mapping atom type to the name (str) of the type.
        For example `type_map[1]` gives the name of the type 1.
    """

    def __init__(
        self,
        models: list[BaseAtomicModel],
        type_map: list[str],
        **kwargs,
    ) -> None:
        super().__init__(type_map, **kwargs)
        super().init_out_stat()

        # check all sub models are of mixed type.
        model_mixed_type = []
        for m in models:
            if not m.mixed_types():
                model_mixed_type.append(m)
        if len(model_mixed_type) > 0:
            raise ValueError(
                f"LinearAtomicModel only supports AtomicModel of mixed type, the following models are not mixed type: {model_mixed_type}."
            )

        self.models = models
        sub_model_type_maps = [md.get_type_map() for md in models]
        err_msg = []
        mapping_list = []
        common_type_map = set(type_map)
        self.type_map = type_map
        for tpmp in sub_model_type_maps:
            if not common_type_map.issubset(set(tpmp)):
                err_msg.append(
                    f"type_map {tpmp} is not a subset of type_map {type_map}"
                )
            mapping_list.append(self.remap_atype(tpmp, self.type_map))
        self.mapping_list = mapping_list
        assert len(err_msg) == 0, "\n".join(err_msg)
        self.mixed_types_list = [model.mixed_types() for model in self.models]

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        return True

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return any(model.has_message_passing() for model in self.models)

    def need_sorted_nlist_for_lower(self) -> bool:
        """Returns whether the atomic model needs sorted nlist when using `forward_lower`."""
        return True

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return max(self.get_model_rcuts())

    def get_type_map(self) -> list[str]:
        """Get the type map."""
        return self.type_map

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        super().change_type_map(
            type_map=type_map, model_with_new_type_stat=model_with_new_type_stat
        )
        for ii, model in enumerate(self.models):
            model.change_type_map(
                type_map=type_map,
                model_with_new_type_stat=model_with_new_type_stat.models[ii]
                if model_with_new_type_stat is not None
                else None,
            )

    def get_model_rcuts(self) -> list[float]:
        """Get the cut-off radius for each individual models."""
        return [model.get_rcut() for model in self.models]

    def get_sel(self) -> list[int]:
        return [max([model.get_nsel() for model in self.models])]

    def set_case_embd(self, case_idx: int):
        """
        Set the case embedding of this atomic model by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        for model in self.models:
            model.set_case_embd(case_idx)

    def get_model_nsels(self) -> list[int]:
        """Get the processed sels for each individual models. Not distinguishing types."""
        return [model.get_nsel() for model in self.models]

    def get_model_sels(self) -> list[Union[int, list[int]]]:
        """Get the sels for each individual models."""
        return [model.get_sel() for model in self.models]

    def _sort_rcuts_sels(self) -> tuple[list[float], list[int]]:
        # sort the pair of rcut and sels in ascending order, first based on sel, then on rcut.
        zipped = sorted(
            zip(self.get_model_rcuts(), self.get_model_nsels()),
            key=lambda x: (x[1], x[0]),
        )
        return [p[0] for p in zipped], [p[1] for p in zipped]

    def enable_compression(
        self,
        min_nbor_dist: float,
        table_extrapolate: float = 5,
        table_stride_1: float = 0.01,
        table_stride_2: float = 0.1,
        check_frequency: int = -1,
    ) -> None:
        """Compress model.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between atoms
        table_extrapolate
            The scale of model extrapolation
        table_stride_1
            The uniform stride of the first table
        table_stride_2
            The uniform stride of the second table
        check_frequency
            The overflow check frequency
        """
        for model in self.models:
            model.enable_compression(
                min_nbor_dist,
                table_extrapolate,
                table_stride_1,
                table_stride_2,
                check_frequency,
            )

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Return atomic prediction.

        Parameters
        ----------
        extended_coord
            coordinates in extended region, (nframes, nall * 3)
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
        xp = array_api_compat.array_namespace(extended_coord, extended_atype, nlist)
        nframes, nloc, nnei = nlist.shape
        extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
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
        ener_list = []
        for i, model in enumerate(self.models):
            type_map_model = self.mapping_list[i]
            ener_list.append(
                model.forward_atomic(
                    extended_coord,
                    type_map_model[extended_atype],
                    nlists_[i],
                    mapping,
                    fparam,
                    aparam,
                )["energy"]
            )
        weights = self._compute_weight(extended_coord, extended_atype, nlists_)

        fit_ret = {
            "energy": xp.sum(xp.stack(ener_list) * xp.stack(weights), axis=0),
        }  # (nframes, nloc, 1)
        return fit_ret

    @staticmethod
    def remap_atype(ori_map: list[str], new_map: list[str]) -> np.ndarray:
        """
        This method is used to map the atype from the common type_map to the original type_map of
        indivial AtomicModels.

        Parameters
        ----------
        ori_map : list[str]
            The original type map of an AtomicModel.
        new_map : list[str]
            The common type map of the DPZBLLinearEnergyAtomicModel, created by the `get_type_map` method,
            must be a subset of the ori_map.

        Returns
        -------
        np.ndarray
        """
        type_2_idx = {atp: idx for idx, atp in enumerate(ori_map)}
        # this maps the atype in the new map to the original map
        mapping = np.array([type_2_idx[new_map[idx]] for idx in range(len(new_map))])
        return mapping

    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                )
            ]
        )

    def serialize(self) -> dict:
        dd = super().serialize()
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "linear",
                "models": [model.serialize() for model in self.models],
                "type_map": self.type_map,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "LinearEnergyAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 2), 2, 2)
        data.pop("@class", None)
        data.pop("type", None)
        models = [
            BaseAtomicModel.get_class_by_type(model["type"]).deserialize(model)
            for model in data["models"]
        ]
        data["models"] = models
        return super().deserialize(data)

    def _compute_weight(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlists_: list[np.ndarray],
    ) -> list[np.ndarray]:
        """This should be a list of user defined weights that matches the number of models to be combined."""
        xp = array_api_compat.array_namespace(extended_coord, extended_atype, nlists_)
        nmodels = len(self.models)
        nframes, nloc, _ = nlists_[0].shape
        # the dtype of weights is the interface data type.
        return [
            xp.ones((nframes, nloc, 1), dtype=GLOBAL_NP_FLOAT_PRECISION) / nmodels
            for _ in range(nmodels)
        ]

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        # tricky...
        return max([model.get_dim_fparam() for model in self.models])

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return max([model.get_dim_aparam() for model in self.models])

    def get_sel_type(self) -> list[int]:
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


@BaseAtomicModel.register("zbl")
class DPZBLLinearEnergyAtomicModel(LinearEnergyAtomicModel):
    """Model linearly combine a list of AtomicModels.

    Parameters
    ----------
    dp_model
        The DPAtomicModel being combined.
    zbl_model
        The PairTable model being combined.
    sw_rmin
        The lower boundary of the interpolation between short-range tabulated interaction and DP.
    sw_rmax
        The upper boundary of the interpolation between short-range tabulated interaction and DP.
    type_map
        Mapping atom type to the name (str) of the type.
        For example `type_map[1]` gives the name of the type 1.
    smin_alpha
        The short-range tabulated interaction will be switched according to the distance of the nearest neighbor.
        This distance is calculated by softmin.
    """

    def __init__(
        self,
        dp_model: DPAtomicModel,
        zbl_model: PairTabAtomicModel,
        sw_rmin: float,
        sw_rmax: float,
        type_map: list[str],
        smin_alpha: Optional[float] = 0.1,
        **kwargs,
    ) -> None:
        models = [dp_model, zbl_model]
        kwargs["models"] = models
        kwargs["type_map"] = type_map
        super().__init__(**kwargs)

        self.sw_rmin = sw_rmin
        self.sw_rmax = sw_rmax
        self.smin_alpha = smin_alpha

    def serialize(self) -> dict:
        dd = super().serialize()
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "zbl",
                "sw_rmin": self.sw_rmin,
                "sw_rmax": self.sw_rmax,
                "smin_alpha": self.smin_alpha,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "DPZBLLinearEnergyAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 2, 2)
        models = [
            BaseAtomicModel.get_class_by_type(model["type"]).deserialize(model)
            for model in data["models"]
        ]
        data["dp_model"], data["zbl_model"] = models[0], models[1]
        data.pop("@class", None)
        data.pop("type", None)
        return super().deserialize(data)

    def set_case_embd(self, case_idx: int):
        """
        Set the case embedding of this atomic model by the given case_idx,
        typically concatenated with the output of the descriptor and fed into the fitting net.
        """
        # only set case_idx for dpmodel
        self.models[0].set_case_embd(case_idx)

    def _compute_weight(
        self,
        extended_coord: np.ndarray,
        extended_atype: np.ndarray,
        nlists_: list[np.ndarray],
    ) -> list[np.ndarray]:
        """ZBL weight.

        Returns
        -------
        list[np.ndarray]
            the atomic ZBL weight for interpolation. (nframes, nloc, 1)
        """
        assert self.sw_rmax > self.sw_rmin, (
            "The upper boundary `sw_rmax` must be greater than the lower boundary `sw_rmin`."
        )

        xp = array_api_compat.array_namespace(extended_coord, extended_atype)
        dp_nlist = nlists_[0]
        zbl_nlist = nlists_[1]

        zbl_nnei = zbl_nlist.shape[-1]
        dp_nnei = dp_nlist.shape[-1]

        # use the larger rr based on nlist
        nlist_larger = zbl_nlist if zbl_nnei >= dp_nnei else dp_nlist
        masked_nlist = xp.clip(nlist_larger, 0, None)
        pairwise_rr = PairTabAtomicModel._get_pairwise_dist(
            extended_coord, masked_nlist
        )

        numerator = xp.sum(
            xp.where(
                nlist_larger != -1,
                pairwise_rr * xp.exp(-pairwise_rr / self.smin_alpha),
                xp.zeros_like(nlist_larger),
            ),
            axis=-1,
        )  # masked nnei will be zero, no need to handle
        denominator = xp.sum(
            xp.where(
                nlist_larger != -1,
                xp.exp(-pairwise_rr / self.smin_alpha),
                xp.zeros_like(nlist_larger),
            ),
            axis=-1,
        )  # handle masked nnei.
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = numerator / denominator
        u = (sigma - self.sw_rmin) / (self.sw_rmax - self.sw_rmin)
        coef = xp.zeros_like(u)
        left_mask = sigma < self.sw_rmin
        mid_mask = (self.sw_rmin <= sigma) & (sigma < self.sw_rmax)
        right_mask = sigma >= self.sw_rmax
        coef = xp.where(left_mask, xp.ones_like(coef), coef)
        with np.errstate(invalid="ignore"):
            smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef = xp.where(mid_mask, smooth, coef)
        coef = xp.where(right_mask, xp.zeros_like(coef), coef)
        # to handle masked atoms
        coef = xp.where(sigma != 0, coef, xp.zeros_like(coef))
        self.zbl_weight = coef
        return [1 - xp.expand_dims(coef, -1), xp.expand_dims(coef, -1)]
