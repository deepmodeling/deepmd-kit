# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Callable,
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
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
    get_multiple_nlist_key,
    nlist_distinguish_types,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.version import (
    check_version_compatibility,
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
        models: List[BaseAtomicModel],
        type_map: List[str],
        **kwargs,
    ):
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

        self.models = torch.nn.ModuleList(models)
        sub_model_type_maps = [md.get_type_map() for md in models]
        err_msg = []
        self.mapping_list = []
        common_type_map = set(type_map)
        self.type_map = type_map
        for tpmp in sub_model_type_maps:
            if not common_type_map.issubset(set(tpmp)):
                err_msg.append(
                    f"type_map {tpmp} is not a subset of type_map {type_map}"
                )
            self.mapping_list.append(self.remap_atype(tpmp, self.type_map))
        assert len(err_msg) == 0, "\n".join(err_msg)

        self.mixed_types_list = [model.mixed_types() for model in self.models]
        self.rcuts = torch.tensor(
            self.get_model_rcuts(), dtype=torch.float64, device=env.DEVICE
        )
        self.nsels = torch.tensor(self.get_model_nsels(), device=env.DEVICE)

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

    def get_out_bias(self) -> torch.Tensor:
        return self.out_bias

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return max(self.get_model_rcuts())

    def get_type_map(self) -> List[str]:
        """Get the type map."""
        return self.type_map

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
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
        zipped = torch.stack(
            [
                self.rcuts,
                self.nsels,
            ],
            dim=0,
        ).T
        inner_sorting = torch.argsort(zipped[:, 1], dim=0)
        inner_sorted = zipped[inner_sorting]
        outer_sorting = torch.argsort(inner_sorted[:, 0], stable=True)
        outer_sorted = inner_sorted[outer_sorting]
        sorted_rcuts: List[float] = outer_sorted[:, 0].tolist()
        sorted_sels: List[int] = outer_sorted[:, 1].to(torch.int64).tolist()
        return sorted_rcuts, sorted_sels

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
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
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        extended_coord = extended_coord.view(nframes, -1, 3)
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
            type_map_model = self.mapping_list[i].to(extended_atype.device)
            # apply bias to each individual model
            ener_list.append(
                model.forward_common_atomic(
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
            "energy": torch.sum(
                torch.stack(ener_list) * torch.stack(weights).to(extended_atype.device),
                dim=0,
            ),
        }  # (nframes, nloc, 1)
        return fit_ret

    def apply_out_stat(
        self,
        ret: Dict[str, torch.Tensor],
        atype: torch.Tensor,
    ):
        """Apply the stat to each atomic output.
        The developer may override the method to define how the bias is applied
        to the atomic output of the model.

        Parameters
        ----------
        ret
            The returned dict by the forward_atomic method
        atype
            The atom types. nf x nloc

        """
        return ret

    @staticmethod
    def remap_atype(ori_map: List[str], new_map: List[str]) -> torch.Tensor:
        """
        This method is used to map the atype from the common type_map to the original type_map of
        indivial AtomicModels. It creates a index mapping for the conversion.

        Parameters
        ----------
        ori_map : List[str]
            The original type map of an AtomicModel.
        new_map : List[str]
            The common type map of the DPZBLLinearEnergyAtomicModel, created by the `get_type_map` method,
            must be a subset of the ori_map.

        Returns
        -------
        torch.Tensor
        """
        type_2_idx = {atp: idx for idx, atp in enumerate(ori_map)}
        # this maps the atype in the new map to the original map
        mapping = torch.tensor(
            [type_2_idx[new_map[idx]] for idx in range(len(new_map))], device=env.DEVICE
        )
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
        data = copy.deepcopy(data)
        check_version_compatibility(data.get("@version", 2), 2, 1)
        data.pop("@class", None)
        data.pop("type", None)
        models = [
            BaseAtomicModel.get_class_by_type(model["type"]).deserialize(model)
            for model in data["models"]
        ]
        data["models"] = models
        return super().deserialize(data)

    def _compute_weight(
        self, extended_coord, extended_atype, nlists_
    ) -> List[torch.Tensor]:
        """This should be a list of user defined weights that matches the number of models to be combined."""
        nmodels = len(self.models)
        nframes, nloc, _ = nlists_[0].shape
        return [
            torch.ones((nframes, nloc, 1), dtype=torch.float64, device=env.DEVICE)
            / nmodels
            for _ in range(nmodels)
        ]

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
        # make torch.jit happy...
        return torch.unique(
            torch.cat(
                [
                    torch.as_tensor(model.get_sel_type(), dtype=torch.int32)
                    for model in self.models
                ]
            )
        ).tolist()

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False

    def compute_or_load_out_stat(
        self,
        merged: Union[Callable[[], List[dict]], List[dict]],
        stat_file_path: Optional[DPPath] = None,
    ):
        """
        Compute the output statistics (e.g. energy bias) for the fitting net from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], List[dict]], List[dict]]
            - List[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], List[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        stat_file_path : Optional[DPPath]
            The path to the stat file.

        """
        for md in self.models:
            md.compute_or_load_out_stat(merged, stat_file_path)

    def compute_or_load_stat(
        self,
        sampled_func,
        stat_file_path: Optional[DPPath] = None,
    ):
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
            The lazy sampled function to get data frames from different data systems.
        stat_file_path
            The dictionary of paths to the statistics files.
        """
        for md in self.models:
            md.compute_or_load_stat(sampled_func, stat_file_path)


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
        The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor.
        This distance is calculated by softmin.
    """

    def __init__(
        self,
        dp_model: DPAtomicModel,
        zbl_model: PairTabAtomicModel,
        sw_rmin: float,
        sw_rmax: float,
        type_map: List[str],
        smin_alpha: Optional[float] = 0.1,
        **kwargs,
    ):
        models = [dp_model, zbl_model]
        kwargs["models"] = models
        kwargs["type_map"] = type_map
        super().__init__(**kwargs)

        self.sw_rmin = sw_rmin
        self.sw_rmax = sw_rmax
        self.smin_alpha = smin_alpha

        # this is a placeholder being updated in _compute_weight, to handle Jit attribute init error.
        self.zbl_weight = torch.empty(0, dtype=torch.float64, device=env.DEVICE)

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
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        models = [
            BaseAtomicModel.get_class_by_type(model["type"]).deserialize(model)
            for model in data["models"]
        ]
        data["dp_model"], data["zbl_model"] = models[0], models[1]
        data.pop("@class", None)
        data.pop("type", None)
        return super().deserialize(data)

    def _compute_weight(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlists_: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """ZBL weight.

        Returns
        -------
        List[torch.Tensor]
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
        masked_nlist = torch.clamp(nlist_larger, 0)
        pairwise_rr = PairTabAtomicModel._get_pairwise_dist(
            extended_coord, masked_nlist
        )
        numerator = torch.sum(
            torch.where(
                nlist_larger != -1,
                pairwise_rr * torch.exp(-pairwise_rr / self.smin_alpha),
                torch.zeros_like(nlist_larger),
            ),
            dim=-1,
        )
        denominator = torch.sum(
            torch.where(
                nlist_larger != -1,
                torch.exp(-pairwise_rr / self.smin_alpha),
                torch.zeros_like(nlist_larger),
            ),
            dim=-1,
        )  # handle masked nnei.

        sigma = numerator / torch.clamp(denominator, 1e-20)  # nfrmes, nloc
        u = (sigma - self.sw_rmin) / (self.sw_rmax - self.sw_rmin)
        coef = torch.zeros_like(u)
        left_mask = sigma < self.sw_rmin
        mid_mask = (self.sw_rmin <= sigma) & (sigma < self.sw_rmax)
        right_mask = sigma >= self.sw_rmax
        coef[left_mask] = 1
        smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef[mid_mask] = smooth[mid_mask]
        coef[right_mask] = 0

        # to handle masked atoms
        coef = torch.where(sigma != 0, coef, torch.zeros_like(coef))
        self.zbl_weight = coef  # nframes, nloc
        return [1 - coef.unsqueeze(-1), coef.unsqueeze(-1)]  # to match the model order.
