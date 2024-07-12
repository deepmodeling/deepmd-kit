# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Callable,
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
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.pair_tab import (
    PairTab,
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


@BaseAtomicModel.register("pairtab")
class PairTabAtomicModel(BaseAtomicModel):
    """Pairwise tabulation energy model.

    This model can be used to tabulate the pairwise energy between atoms for either
    short-range or long-range interactions, such as D3, LJ, ZBL, etc. It should not
    be used alone, but rather as one submodel of a linear (sum) model, such as
    DP+D3.

    Do not put the model on the first model of a linear model, since the linear
    model fetches the type map from the first model.

    At this moment, the model does not smooth the energy at the cutoff radius, so
    one needs to make sure the energy has been smoothed to zero.

    Parameters
    ----------
    tab_file : str
        The path to the tabulation file.
    rcut : float
        The cutoff radius.
    sel : int or list[int]
        The maxmum number of atoms in the cut-off radius.
    type_map : List[str]
        Mapping atom type to the name (str) of the type.
        For example `type_map[1]` gives the name of the type 1.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    atom_ener
        Specifying atomic energy contribution in vacuum. The `set_davg_zero` key in the descrptor should be set.

    """

    def __init__(
        self,
        tab_file: str,
        rcut: float,
        sel: Union[int, List[int]],
        type_map: List[str],
        **kwargs,
    ):
        super().__init__(type_map, **kwargs)
        super().init_out_stat()
        self.tab_file = tab_file
        self.rcut = rcut
        self.tab = self._set_pairtab(tab_file, rcut)

        self.type_map = type_map
        self.ntypes = len(type_map)

        # handle deserialization with no input file
        if self.tab_file is not None:
            (
                tab_info,
                tab_data,
            ) = self.tab.get()  # this returns -> Tuple[np.array, np.array]
            nspline, ntypes_tab = tab_info[-2:].astype(int)
            self.register_buffer("tab_info", torch.from_numpy(tab_info))
            self.register_buffer(
                "tab_data",
                torch.from_numpy(tab_data).reshape(ntypes_tab, ntypes_tab, nspline, 4),
            )
            if self.ntypes != ntypes_tab:
                raise ValueError(
                    "The `type_map` provided does not match the number of columns in the table."
                )
        else:
            self.register_buffer("tab_info", None)
            self.register_buffer("tab_data", None)
        self.bias_atom_e = torch.zeros(
            self.ntypes, 1, dtype=env.GLOBAL_PT_ENER_FLOAT_PRECISION, device=env.DEVICE
        )

        # self.model_type = "ener"
        # self.model_version = MODEL_VERSION ## this shoud be in the parent class

        if isinstance(sel, int):
            self.sel = sel
        elif isinstance(sel, list):
            self.sel = sum(sel)
        else:
            raise TypeError("sel must be int or list[int]")

    @torch.jit.ignore
    def _set_pairtab(self, tab_file: str, rcut: float) -> PairTab:
        return PairTab(tab_file, rcut)

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

    def get_out_bias(self) -> torch.Tensor:
        return self.out_bias

    def get_rcut(self) -> float:
        return self.rcut

    def get_type_map(self) -> List[str]:
        return self.type_map

    def get_sel(self) -> List[int]:
        return [self.sel]

    def get_nsel(self) -> int:
        return self.sel

    def mixed_types(self) -> bool:
        """If true, the model
        1. assumes total number of atoms aligned across frames;
        2. uses a neighbor list that does not distinguish different atomic types.

        If false, the model
        1. assumes total number of atoms of each atom type aligned across frames;
        2. uses a neighbor list that distinguishes different atomic types.

        """
        # to match DPA1 and DPA2.
        return True

    def has_message_passing(self) -> bool:
        """Returns whether the atomic model has message passing."""
        return False

    def change_type_map(
        self, type_map: List[str], model_with_new_type_stat=None
    ) -> None:
        """Change the type related params to new ones, according to `type_map` and the original one in the model.
        If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
        """
        assert type_map == self.type_map, (
            "PairTabAtomicModel does not support changing type map now. "
            "This feature is currently not implemented because it would require additional work to change the tab file. "
            "We may consider adding this support in the future if there is a clear demand for it."
        )

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "@version": 2,
                "type": "pairtab",
                "tab": self.tab.serialize(),
                "rcut": self.rcut,
                "sel": self.sel,
                "type_map": self.type_map,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "PairTabAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 2, 1)
        tab = PairTab.deserialize(data.pop("tab"))
        data.pop("@class", None)
        data.pop("type", None)
        data["tab_file"] = None
        tab_model = super().deserialize(data)

        tab_model.tab = tab
        tab_model.register_buffer("tab_info", torch.from_numpy(tab_model.tab.tab_info))
        nspline, ntypes = tab_model.tab.tab_info[-2:].astype(int)
        tab_model.register_buffer(
            "tab_data",
            torch.from_numpy(tab_model.tab.tab_data).reshape(
                ntypes, ntypes, nspline, 4
            ),
        )
        return tab_model

    def compute_or_load_stat(
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
        self.compute_or_load_out_stat(merged, stat_file_path)

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        nframes, nloc, nnei = nlist.shape
        extended_coord = extended_coord.view(nframes, -1, 3)
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)

        # this will mask all -1 in the nlist
        mask = nlist >= 0
        masked_nlist = nlist * mask

        atype = extended_atype[:, :nloc]  # (nframes, nloc)
        pairwise_rr = self._get_pairwise_dist(
            extended_coord, masked_nlist
        )  # (nframes, nloc, nnei)
        self.tab_data = self.tab_data.to(device=extended_coord.device).view(
            int(self.tab_info[-1]), int(self.tab_info[-1]), int(self.tab_info[2]), 4
        )

        # to calculate the atomic_energy, we need 3 tensors, i_type, j_type, pairwise_rr
        # i_type : (nframes, nloc), this is atype.
        # j_type : (nframes, nloc, nnei)
        j_type = extended_atype[
            torch.arange(extended_atype.size(0), device=extended_coord.device)[
                :, None, None
            ],
            masked_nlist,
        ]

        raw_atomic_energy = self._pair_tabulated_inter(
            nlist, atype, j_type, pairwise_rr
        )

        atomic_energy = 0.5 * torch.sum(
            torch.where(
                nlist != -1, raw_atomic_energy, torch.zeros_like(raw_atomic_energy)
            ),
            dim=-1,
        ).unsqueeze(-1)

        return {"energy": atomic_energy}

    def _pair_tabulated_inter(
        self,
        nlist: torch.Tensor,
        i_type: torch.Tensor,
        j_type: torch.Tensor,
        rr: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise tabulated energy.

        Parameters
        ----------
        nlist : torch.Tensor
            The unmasked neighbour list. (nframes, nloc)
        i_type : torch.Tensor
            The integer representation of atom type for all local atoms for all frames. (nframes, nloc)
        j_type : torch.Tensor
            The integer representation of atom type for all neighbour atoms of all local atoms for all frames. (nframes, nloc, nnei)
        rr : torch.Tensor
            The salar distance vector between two atoms. (nframes, nloc, nnei)

        Returns
        -------
        torch.Tensor
            The masked atomic energy for all local atoms for all frames. (nframes, nloc, nnei)

        Raises
        ------
        Exception
            If the distance is beyond the table.

        Notes
        -----
        This function is used to calculate the pairwise energy between two atoms.
        It uses a table containing cubic spline coefficients calculated in PairTab.
        """
        nframes, nloc, nnei = nlist.shape
        rmin = self.tab_info[0]
        hh = self.tab_info[1]
        hi = 1.0 / hh

        nspline = int(self.tab_info[2] + 0.1)

        uu = (rr - rmin) * hi  # this is broadcasted to (nframes,nloc,nnei)

        # if nnei of atom 0 has -1 in the nlist, uu would be 0.
        # this is to handle the nlist where the mask is set to 0, so that we don't raise exception for those atoms.
        uu = torch.where(nlist != -1, uu, nspline + 1)

        if torch.any(uu < 0):
            raise Exception("coord go beyond table lower boundary")

        idx = uu.to(torch.int)

        uu -= idx

        table_coef = self._extract_spline_coefficient(
            i_type, j_type, idx, self.tab_data, nspline
        )
        table_coef = table_coef.view(nframes, nloc, nnei, 4)
        ener = self._calculate_ener(table_coef, uu)

        # here we need to overwrite energy to zero at rcut and beyond.
        mask_beyond_rcut = rr >= self.rcut
        # also overwrite values beyond extrapolation to zero
        extrapolation_mask = rr >= rmin + nspline * hh
        ener[mask_beyond_rcut] = 0
        ener[extrapolation_mask] = 0

        return ener

    @staticmethod
    def _get_pairwise_dist(coords: torch.Tensor, nlist: torch.Tensor) -> torch.Tensor:
        """Get pairwise distance `dr`.

        Parameters
        ----------
        coords : torch.Tensor
            The coordinate of the atoms, shape of (nframes, nall, 3).
        nlist
            The masked nlist, shape of (nframes, nloc, nnei)

        Returns
        -------
        torch.Tensor
            The pairwise distance between the atoms (nframes, nloc, nnei).
        """
        nframes, nloc, nnei = nlist.shape
        coord_l = coords[:, :nloc].view(nframes, -1, 1, 3)
        index = nlist.view(nframes, -1).unsqueeze(-1).expand(-1, -1, 3)
        coord_r = torch.gather(coords, 1, index)
        coord_r = coord_r.view(nframes, nloc, nnei, 3)
        diff = coord_r - coord_l
        pairwise_rr = torch.linalg.norm(diff, dim=-1, keepdim=True).squeeze(-1)
        return pairwise_rr

    @staticmethod
    def _extract_spline_coefficient(
        i_type: torch.Tensor,
        j_type: torch.Tensor,
        idx: torch.Tensor,
        tab_data: torch.Tensor,
        nspline: int,
    ) -> torch.Tensor:
        """Extract the spline coefficient from the table.

        Parameters
        ----------
        i_type : torch.Tensor
            The integer representation of atom type for all local atoms for all frames. (nframes, nloc)
        j_type : torch.Tensor
            The integer representation of atom type for all neighbour atoms of all local atoms for all frames. (nframes, nloc, nnei)
        idx : torch.Tensor
            The index of the spline coefficient. (nframes, nloc, nnei)
        tab_data : torch.Tensor
            The table storing all the spline coefficient. (ntype, ntype, nspline, 4)
        nspline : int
            The number of splines in the table.

        Returns
        -------
        torch.Tensor
            The spline coefficient. (nframes, nloc, nnei, 4), shape may be squeezed.

        """
        # (nframes, nloc, nnei)
        expanded_i_type = i_type.unsqueeze(-1).expand(-1, -1, j_type.shape[-1])

        # handle the case where idx is beyond the number of splines
        clipped_indices = torch.clamp(idx, 0, nspline - 1).to(torch.int64)

        nframes = i_type.shape[0]
        nloc = i_type.shape[1]
        nnei = j_type.shape[2]
        ntypes = tab_data.shape[0]
        # tab_data_idx: (nframes, nloc, nnei)
        tab_data_idx = (
            expanded_i_type * ntypes * nspline + j_type * nspline + clipped_indices
        )
        # tab_data: (ntype, ntype, nspline, 4)
        tab_data = tab_data.view(ntypes * ntypes * nspline, 4)
        # tab_data_idx: (nframes * nloc * nnei, 4)
        tab_data_idx = tab_data_idx.view(nframes * nloc * nnei, 1).expand(-1, 4)
        # (nframes, nloc, nnei, 4)
        final_coef = torch.gather(tab_data, 0, tab_data_idx).view(
            nframes, nloc, nnei, 4
        )

        # when the spline idx is beyond the table, all spline coefficients are set to `0`, and the resulting ener corresponding to the idx is also `0`.
        final_coef[idx > nspline] = 0
        return final_coef

    @staticmethod
    def _calculate_ener(coef: torch.Tensor, uu: torch.Tensor) -> torch.Tensor:
        """Calculate energy using spline coeeficients.

        Parameters
        ----------
        coef : torch.Tensor
            The spline coefficients. (nframes, nloc, nnei, 4)
        uu : torch.Tensor
            The atom displancemnt used in interpolation and extrapolation (nframes, nloc, nnei)

        Returns
        -------
        torch.Tensor
            The atomic energy for all local atoms for all frames. (nframes, nloc, nnei)
        """
        a3, a2, a1, a0 = torch.unbind(coef, dim=-1)
        etmp = (a3 * uu + a2) * uu + a1  # this should be elementwise operations.
        ener = etmp * uu + a0  # this energy has the extrapolated value when rcut > rmax
        return ener

    def get_dim_fparam(self) -> int:
        """Get the number (dimension) of frame parameters of this atomic model."""
        return 0

    def get_dim_aparam(self) -> int:
        """Get the number (dimension) of atomic parameters of this atomic model."""
        return 0

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    def is_aparam_nall(self) -> bool:
        """Check whether the shape of atomic parameters is (nframes, nall, ndim).

        If False, the shape is (nframes, nloc, ndim).
        """
        return False
