# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.pair_tab import (
    PairTab,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


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
    """

    def __init__(
        self, tab_file: str, rcut: float, sel: Union[int, List[int]], **kwargs
    ):
        super().__init__()
        self.tab_file = tab_file
        self.rcut = rcut

        self.tab = PairTab(self.tab_file, rcut=rcut)

        if self.tab_file is not None:
            self.tab_info, self.tab_data = self.tab.get()
        else:
            self.tab_info, self.tab_data = None, None

        if isinstance(sel, int):
            self.sel = sel
        elif isinstance(sel, list):
            self.sel = sum(sel)
        else:
            raise TypeError("sel must be int or list[int]")

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

    def get_rcut(self) -> float:
        return self.rcut

    def get_type_map(self) -> Optional[List[str]]:
        raise NotImplementedError("TODO: get_type_map should be implemented")

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

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "type": "pairtab",
                "@version": 1,
                "tab": self.tab.serialize(),
                "rcut": self.rcut,
                "sel": self.sel,
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data) -> "PairTabAtomicModel":
        data = copy.deepcopy(data)
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        data.pop("type")
        rcut = data.pop("rcut")
        sel = data.pop("sel")
        tab = PairTab.deserialize(data.pop("tab"))
        tab_model = cls(None, rcut, sel, **data)
        tab_model.tab = tab
        tab_model.tab_info = tab_model.tab.tab_info
        tab_model.tab_data = tab_model.tab.tab_data
        return tab_model

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[np.ndarray] = None,
        fparam: Optional[np.ndarray] = None,
        aparam: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        nframes, nloc, nnei = nlist.shape
        extended_coord = extended_coord.reshape(nframes, -1, 3)

        # this will mask all -1 in the nlist
        mask = nlist >= 0
        masked_nlist = nlist * mask

        atype = extended_atype[:, :nloc]  # (nframes, nloc)
        pairwise_rr = self._get_pairwise_dist(
            extended_coord, masked_nlist
        )  # (nframes, nloc, nnei)
        self.tab_data = self.tab_data.reshape(
            self.tab.ntypes, self.tab.ntypes, self.tab.nspline, 4
        )

        # (nframes, nloc, nnei)
        j_type = extended_atype[
            np.arange(extended_atype.shape[0])[:, None, None], masked_nlist
        ]

        raw_atomic_energy = self._pair_tabulated_inter(
            nlist, atype, j_type, pairwise_rr
        )
        atomic_energy = 0.5 * np.sum(
            np.where(nlist != -1, raw_atomic_energy, np.zeros_like(raw_atomic_energy)),
            axis=-1,
        ).reshape(nframes, nloc, 1)

        return {"energy": atomic_energy}

    def _pair_tabulated_inter(
        self,
        nlist: np.ndarray,
        i_type: np.ndarray,
        j_type: np.ndarray,
        rr: np.ndarray,
    ) -> np.ndarray:
        """Pairwise tabulated energy.

        Parameters
        ----------
        nlist : np.ndarray
            The unmasked neighbour list. (nframes, nloc)
        i_type : np.ndarray
            The integer representation of atom type for all local atoms for all frames. (nframes, nloc)
        j_type : np.ndarray
            The integer representation of atom type for all neighbour atoms of all local atoms for all frames. (nframes, nloc, nnei)
        rr : np.ndarray
            The salar distance vector between two atoms. (nframes, nloc, nnei)

        Returns
        -------
        np.ndarray
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
        uu = np.where(nlist != -1, uu, nspline + 1)

        if np.any(uu < 0):
            raise Exception("coord go beyond table lower boundary")

        idx = uu.astype(int)

        uu -= idx
        table_coef = self._extract_spline_coefficient(
            i_type, j_type, idx, self.tab_data, nspline
        )
        table_coef = table_coef.reshape(nframes, nloc, nnei, 4)
        ener = self._calculate_ener(table_coef, uu)
        # here we need to overwrite energy to zero at rcut and beyond.
        mask_beyond_rcut = rr >= self.rcut
        # also overwrite values beyond extrapolation to zero
        extrapolation_mask = rr >= self.tab.rmin + nspline * self.tab.hh
        ener[mask_beyond_rcut] = 0
        ener[extrapolation_mask] = 0

        return ener

    @staticmethod
    def _get_pairwise_dist(coords: np.ndarray, nlist: np.ndarray) -> np.ndarray:
        """Get pairwise distance `dr`.

        Parameters
        ----------
        coords : np.ndarray
            The coordinate of the atoms, shape of (nframes, nall, 3).
        nlist
            The masked nlist, shape of (nframes, nloc, nnei).

        Returns
        -------
        np.ndarray
            The pairwise distance between the atoms (nframes, nloc, nnei).
        """
        batch_indices = np.arange(nlist.shape[0])[:, None, None]
        neighbor_atoms = coords[batch_indices, nlist]
        loc_atoms = coords[:, : nlist.shape[1], :]
        pairwise_dr = loc_atoms[:, :, None, :] - neighbor_atoms
        pairwise_rr = np.sqrt(np.sum(np.power(pairwise_dr, 2), axis=-1))

        return pairwise_rr

    @staticmethod
    def _extract_spline_coefficient(
        i_type: np.ndarray,
        j_type: np.ndarray,
        idx: np.ndarray,
        tab_data: np.ndarray,
        nspline: int,
    ) -> np.ndarray:
        """Extract the spline coefficient from the table.

        Parameters
        ----------
        i_type : np.ndarray
            The integer representation of atom type for all local atoms for all frames. (nframes, nloc)
        j_type : np.ndarray
            The integer representation of atom type for all neighbour atoms of all local atoms for all frames. (nframes, nloc, nnei)
        idx : np.ndarray
            The index of the spline coefficient. (nframes, nloc, nnei)
        tab_data : np.ndarray
            The table storing all the spline coefficient. (ntype, ntype, nspline, 4)
        nspline : int
            The number of splines in the table.

        Returns
        -------
        np.ndarray
            The spline coefficient. (nframes, nloc, nnei, 4), shape may be squeezed.
        """
        # (nframes, nloc, nnei)
        expanded_i_type = np.broadcast_to(
            i_type[:, :, np.newaxis],
            (i_type.shape[0], i_type.shape[1], j_type.shape[-1]),
        )

        # (nframes, nloc, nnei, nspline, 4)
        expanded_tab_data = tab_data[expanded_i_type, j_type]

        # (nframes, nloc, nnei, 1, 4)
        expanded_idx = np.broadcast_to(
            idx[..., np.newaxis, np.newaxis], (*idx.shape, 1, 4)
        )
        clipped_indices = np.clip(expanded_idx, 0, nspline - 1).astype(int)

        # (nframes, nloc, nnei, 4)
        final_coef = np.squeeze(
            np.take_along_axis(expanded_tab_data, clipped_indices, 3)
        )

        # when the spline idx is beyond the table, all spline coefficients are set to `0`, and the resulting ener corresponding to the idx is also `0`.
        final_coef[expanded_idx.squeeze() > nspline] = 0
        return final_coef

    @staticmethod
    def _calculate_ener(coef: np.ndarray, uu: np.ndarray) -> np.ndarray:
        """Calculate energy using spline coeeficients.

        Parameters
        ----------
        coef : np.ndarray
            The spline coefficients. (nframes, nloc, nnei, 4)
        uu : np.ndarray
            The atom displancemnt used in interpolation and extrapolation (nframes, nloc, nnei)

        Returns
        -------
        np.ndarray
            The atomic energy for all local atoms for all frames. (nframes, nloc, nnei)
        """
        a3, a2, a1, a0 = coef[..., 0], coef[..., 1], coef[..., 2], coef[..., 3]
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
