# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np
import torch
from torch import (
    nn,
)

from deepmd.model_format import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.pair_tab import (
    PairTab,
)

from .atomic_model import (
    AtomicModel,
)


class PairTabModel(nn.Module, AtomicModel):
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

        # check table data against rcut and update tab_file if needed.
        self._check_table_upper_boundary()

        self.tab = PairTab(self.tab_file)
        self.ntypes = self.tab.ntypes

        tab_info, tab_data = self.tab.get()  # this returns -> Tuple[np.array, np.array]
        self.tab_info = torch.from_numpy(tab_info)
        self.tab_data = torch.from_numpy(tab_data)

        # self.model_type = "ener"
        # self.model_version = MODEL_VERSION ## this shoud be in the parent class

        if isinstance(sel, int):
            self.sel = sel
        elif isinstance(sel, list):
            self.sel = sum(sel)
        else:
            raise TypeError("sel must be int or list[int]")

    def get_fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy", shape=[1], reduciable=True, differentiable=True
                )
            ]
        )

    def get_rcut(self) -> float:
        return self.rcut

    def get_sel(self) -> int:
        return self.sel

    def distinguish_types(self) -> bool:
        # to match DPA1 and DPA2.
        return False

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        nframes, nloc, nnei = nlist.shape

        # this will mask all -1 in the nlist
        masked_nlist = torch.clamp(nlist, 0)

        atype = extended_atype[:, :nloc]  # (nframes, nloc)
        pairwise_dr = self._get_pairwise_dist(
            extended_coord
        )  # (nframes, nall, nall, 3)
        pairwise_rr = pairwise_dr.pow(2).sum(-1).sqrt()  # (nframes, nall, nall)

        self.tab_data = self.tab_data.reshape(
            self.tab.ntypes, self.tab.ntypes, self.tab.nspline, 4
        )

        # to calculate the atomic_energy, we need 3 tensors, i_type, j_type, rr
        # i_type : (nframes, nloc), this is atype.
        # j_type : (nframes, nloc, nnei)
        j_type = extended_atype[
            torch.arange(extended_atype.size(0))[:, None, None], masked_nlist
        ]

        # slice rr to get (nframes, nloc, nnei)
        rr = torch.gather(pairwise_rr[:, :nloc, :], 2, masked_nlist)

        raw_atomic_energy = self._pair_tabulated_inter(nlist, atype, j_type, rr)

        atomic_energy = 0.5 * torch.sum(
            torch.where(
                nlist != -1, raw_atomic_energy, torch.zeros_like(raw_atomic_energy)
            ),
            dim=-1,
        )

        return {"energy": atomic_energy}

    def _check_table_upper_boundary(self):
        """Update User Provided Table Based on `rcut`.

        This function checks the upper boundary provided in the table against rcut.
        If the table upper boundary values decay to zero before rcut, padding zeros will
        be added to the table to cover rcut; if the table upper boundary values do not decay to zero
        before ruct, linear extrapolation will be performed till rcut. In both cases, the table file
        will be overwritten.

        Examples
        --------
        table = [[0.005 1.    2.    3.   ]
                [0.01  0.8   1.6   2.4  ]
                [0.015 0.    1.    1.5  ]]

        rcut = 0.022

        new_table = [[0.005 1.    2.    3.   ]
                    [0.01  0.8   1.6   2.4  ]
                    [0.015 0.    1.    1.5  ]
                    [0.02  0.    0.    0.   ]
                    [0.025 0.    0.    0.   ]]

        ----------------------------------------------

        table = [[0.005 1.    2.    3.   ]
                [0.01  0.8   1.6   2.4  ]
                [0.015 0.5   1.    1.5  ]
                [0.02  0.25  0.4   0.75 ]
                [0.025 0.    0.1   0.   ]
                [0.03  0.    0.    0.   ]]

        rcut = 0.031

        new_table = [[0.005 1.    2.    3.   ]
                    [0.01  0.8   1.6   2.4  ]
                    [0.015 0.5   1.    1.5  ]
                    [0.02  0.25  0.4   0.75 ]
                    [0.025 0.    0.1   0.   ]
                    [0.03  0.    0.    0.   ]
                    [0.035 0.    0.    0.   ]]
        """
        raw_data = np.loadtxt(self.tab_file)
        upper = raw_data[-1][0]
        upper_val = raw_data[-1][1:]
        upper_idx = raw_data.shape[0] - 1
        increment = raw_data[1][0] - raw_data[0][0]

        # the index of table for the grid point right after rcut
        rcut_idx = int(self.rcut / increment)

        if np.all(upper_val == 0):
            # if table values decay to `0` after rcut
            if self.rcut < upper and np.any(raw_data[rcut_idx - 1][1:] != 0):
                logging.warning(
                    "The energy provided in the table does not decay to 0 at rcut."
                )
            # if table values decay to `0` at rcut, do nothing

            # if table values decay to `0` before rcut, pad table with `0`s.
            elif self.rcut > upper:
                pad_zero = np.zeros((rcut_idx - upper_idx, 4))
                pad_zero[:, 0] = np.linspace(
                    upper + increment, increment * (rcut_idx + 1), rcut_idx - upper_idx
                )
                raw_data = np.concatenate((raw_data, pad_zero), axis=0)
        else:
            # if table values do not decay to `0` at rcut
            if self.rcut <= upper:
                logging.warning(
                    "The energy provided in the table does not decay to 0 at rcut."
                )
            # if rcut goes beyond table upper bond, need extrapolation, ensure values decay to `0` before rcut.
            else:
                logging.warning(
                    "The rcut goes beyond table upper boundary, performing linear extrapolation."
                )
                pad_linear = np.zeros((rcut_idx - upper_idx + 1, 4))
                pad_linear[:, 0] = np.linspace(
                    upper, increment * (rcut_idx + 1), rcut_idx - upper_idx + 1
                )
                pad_linear[:-1, 1:] = np.array(
                    [
                        np.linspace(start, 0, rcut_idx - upper_idx)
                        for start in upper_val
                    ]
                ).T
                raw_data = np.concatenate((raw_data[:-1, :], pad_linear), axis=0)

        # over writing file with padding if applicable.
        with open(self.tab_file, "wb") as f:
            np.savetxt(f, raw_data)

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
        rmin = self.tab_info[0]
        hh = self.tab_info[1]
        hi = 1.0 / hh

        self.nspline = int(self.tab_info[2] + 0.1)

        uu = (rr - rmin) * hi  # this is broadcasted to (nframes,nloc,nnei)

        # if nnei of atom 0 has -1 in the nlist, uu would be 0.
        # this is to handel the nlist where the mask is set to 0, so that we don't raise exception for those atoms.
        uu = torch.where(nlist != -1, uu, self.nspline + 1)

        if torch.any(uu < 0):
            raise Exception("coord go beyond table lower boundary")

        idx = uu.to(torch.int)

        uu -= idx

        final_coef = self._extract_spline_coefficient(i_type, j_type, idx)

        a3, a2, a1, a0 = torch.unbind(final_coef, dim=-1)  # 4 * (nframes, nloc, nnei)

        etmp = (a3 * uu + a2) * uu + a1  # this should be elementwise operations.
        ener = etmp * uu + a0
        return ener

    @staticmethod
    def _get_pairwise_dist(coords: torch.Tensor) -> torch.Tensor:
        """Get pairwise distance `dr`.

        Parameters
        ----------
        coords : torch.Tensor
            The coordinate of the atoms shape of (nframes * nall * 3).

        Returns
        -------
        torch.Tensor
            The pairwise distance between the atoms (nframes * nall * nall * 3).

        Examples
        --------
        coords = torch.tensor([[
                [0,0,0],
                [1,3,5],
                [2,4,6]
            ]])

        dist = tensor([[
            [[ 0,  0,  0],
            [-1, -3, -5],
            [-2, -4, -6]],

            [[ 1,  3,  5],
            [ 0,  0,  0],
            [-1, -1, -1]],

            [[ 2,  4,  6],
            [ 1,  1,  1],
            [ 0,  0,  0]]
            ]])
        """
        return coords.unsqueeze(2) - coords.unsqueeze(1)

    def _extract_spline_coefficient(
        self, i_type: torch.Tensor, j_type: torch.Tensor, idx: torch.Tensor
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

        Returns
        -------
        torch.Tensor
            The spline coefficient. (nframes, nloc, nnei, 4)

        """
        # (nframes, nloc, nnei)
        expanded_i_type = i_type.unsqueeze(-1).expand(-1, -1, j_type.shape[-1])

        # (nframes, nloc, nnei, nspline, 4)
        expanded_tab_data = self.tab_data[expanded_i_type, j_type]

        # (nframes, nloc, nnei, 1, 4)
        expanded_idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, 4)

        # handle the case where idx is beyond the number of splines
        clipped_indices = torch.clamp(expanded_idx, 0, self.nspline - 1).to(torch.int64)

        # (nframes, nloc, nnei, 4)
        final_coef = torch.gather(expanded_tab_data, 3, clipped_indices).squeeze()

        # when the spline idx is beyond the table, all spline coefficients are set to `0`, and the resulting ener corresponding to the idx is also `0`.
        final_coef[expanded_idx.squeeze() >= self.nspline] = 0

        return final_coef
