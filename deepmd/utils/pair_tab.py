#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Optional,
    Tuple,
)

import numpy as np
from scipy.interpolate import (
    CubicSpline,
)

from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)


class PairTab:
    """Pairwise tabulated potential.

    Parameters
    ----------
    filename
            File name for the short-range tabulated potential.
            The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes.
            The first colume is the distance between atoms.
            The second to the last columes are energies for pairs of certain types.
            For example we have two atom types, 0 and 1.
            The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
    """

    def __init__(self, filename: str, rcut: Optional[float] = None) -> None:
        """Constructor."""
        self.reinit(filename, rcut)

    def reinit(self, filename: str, rcut: Optional[float] = None) -> None:
        """Initialize the tabulated interaction.

        Parameters
        ----------
        filename
            File name for the short-range tabulated potential.
            The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes.
            The first colume is the distance between atoms.
            The second to the last columes are energies for pairs of certain types.
            For example we have two atom types, 0 and 1.
            The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
        """
        if filename is None:
            self.tab_info, self.tab_data = None, None
            return
        self.vdata = np.loadtxt(filename)
        self.rmin = self.vdata[0][0]
        self.rmax = self.vdata[-1][0]
        self.hh = self.vdata[1][0] - self.vdata[0][0]
        ncol = self.vdata.shape[1] - 1
        n0 = (-1 + np.sqrt(1 + 8 * ncol)) * 0.5
        self.ntypes = int(n0 + 0.1)
        assert self.ntypes * (self.ntypes + 1) // 2 == ncol, (
            "number of volumes provided in %s does not match guessed number of types %d"
            % (filename, self.ntypes)
        )

        # check table data against rcut and update tab_file if needed, table upper boundary is used as rcut if not provided.
        self.rcut = rcut if rcut is not None else self.rmax
        self._check_table_upper_boundary()
        self.nspline = (
            self.vdata.shape[0] - 1
        )  # this nspline is updated based on the expanded table.
        self.tab_info = np.array([self.rmin, self.hh, self.nspline, self.ntypes])
        self.tab_data = self._make_data()

    def serialize(self) -> dict:
        return {
            "@class": "PairTab",
            "@version": 1,
            "rmin": self.rmin,
            "rmax": self.rmax,
            "hh": self.hh,
            "ntypes": self.ntypes,
            "rcut": self.rcut,
            "nspline": self.nspline,
            "@variables": {
                "vdata": self.vdata,
                "tab_info": self.tab_info,
                "tab_data": self.tab_data,
            },
        }

    @classmethod
    def deserialize(cls, data) -> "PairTab":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class")
        variables = data.pop("@variables")
        tab = PairTab(None, None)
        tab.vdata = variables["vdata"]
        tab.rmin = data["rmin"]
        tab.rmax = data["rmax"]
        tab.hh = data["hh"]
        tab.ntypes = data["ntypes"]
        tab.rcut = data["rcut"]
        tab.nspline = data["nspline"]
        tab.tab_info = variables["tab_info"]
        tab.tab_data = variables["tab_data"]
        return tab

    def _check_table_upper_boundary(self) -> None:
        """Update User Provided Table Based on `rcut`.

        This function checks the upper boundary provided in the table against rcut.
        If the table upper boundary values decay to zero before rcut, padding zeros will
        be added to the table to cover rcut; if the table upper boundary values do not decay to zero
        before ruct, extrapolation will be performed till rcut.

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
        upper_val = self.vdata[-1][1:]
        upper_idx = self.vdata.shape[0] - 1
        self.ncol = self.vdata.shape[1]

        # the index in table for the grid point of rcut, always give the point after rcut.
        rcut_idx = int(np.ceil(self.rcut / self.hh - self.rmin / self.hh))
        if np.all(upper_val == 0):
            # if table values decay to `0` after rcut
            if self.rcut < self.rmax and np.any(self.vdata[rcut_idx - 1][1:] != 0):
                log.warning(
                    "The energy provided in the table does not decay to 0 at rcut."
                )
            # if table values decay to `0` at rcut, do nothing

            # if table values decay to `0` before rcut, pad table with `0`s.
            elif self.rcut > self.rmax:
                pad_zero = np.zeros((rcut_idx - upper_idx, self.ncol))
                pad_zero[:, 0] = np.linspace(
                    self.rmax + self.hh,
                    self.rmax + self.hh * (rcut_idx - upper_idx),
                    rcut_idx - upper_idx,
                )
                self.vdata = np.concatenate((self.vdata, pad_zero), axis=0)
        else:
            # if table values do not decay to `0` at rcut
            if self.rcut <= self.rmax:
                log.warning(
                    "The energy provided in the table does not decay to 0 at rcut."
                )
            # if rcut goes beyond table upper bond, need extrapolation, ensure values decay to `0` before rcut.
            else:
                log.warning(
                    "The rcut goes beyond table upper boundary, performing extrapolation."
                )
                pad_extrapolation = np.zeros((rcut_idx - upper_idx, self.ncol))

                pad_extrapolation[:, 0] = np.linspace(
                    self.rmax + self.hh,
                    self.rmax + self.hh * (rcut_idx - upper_idx),
                    rcut_idx - upper_idx,
                )
                # need to calculate table values to fill in with cubic spline
                pad_extrapolation = self._extrapolate_table(pad_extrapolation)

                self.vdata = np.concatenate((self.vdata, pad_extrapolation), axis=0)

    def get(self) -> Tuple[np.array, np.array]:
        """Get the serialized table."""
        return self.tab_info, self.tab_data

    def _extrapolate_table(self, pad_extrapolation: np.array) -> np.array:
        """Soomth extrapolation between table upper boundary and rcut.

        This method should only be used when the table upper boundary `rmax` is smaller than `rcut`, and
        the table upper boundary values are not zeros. To simplify the problem, we use a single
        cubic spline between `rmax` and `rcut` for each pair of atom types. One can substitute this extrapolation
        to higher order polynomials if needed.

        There are two scenarios:
            1. `ruct` - `rmax` >= hh:
                Set values at the grid point right before `rcut` to 0, and perform exterapolation between
                the grid point and `rmax`, this allows smooth decay to 0 at `rcut`.
            2. `rcut` - `rmax` < hh:
                Set values at `rmax + hh` to 0, and perform extrapolation between `rmax` and `rmax + hh`.

        Parameters
        ----------
        pad_extrapolation : np.array
            The emepty grid that holds the extrapolation values.

        Returns
        -------
        np.array
            The cubic spline extrapolation.
        """
        # in theory we should check if the table has at least two rows.
        slope = self.vdata[-1, 1:] - self.vdata[-2, 1:]  # shape of (ncol-1, )

        # for extrapolation, we want values decay to `0` prior to `ruct` if possible
        # here we try to find the grid point prior to `rcut`
        grid_point = (
            -2 if pad_extrapolation[-1, 0] / self.hh - self.rmax / self.hh >= 2 else -1
        )
        temp_grid = np.stack((self.vdata[-1, :], pad_extrapolation[grid_point, :]))
        vv = temp_grid[:, 1:]
        xx = temp_grid[:, 0]
        cs = CubicSpline(xx, vv, bc_type=((1, slope), (1, np.zeros_like(slope))))
        xx_grid = pad_extrapolation[:, 0]
        res = cs(xx_grid)

        pad_extrapolation[:, 1:] = res

        # Note: when doing cubic spline, if we want to ensure values decay to zero prior to `rcut`
        # this may cause values be positive post `rcut`, we need to overwrite those values to zero
        pad_extrapolation = (
            pad_extrapolation if grid_point == -1 else pad_extrapolation[:-1, :]
        )
        return pad_extrapolation

    def _make_data(self):
        data = np.zeros([self.ntypes * self.ntypes * 4 * self.nspline])
        stride = 4 * self.nspline
        idx_iter = 0
        xx = self.vdata[:, 0]
        for t0 in range(self.ntypes):
            for t1 in range(t0, self.ntypes):
                vv = self.vdata[:, 1 + idx_iter]
                cs = CubicSpline(xx, vv, bc_type="clamped")
                dd = cs(xx, 1)
                dd *= self.hh
                dtmp = np.zeros(stride)
                for ii in range(self.nspline):
                    dtmp[ii * 4 + 0] = 2 * vv[ii] - 2 * vv[ii + 1] + dd[ii] + dd[ii + 1]
                    dtmp[ii * 4 + 1] = (
                        -3 * vv[ii] + 3 * vv[ii + 1] - 2 * dd[ii] - dd[ii + 1]
                    )
                    dtmp[ii * 4 + 2] = dd[ii]
                    dtmp[ii * 4 + 3] = vv[ii]
                data[
                    (t0 * self.ntypes + t1) * stride : (t0 * self.ntypes + t1) * stride
                    + stride
                ] = dtmp
                data[
                    (t1 * self.ntypes + t0) * stride : (t1 * self.ntypes + t0) * stride
                    + stride
                ] = dtmp
                idx_iter += 1
        return data
