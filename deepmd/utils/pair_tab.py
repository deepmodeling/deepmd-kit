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

    def _check_table_upper_boundary(self) -> None:
        """Update User Provided Table Based on `rcut`.

        This function checks the upper boundary provided in the table against rcut.
        If the table upper boundary values decay to zero before rcut, padding zeros will
        be added to the table to cover rcut; if the table upper boundary values do not decay to zero
        before ruct, linear extrapolation will be performed till rcut.

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
        upper_val = self.vdata[-1][1:]
        upper_idx = self.vdata.shape[0] - 1
        ncol = self.vdata.shape[1]
        # the index of table for the grid point right after rcut
        rcut_idx = int(self.rcut / self.hh)

        if np.all(upper_val == 0):
            # if table values decay to `0` after rcut
            if self.rcut < self.rmax and np.any(self.vdata[rcut_idx - 1][1:] != 0):
                logging.warning(
                    "The energy provided in the table does not decay to 0 at rcut."
                )
            # if table values decay to `0` at rcut, do nothing

            # if table values decay to `0` before rcut, pad table with `0`s.
            elif self.rcut > self.rmax:
                pad_zero = np.zeros((rcut_idx - upper_idx, ncol))
                pad_zero[:, 0] = np.linspace(
                    self.rmax + self.hh, self.hh * (rcut_idx + 1), rcut_idx - upper_idx
                )
                self.vdata = np.concatenate((self.vdata, pad_zero), axis=0)
        else:
            # if table values do not decay to `0` at rcut
            if self.rcut < self.rmax:
                logging.warning(
                    "The energy provided in the table does not decay to 0 at rcut."
                )
            # if rcut goes beyond table upper bond, need extrapolation, ensure values decay to `0` before rcut.
            else:
                logging.warning(
                    "The rcut goes beyond table upper boundary, performing extrapolation."
                )
                pad_linear = np.zeros((rcut_idx - upper_idx + 1, ncol))
                pad_linear[:, 0] = np.linspace(
                    self.rmax, self.hh * (rcut_idx + 1), rcut_idx - upper_idx + 1
                )
                pad_linear[:-1, 1:] = np.array(
                    [np.linspace(start, 0, rcut_idx - upper_idx) for start in upper_val]
                ).T
                self.vdata = np.concatenate((self.vdata[:-1, :], pad_linear), axis=0)

    def get(self) -> Tuple[np.array, np.array]:
        """Get the serialized table."""
        return self.tab_info, self.tab_data

    def _make_data(self):
        data = np.zeros([self.ntypes * self.ntypes * 4 * self.nspline])
        stride = 4 * self.nspline
        idx_iter = 0
        xx = self.vdata[:, 0]
        for t0 in range(self.ntypes):
            for t1 in range(t0, self.ntypes):
                vv = self.vdata[:, 1 + idx_iter]
                cs = CubicSpline(xx, vv)
                dd = cs(xx, 1)
                dd *= self.hh
                dtmp = np.zeros(stride)
                for ii in range(self.nspline):
                    # check if vv is zero, if so, that's case 1, set all coefficients to 0,
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
