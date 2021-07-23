#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List

from scipy.interpolate import CubicSpline

class PairTab (object):
    def __init__(self,
                 filename : str
    ) -> None:
        """
        Constructor

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
        self.reinit(filename)
        
    def reinit(self,
               filename : str
    ) -> None:
        """
        Initialize the tabulated interaction

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
        self.hh = self.vdata[1][0] - self.vdata[0][0]
        self.nspline = self.vdata.shape[0] - 1
        ncol = self.vdata.shape[1] - 1
        n0 = (-1 + np.sqrt(1 + 8 * ncol)) * 0.5
        self.ntypes = int(n0 + 0.1)
        assert(self.ntypes * (self.ntypes+1) // 2 == ncol),\
            "number of volumes provided in %s does not match guessed number of types %d" % (filename, self.ntypes)
        self.tab_info = np.array([self.rmin, self.hh, self.nspline, self.ntypes])
        self.tab_data = self._make_data()

    def get(self) -> Tuple[np.array, np.array]:
        """
        Get the serialized table. 
        """
        return self.tab_info, self.tab_data

    def _make_data(self) :
        data = np.zeros([self.ntypes * self.ntypes * 4 * self.nspline])
        stride = 4 * self.nspline
        idx_iter = 0
        xx = self.vdata[:,0]
        for t0 in range(self.ntypes) :
            for t1 in range(t0, self.ntypes) :
                vv = self.vdata[:,1+idx_iter]
                cs = CubicSpline(xx, vv)
                dd = cs(xx, 1)
                dd *= self.hh
                dtmp = np.zeros(stride)
                for ii in range(self.nspline) :
                    dtmp[ii*4+0] = 2 * vv[ii] - 2 * vv[ii+1] +     dd[ii] + dd[ii+1]
                    dtmp[ii*4+1] =-3 * vv[ii] + 3 * vv[ii+1] - 2 * dd[ii] - dd[ii+1]
                    dtmp[ii*4+2] = dd[ii]
                    dtmp[ii*4+3] = vv[ii]
                data[(t0 * self.ntypes + t1) * stride : (t0 * self.ntypes + t1) * stride + stride] \
                    = dtmp
                data[(t1 * self.ntypes + t0) * stride : (t1 * self.ntypes + t0) * stride + stride] \
                    = dtmp
                idx_iter += 1
        return data
