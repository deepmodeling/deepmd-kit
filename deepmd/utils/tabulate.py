# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    ABC,
    abstractmethod,
)
from functools import (
    lru_cache,
)

import numpy as np
from scipy.special import (
    comb,
)

log = logging.getLogger(__name__)


class BaseTabulate(ABC):
    """A base class for pt and tf tabulation."""

    def __init__(
        self,
        descrpt,
        neuron,
        type_one_side,
        exclude_types,
        is_pt,
    ) -> None:
        """Constructor."""
        super().__init__()

        """Shared attributes."""
        self.descrpt = descrpt
        self.neuron = neuron
        self.type_one_side = type_one_side
        self.exclude_types = exclude_types
        self.is_pt = is_pt

        """Need to be initialized in the subclass."""
        self.descrpt_type = "Base"

        self.sel_a = []
        self.rcut = 0.0
        self.rcut_smth = 0.0

        self.davg = np.array([])
        self.dstd = np.array([])
        self.ntypes = 0

        self.layer_size = 0
        self.table_size = 0

        self.bias = {}
        self.matrix = {}

        self.data_type = None
        self.last_layer_size = 0

        """Save the tabulation result."""
        self.data = {}

        self.upper = {}
        self.lower = {}

    def build(
        self, min_nbor_dist: float, extrapolate: float, stride0: float, stride1: float
    ) -> tuple[dict[str, int], dict[str, int]]:
        r"""Build the tables for model compression.

        Parameters
        ----------
        min_nbor_dist
            The nearest distance between neighbor atoms
        extrapolate
            The scale of model extrapolation
        stride0
            The uniform stride of the first table
        stride1
            The uniform stride of the second table

        Returns
        -------
        lower : dict[str, int]
            The lower boundary of environment matrix by net
        upper : dict[str, int]
            The upper boundary of environment matrix by net
        """
        # tabulate range [lower, upper] with stride0 'stride0'
        lower, upper = self._get_env_mat_range(min_nbor_dist)
        if self.descrpt_type in ("Atten", "AEbdV2"):
            uu = np.max(upper)
            ll = np.min(lower)
            xx = np.arange(ll, uu, stride0, dtype=self.data_type)
            xx = np.append(
                xx,
                np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type),
            )
            xx = np.append(xx, np.array([extrapolate * uu], dtype=self.data_type))
            nspline = ((uu - ll) / stride0 + (extrapolate * uu - uu) / stride1).astype(
                int
            )
            self._build_lower(
                "filter_net", xx, 0, uu, ll, stride0, stride1, extrapolate, nspline
            )
        elif self.descrpt_type == "A":
            for ii in range(self.table_size):
                if (self.type_one_side and not self._all_excluded(ii)) or (
                    not self.type_one_side
                    and (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types
                ):
                    if self.type_one_side:
                        net = "filter_-1_net_" + str(ii)
                        # upper and lower should consider all types which are not excluded and sel>0
                        idx = [
                            (type_i, ii) not in self.exclude_types
                            and self.sel_a[type_i] > 0
                            for type_i in range(self.ntypes)
                        ]
                        uu = np.max(upper[idx])
                        ll = np.min(lower[idx])
                    else:
                        ielement = ii // self.ntypes
                        net = (
                            "filter_" + str(ielement) + "_net_" + str(ii % self.ntypes)
                        )
                        if self.is_pt:
                            uu = np.max(upper[ielement])
                            ll = np.min(lower[ielement])
                        else:
                            uu = upper[ielement]
                            ll = lower[ielement]
                    xx = np.arange(ll, uu, stride0, dtype=self.data_type)
                    xx = np.append(
                        xx,
                        np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type),
                    )
                    xx = np.append(
                        xx, np.array([extrapolate * uu], dtype=self.data_type)
                    )
                    nspline = (
                        (uu - ll) / stride0 + (extrapolate * uu - uu) / stride1
                    ).astype(int)
                    self._build_lower(
                        net, xx, ii, uu, ll, stride0, stride1, extrapolate, nspline
                    )
        elif self.descrpt_type == "T":
            xx_all = []
            for ii in range(self.ntypes):
                """Pt and tf is different here. Pt version is a two-dimensional array."""
                if self.is_pt:
                    uu = np.max(upper[ii])
                    ll = np.min(lower[ii])
                else:
                    ll = lower[ii]
                    uu = upper[ii]
                xx = np.arange(extrapolate * ll, ll, stride1, dtype=self.data_type)
                xx = np.append(xx, np.arange(ll, uu, stride0, dtype=self.data_type))
                xx = np.append(
                    xx,
                    np.arange(
                        uu,
                        extrapolate * uu,
                        stride1,
                        dtype=self.data_type,
                    ),
                )
                xx = np.append(xx, np.array([extrapolate * uu], dtype=self.data_type))
                xx_all.append(xx)
            nspline = (
                (upper - lower) / stride0
                + 2 * ((extrapolate * upper - upper) / stride1)
            ).astype(int)
            idx = 0
            for ii in range(self.ntypes):
                if self.is_pt:
                    uu = np.max(upper[ii])
                    ll = np.min(lower[ii])
                else:
                    ll = lower[ii]
                    uu = upper[ii]
                for jj in range(ii, self.ntypes):
                    net = "filter_" + str(ii) + "_net_" + str(jj)
                    self._build_lower(
                        net,
                        xx_all[ii],
                        idx,
                        uu,
                        ll,
                        stride0,
                        stride1,
                        extrapolate,
                        nspline[ii][0] if self.is_pt else nspline[ii],
                    )
                    idx += 1
        elif self.descrpt_type == "R":
            for ii in range(self.table_size):
                if (self.type_one_side and not self._all_excluded(ii)) or (
                    not self.type_one_side
                    and (ii // self.ntypes, ii % self.ntypes) not in self.exclude_types
                ):
                    if self.type_one_side:
                        net = "filter_-1_net_" + str(ii)
                        # upper and lower should consider all types which are not excluded and sel>0
                        idx = [
                            (type_i, ii) not in self.exclude_types
                            and self.sel_a[type_i] > 0
                            for type_i in range(self.ntypes)
                        ]
                        uu = np.max(upper[idx])
                        ll = np.min(lower[idx])
                    else:
                        ielement = ii // self.ntypes
                        net = (
                            "filter_" + str(ielement) + "_net_" + str(ii % self.ntypes)
                        )
                        uu = upper[ielement]
                        ll = lower[ielement]
                    xx = np.arange(ll, uu, stride0, dtype=self.data_type)
                    xx = np.append(
                        xx,
                        np.arange(uu, extrapolate * uu, stride1, dtype=self.data_type),
                    )
                    xx = np.append(
                        xx, np.array([extrapolate * uu], dtype=self.data_type)
                    )
                    nspline = (
                        (uu - ll) / stride0 + (extrapolate * uu - uu) / stride1
                    ).astype(int)
                    self._build_lower(
                        net, xx, ii, uu, ll, stride0, stride1, extrapolate, nspline
                    )
        else:
            raise RuntimeError("Unsupported descriptor")

        self._convert_numpy_to_tensor()
        if self.is_pt:
            self._convert_numpy_float_to_int()
        return self.lower, self.upper

    def _build_lower(
        self, net, xx, idx, upper, lower, stride0, stride1, extrapolate, nspline
    ) -> None:
        vv, dd, d2 = self._make_data(xx, idx)
        self.data[net] = np.zeros(
            [nspline, 6 * self.last_layer_size], dtype=self.data_type
        )

        # tt.shape: [nspline, self.last_layer_size]
        if self.descrpt_type in ("Atten", "A", "AEbdV2"):
            tt = np.full((nspline, self.last_layer_size), stride1)  # pylint: disable=no-explicit-dtype
            tt[: int((upper - lower) / stride0), :] = stride0
        elif self.descrpt_type == "T":
            tt = np.full((nspline, self.last_layer_size), stride1)  # pylint: disable=no-explicit-dtype
            tt[
                int((lower - extrapolate * lower) / stride1) + 1 : (
                    int((lower - extrapolate * lower) / stride1)
                    + int((upper - lower) / stride0)
                ),
                :,
            ] = stride0
        elif self.descrpt_type == "R":
            tt = np.full((nspline, self.last_layer_size), stride1)  # pylint: disable=no-explicit-dtype
            tt[: int((upper - lower) / stride0), :] = stride0
        else:
            raise RuntimeError("Unsupported descriptor")

        # hh.shape: [nspline, self.last_layer_size]
        hh = (
            vv[1 : nspline + 1, : self.last_layer_size]
            - vv[:nspline, : self.last_layer_size]
        )

        self.data[net][:, : 6 * self.last_layer_size : 6] = vv[
            :nspline, : self.last_layer_size
        ]
        self.data[net][:, 1 : 6 * self.last_layer_size : 6] = dd[
            :nspline, : self.last_layer_size
        ]
        self.data[net][:, 2 : 6 * self.last_layer_size : 6] = (
            0.5 * d2[:nspline, : self.last_layer_size]
        )
        self.data[net][:, 3 : 6 * self.last_layer_size : 6] = (
            1 / (2 * tt * tt * tt)
        ) * (
            20 * hh
            - (
                8 * dd[1 : nspline + 1, : self.last_layer_size]
                + 12 * dd[:nspline, : self.last_layer_size]
            )
            * tt
            - (
                3 * d2[:nspline, : self.last_layer_size]
                - d2[1 : nspline + 1, : self.last_layer_size]
            )
            * tt
            * tt
        )
        self.data[net][:, 4 : 6 * self.last_layer_size : 6] = (
            1 / (2 * tt * tt * tt * tt)
        ) * (
            -30 * hh
            + (
                14 * dd[1 : nspline + 1, : self.last_layer_size]
                + 16 * dd[:nspline, : self.last_layer_size]
            )
            * tt
            + (
                3 * d2[:nspline, : self.last_layer_size]
                - 2 * d2[1 : nspline + 1, : self.last_layer_size]
            )
            * tt
            * tt
        )
        self.data[net][:, 5 : 6 * self.last_layer_size : 6] = (
            1 / (2 * tt * tt * tt * tt * tt)
        ) * (
            12 * hh
            - 6
            * (
                dd[1 : nspline + 1, : self.last_layer_size]
                + dd[:nspline, : self.last_layer_size]
            )
            * tt
            + (
                d2[1 : nspline + 1, : self.last_layer_size]
                - d2[:nspline, : self.last_layer_size]
            )
            * tt
            * tt
        )

        self.upper[net] = upper
        self.lower[net] = lower

    @abstractmethod
    def _make_data(self, xx, idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate tabulation data for the given input.

        Parameters
        ----------
        xx : np.ndarray
            Input values to tabulate
        idx : int
            Index for accessing the correct network parameters

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Values, first derivatives, and second derivatives
        """
        pass

    @lru_cache
    def _all_excluded(self, ii: int) -> bool:
        """Check if type ii excluds all types.

        Parameters
        ----------
        ii : int
            type index

        Returns
        -------
        bool
            if type ii excluds all types
        """
        return all((ii, type_i) in self.exclude_types for type_i in range(self.ntypes))

    @abstractmethod
    def _get_descrpt_type(self):
        """Get the descrpt type."""
        pass

    @abstractmethod
    def _get_layer_size(self):
        """Get the number of embedding layer."""
        pass

    def _get_table_size(self):
        table_size = 0
        if self.descrpt_type in ("Atten", "AEbdV2"):
            table_size = 1
        elif self.descrpt_type == "A":
            table_size = self.ntypes * self.ntypes
            if self.type_one_side:
                table_size = self.ntypes
        elif self.descrpt_type == "T":
            table_size = int(comb(self.ntypes + 1, 2))
        elif self.descrpt_type == "R":
            table_size = self.ntypes * self.ntypes
            if self.type_one_side:
                table_size = self.ntypes
        else:
            raise RuntimeError("Unsupported descriptor")
        return table_size

    def _get_data_type(self):
        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                return type(item[0][0])
        return None

    def _get_last_layer_size(self):
        for item in self.matrix["layer_" + str(self.layer_size)]:
            if len(item) != 0:
                return item.shape[1]
        return 0

    @abstractmethod
    def _get_bias(self):
        """Get bias of embedding net."""
        pass

    @abstractmethod
    def _get_matrix(self):
        """Get weight matrx of embedding net."""
        pass

    @abstractmethod
    def _convert_numpy_to_tensor(self):
        """Convert self.data from np.ndarray to torch.Tensor."""
        pass

    def _convert_numpy_float_to_int(self) -> None:
        """Convert self.lower and self.upper from np.float32 or np.float64 to int."""
        self.lower = {k: int(v) for k, v in self.lower.items()}
        self.upper = {k: int(v) for k, v in self.upper.items()}

    def _get_env_mat_range(self, min_nbor_dist):
        """Change the embedding net range to sw / min_nbor_dist."""
        sw = self._spline5_switch(min_nbor_dist, self.rcut_smth, self.rcut)
        if self.descrpt_type in ("Atten", "A", "AEbdV2"):
            lower = -self.davg[:, 0] / self.dstd[:, 0]
            upper = ((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0]
        elif self.descrpt_type == "T":
            var = np.square(sw / (min_nbor_dist * self.dstd[:, 1:4]))
            lower = np.min(-var, axis=1)
            upper = np.max(var, axis=1)
        elif self.descrpt_type == "R":
            lower = -self.davg[:, 0] / self.dstd[:, 0]
            upper = ((1 / min_nbor_dist) * sw - self.davg[:, 0]) / self.dstd[:, 0]
        else:
            raise RuntimeError("Unsupported descriptor")
        log.info("training data with lower boundary: " + str(lower))
        log.info("training data with upper boundary: " + str(upper))
        # returns element-wise lower and upper
        return np.floor(lower), np.ceil(upper)

    def _spline5_switch(self, xx, rmin, rmax):
        if xx < rmin:
            vv = 1
        elif xx < rmax:
            uu = (xx - rmin) / (rmax - rmin)
            vv = uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1
        else:
            vv = 0
        return vv
