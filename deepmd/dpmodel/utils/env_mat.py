# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel import (
    NativeOP,
)
from deepmd.dpmodel.array_api import (
    support_array_api,
)


@support_array_api(version="2022.12")
def compute_smooth_weight(
    distance: np.ndarray,
    rmin: float,
    rmax: float,
):
    """Compute smooth weight for descriptor elements."""
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    xp = array_api_compat.array_namespace(distance)
    min_mask = distance <= rmin
    max_mask = distance >= rmax
    mid_mask = xp.logical_not(xp.logical_or(min_mask, max_mask))
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu * uu * uu * (-6.0 * uu * uu + 15.0 * uu - 10.0) + 1.0
    return vv * xp.astype(mid_mask, distance.dtype) + xp.astype(
        min_mask, distance.dtype
    )


def _make_env_mat(
    nlist,
    coord,
    rcut: float,
    ruct_smth: float,
    radial_only: bool = False,
    protection: float = 0.0,
):
    """Make smooth environment matrix."""
    xp = array_api_compat.array_namespace(nlist)
    nf, nloc, nnei = nlist.shape
    # nf x nall x 3
    coord = xp.reshape(coord, (nf, -1, 3))
    mask = nlist >= 0
    nlist = nlist * xp.astype(mask, nlist.dtype)
    # nf x (nloc x nnei) x 3
    # index = xp.reshape(nlist, (nf, -1, 1))
    # index = xp.tile(xp.reshape(nlist, (nf, -1, 1)), (1, 1, 3))
    # coord_r = xp.take_along_axis(coord,  xp.tile(index, (1, 1, 3)), 1)
    # note: array api doesn't contain take_along_axis until the next version
    # reimplement
    nall = coord.shape[1]
    index = xp.reshape(nlist, (nf * nloc * nnei,)) + xp.repeat(
        (xp.arange(nf) * nall), nloc * nnei
    )
    coord_ = xp.reshape(coord, (-1, 3))
    coord_r = xp.take(coord_, index, axis=0)
    # nf x nloc x nnei x 3
    coord_r = xp.reshape(coord_r, (nf, nloc, nnei, 3))
    # nf x nloc x 1 x 3
    coord_l = xp.reshape(coord[:, :nloc, ...], (nf, -1, 1, 3))
    # nf x nloc x nnei x 3
    diff = coord_r - coord_l
    # nf x nloc x nnei
    length = xp.linalg.vector_norm(diff, axis=-1, keepdims=True)
    # for index 0 nloc atom
    length = length + xp.astype(~xp.expand_dims(mask, axis=-1), length.dtype)
    t0 = 1 / (length + protection)
    t1 = diff / (length + protection) ** 2
    weight = compute_smooth_weight(length, ruct_smth, rcut)
    weight = weight * xp.astype(xp.expand_dims(mask, axis=-1), weight.dtype)
    if radial_only:
        env_mat = t0 * weight
    else:
        env_mat = xp.concat([t0, t1], axis=-1) * weight
    return env_mat, diff * xp.astype(xp.expand_dims(mask, axis=-1), diff.dtype), weight


class EnvMat(NativeOP):
    def __init__(
        self,
        rcut,
        rcut_smth,
        protection: float = 0.0,
    ):
        self.rcut = rcut
        self.rcut_smth = rcut_smth
        self.protection = protection

    def call(
        self,
        coord_ext: np.ndarray,
        atype_ext: np.ndarray,
        nlist: np.ndarray,
        davg: Optional[np.ndarray] = None,
        dstd: Optional[np.ndarray] = None,
        radial_only: bool = False,
    ) -> Union[np.ndarray, np.ndarray]:
        """Compute the environment matrix.

        Parameters
        ----------
        nlist
            The neighbor list. shape: nf x nloc x nnei
        coord_ext
            The extended coordinates of atoms. shape: nf x (nallx3)
        atype_ext
            The extended aotm types. shape: nf x nall
        davg
            The data avg. shape: nt x nnei x (4 or 1)
        dstd
            The inverse of data std. shape: nt x nnei x (4 or 1)
        radial_only
            Whether to only compute radial part of the environment matrix.
            If True, the output will be of shape nf x nloc x nnei x 1.
            Otherwise, the output will be of shape nf x nloc x nnei x 4.
            Default: False.

        Returns
        -------
        env_mat
            The environment matrix. shape: nf x nloc x nnei x (4 or 1)
        diff
            The relative coordinate of neighbors. shape: nf x nloc x nnei x 3
        switch
            The value of switch function. shape: nf x nloc x nnei
        """
        xp = array_api_compat.array_namespace(coord_ext, atype_ext, nlist)
        em, diff, sw = self._call(nlist, coord_ext, radial_only)
        nf, nloc, nnei = nlist.shape
        atype = atype_ext[:, :nloc]
        if davg is not None:
            em -= xp.reshape(xp.take(davg, xp.reshape(atype, (-1,)), axis=0), em.shape)
        if dstd is not None:
            em /= xp.reshape(xp.take(dstd, xp.reshape(atype, (-1,)), axis=0), em.shape)
        return em, diff, sw

    def _call(self, nlist, coord_ext, radial_only):
        em, diff, ww = _make_env_mat(
            nlist,
            coord_ext,
            self.rcut,
            self.rcut_smth,
            radial_only=radial_only,
            protection=self.protection,
        )
        return em, diff, ww

    def serialize(
        self,
    ) -> dict:
        return {
            "rcut": self.rcut,
            "rcut_smth": self.rcut_smth,
        }

    @classmethod
    def deserialize(
        cls,
        data: dict,
    ) -> "EnvMat":
        return cls(**data)
