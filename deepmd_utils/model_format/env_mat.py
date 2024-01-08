# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from .common import (
    NativeOP,
)


def compute_smooth_weight(
    distance: np.ndarray,
    rmin: float,
    rmax: float,
):
    """Compute smooth weight for descriptor elements."""
    min_mask = distance <= rmin
    max_mask = distance >= rmax
    mid_mask = np.logical_not(np.logical_or(min_mask, max_mask))
    uu = (distance - rmin) / (rmax - rmin)
    vv = uu * uu * uu * (-6.0 * uu * uu + 15.0 * uu - 10.0) + 1.0
    return vv * mid_mask + min_mask


def _make_env_mat(
    nlist,
    coord,
    rcut: float,
    ruct_smth: float,
):
    """Make smooth environment matrix."""
    nf, nloc, nnei = nlist.shape
    # nf x nall x 3
    coord = coord.reshape(nf, -1, 3)
    mask = nlist >= 0
    nlist = nlist * mask
    # nf x (nloc x nnei) x 3
    index = np.tile(nlist.reshape(nf, -1, 1), (1, 1, 3))
    coord_r = np.take_along_axis(coord, index, 1)
    # nf x nloc x nnei x 3
    coord_r = coord_r.reshape(nf, nloc, nnei, 3)
    # nf x nloc x 1 x 3
    coord_l = coord[:, :nloc].reshape(nf, -1, 1, 3)
    # nf x nloc x nnei x 3
    diff = coord_r - coord_l
    # nf x nloc x nnei
    length = np.linalg.norm(diff, axis=-1, keepdims=True)
    # for index 0 nloc atom
    length = length + ~np.expand_dims(mask, -1)
    t0 = 1 / length
    t1 = diff / length**2
    weight = compute_smooth_weight(length, ruct_smth, rcut)
    env_mat_se_a = np.concatenate([t0, t1], axis=-1) * weight * np.expand_dims(mask, -1)
    return env_mat_se_a, diff * np.expand_dims(mask, -1), weight


class EnvMat(NativeOP):
    def __init__(
        self,
        rcut,
        rcut_smth,
    ):
        self.rcut = rcut
        self.rcut_smth = rcut_smth

    def call(
        self,
        nlist,
        coord_ext,
    ):
        em, diff, ww = _make_env_mat(nlist, coord_ext, self.rcut, self.rcut_smth)
        return em, ww

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
