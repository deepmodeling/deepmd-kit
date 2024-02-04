# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import numpy as np

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.dpmodel.utils.nlist import (
    build_multiple_neighbor_list,
)

from .base_atomic_model import (
    BaseAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)
from .pair_tab_model import (
    PairTabModel,
)


class LinearModel(BaseAtomicModel):
    """Model linearly combine a list of AtomicModels.

    Parameters
    ----------
    models
            This linear model should take a DPAtomicModel and a PairTable model.
    """

    def __init__(
        self,
        dp_models: DPAtomicModel,
        zbl_model: PairTabModel,
        **kwargs,
    ):
        super().__init__()
        self.dp_model = dp_models
        self.zbl_model = zbl_model
        self.rcut = self.get_rcut()
        self.sel = self.get_sel()

    def get_fitting_output_def(self) -> FittingOutputDef:
        """Get the output def of the fitting net."""
        return (
            self.fitting_net.output_def()
            if self.fitting_net is not None
            else self.coord_denoise_net.output_def()
        )

    def get_rcut(self) -> float:
        """Get the cut-off radius."""
        return self.get_rcuts()[-1]

    def get_rcuts(self) -> float:
        """Get the cut-off radius for each individual models in ascending order."""
        return sorted([self.zbl_model.get_rcut(), self.dp_model.get_rcut()])

    def get_sel(self) -> int:
        """Get the neighbor selection."""
        return self.get_sels()[-1]

    def get_sels(self) -> int:
        """Get the neighbor selection for each individual models in ascending order."""
        return sorted([self.zbl_model.get_sel(), sum(self.dp_model.get_sel())])

    def distinguish_types(self) -> bool:
        """If distinguish different types by sorting."""
        return self.dp_model.distinguish_types() and self.zbl_model.distinguish_types()

    def forward_atomic(
        self,
        extended_coord,
        extended_atype,
        nlist,
        ra: float,
        rb: float,
        alpha: Optional[float] = 0.1,
        mapping: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Return atomic prediction.

        Note: currently only support the linear combination of a ZBL model and a DP model,
        the weight is calculated based on this paper:
        Appl. Phys. Lett. 114, 244101 (2019); https://doi.org/10.1063/1.5098061

        Parameters
        ----------
        extended_coord
            coodinates in extended region, (nframes, nall * 3)
        extended_atype
            atomic type in extended region, (nframes, nall)
        nlist
            neighbor list, (nframes, nloc, nsel).
        mapping
            mapps the extended indices to local indices
        ra : float
            inclusive lower boundary of the range in which the ZBL potential and the deep potential are interpolated.
        rb : float
            exclusive upper boundary of the range in which the ZBL potential and the deep potential are interpolated.
        alpha : float
            a tunable scale of the distances between atoms.

        Returns
        -------
        result_dict
            the result dict, defined by the fitting net output def.
        """
        nframes, nloc, nnei = nlist.shape
        extended_coord = extended_coord.reshape(nframes, -1, 3)
        nlists = build_multiple_neighbor_list(
            extended_coord,
            nlist,
            [self.zbl_model.rcut, self.dp_model.get_rcut()],
            [self.zbl_model.sel, sum(self.dp_model.get_sel())],
        )
        zbl_nlist = nlists[str(self.zbl_model.rcut) + "_" + str(self.zbl_model.sel)]
        dp_nlist = nlists[
            str(self.dp_model.get_rcut()) + "_" + str(sum(self.dp_model.get_sel()))
        ]

        zbl_nnei = zbl_nlist.shape[-1]
        dp_nnei = dp_nlist.shape[-1]

        # use the larger rr based on nlist
        nlist_ = zbl_nlist if zbl_nnei >= dp_nnei else dp_nlist
        masked_nlist = np.clip(nlist_, 0, None)
        pairwise_rr = np.sqrt(
            np.sum(
                np.power(
                    (
                        np.expand_dims(extended_coord, 2)
                        - np.expand_dims(extended_coord, 1)
                    ),
                    2,
                ),
                axis=-1,
            )
        )

        rr = np.take_along_axis(pairwise_rr[:, :nloc, :], masked_nlist, 2)
        # (nframes, nloc, 1)
        self.zbl_weight = self._compute_weight(nlist_, rr, ra, rb, alpha)
        # (nframes, nloc, 1)
        dp_energy = self.dp_model.forward_atomic(
            extended_coord, extended_atype, dp_nlist
        )["energy"]
        # (nframes, nloc, 1)
        zbl_energy = self.zbl_model.forward_atomic(
            extended_coord, extended_atype, zbl_nlist
        )["energy"]

        fit_ret = {
            "energy": (self.zbl_weight * zbl_energy + (1 - self.zbl_weight) * dp_energy)
        }  # (nframes, nloc, 1)
        return fit_ret

    def serialize(self) -> dict:
        return {
            "dp_model": self.dp_model.serialize(),
            "zbl_model": self.zbl_model.serialize(),
        }

    @classmethod
    def deserialize(cls, data) -> "LinearModel":
        dp_model = DPAtomicModel.deserialize(data["dp_model"])
        zbl_model = PairTabModel.deserialize(data["zbl_model"])
        return cls(dp_model, zbl_model)

    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy", shape=[1], reduciable=True, differentiable=True
                )
            ]
        )

    @staticmethod
    def _compute_weight(
        nlist: np.ndarray,
        rr: np.ndarray,
        ra: float,
        rb: float,
        alpha: Optional[float] = 0.1,
    ) -> np.ndarray:
        """ZBL weight.

        Parameters
        ----------
        nlist : np.ndarray
            the neighbour list, (nframes, nloc, nnei).
        rr : np.ndarray
            pairwise distance between atom i and atom j, (nframes, nloc, nnei).
        ra : float
            inclusive lower boundary of the range in which the ZBL potential and the deep potential are interpolated.
        rb : float
            exclusive upper boundary of the range in which the ZBL potential and the deep potential are interpolated.
        alpha : float
            a tunable scale of the distances between atoms.

        Returns
        -------
        np.ndarray
            the atomic ZBL weight for interpolation. (nframes, nloc)
        """
        assert (
            rb > ra
        ), "The upper boundary `rb` must be greater than the lower boundary `ra`."

        numerator = np.sum(
            rr * np.exp(-rr / alpha), axis=-1
        )  # masked nnei will be zero, no need to handle
        denominator = np.sum(
            np.where(nlist != -1, np.exp(-rr / alpha), np.zeros_like(nlist)),
            axis=-1,
        )  # handle masked nnei.
        sigma = numerator / denominator
        u = (sigma - ra) / (rb - ra)
        coef = np.zeros_like(u)
        left_mask = sigma < ra
        mid_mask = (ra <= sigma) & (sigma < rb)
        right_mask = sigma >= rb
        coef[left_mask] = 1
        smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef[mid_mask] = smooth[mid_mask]
        coef[right_mask] = 0
        return np.expand_dims(coef, -1)
