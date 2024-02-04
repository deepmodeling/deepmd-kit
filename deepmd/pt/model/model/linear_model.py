# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
)

from .base_atomic_model import (
    BaseAtomicModel,
)
from .dp_atomic_model import (
    DPAtomicModel,
)
from .model import (
    BaseModel,
)
from .pair_tab import (
    PairTabModel,
)


class LinearModel(BaseModel, BaseAtomicModel):
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
        mapping: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return atomic prediction.

        Note: currently only support the linear combination of a ZBL model and a DP model,
        the weight is calculated based on this paper:
        Appl. Phys. Lett. 114, 244101 (2019); https://doi.org/10.1063/1.5098061

        Parameters
        ----------
        extended_coord
            coodinates in extended region
        extended_atype
            atomic type in extended region
        nlist
            neighbor list. nf x nloc x nsel
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
        nlists = build_multiple_neighbor_list(
            extended_coord, nlist, self.get_rcuts(), self.get_sels()
        )

        zbl_nlist = nlists[str(self.zbl_model.rcut) + "_" + str(self.zbl_model.sel)]
        dp_nlist = nlists[
            str(self.dp_model.rcut) + "_" + str(sum(self.dp_model.sel))
        ]  # need to handle sel dtype.

        nframe, nloc, zbl_nnei = zbl_nlist.shape
        dp_nnei = dp_nlist.shape[-1]

        # use a larger rr based on nlist
        nlist_ = zbl_nlist if zbl_nnei >= dp_nnei else dp_nnei
        masked_nlist = torch.clamp(nlist_, 0)
        pairwise_rr = (
            (extended_coord.unsqueeze(2) - extended_coord.unsqueeze(1))
            .pow(2)
            .sum(-1)
            .sqrt()
        )
        rr = torch.gather(pairwise_rr[:, : nloc, :], 2, masked_nlist)
        assert rr.shape == nlist_.shape
        # (nframes, nloc)
        zbl_weight = self._compute_weight(rr, ra, rb, alpha)
        # (nframes, nloc)
        dp_energy = self.dp_model.forward_atomic(
            extended_coord, extended_atype, dp_nlist
        )["energy"].squeeze(-1)
        print(dp_energy.shape)
        # (nframes, nloc)
        zbl_energy = self.zbl_model.forward_atomic(
            extended_coord, extended_atype, zbl_nlist
        )["energy"]
        assert zbl_energy.shape == (nframe, nloc)
        
        fit_ret = (
            zbl_weight * zbl_energy + (1 - zbl_weight) * dp_energy
        )  # (nframes, nloc)
        return fit_ret
    
    def serialize(self):
        pass

    def deserialize(self):
        pass

    def fitting_output_def(self):
        pass
        

    def _compute_weight(
        self, rr: torch.Tensor, ra: float, rb: float, alpha: Optional[float] = 0.1
    ) -> torch.Tensor:
        """ZBL weight.

        Parameters
        ----------
        rr : torch.Tensor
            pairwise distance between atom i and atom j, (nframes, nloc, nnei).
        ra : float
            inclusive lower boundary of the range in which the ZBL potential and the deep potential are interpolated.
        rb : float
            exclusive upper boundary of the range in which the ZBL potential and the deep potential are interpolated.
        alpha : float
            a tunable scale of the distances between atoms.

        Returns
        -------
        torch.Tensor
            the atomic ZBL weight for interpolation. (nframes, nloc)
        """
        assert (
            rb > ra
        ), "The upper boundary `rb` must be greater than the lower boundary `ra`."


        sigma = torch.sum(rr * torch.exp(-rr / alpha), dim=-1) / torch.sum(
            torch.exp(-rr / alpha), dim=-1
        )  # (nframes, nloc)
        u = (sigma - ra) / (rb - ra)
        coef = torch.zeros_like(u)
        left_mask = sigma < ra
        mid_mask = (ra<=sigma) & (sigma<rb)
        right_mask = sigma >= rb
        coef[left_mask] = 0
        smooth = -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        coef[mid_mask] = smooth[mid_mask]
        coef[right_mask] = 1
        return coef
        # if sigma < ra:
        #     return torch.ones_like(u)
        # elif sigma < rb:
        #     return -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        # elif sigma >= rb:
        #     return torch.zeros_like(u)
