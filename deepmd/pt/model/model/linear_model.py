# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
)

import torch

from deepmd.model_format import (
    FittingOutputDef,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
)

from .atomic_model import (
    AtomicModel,
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


class LinearModel(BaseModel, AtomicModel):
    """Model linearly combine a list of AtomicModels.

    Parameters
    ----------
    models
            This linear model should take a DPAtomicModel and a PairTable model in the exact order.
    """

    def __init__(
        self,
        models: List[AtomicModel],
        **kwargs,
    ):
        super().__init__()
        self.models = models
        self.dp_model = models[0]
        self.zbl_model = models[1]
        assert (
            isinstance(self.zbl_model, PairTabModel)
            and isinstance(self.dp_model, DPAtomicModel)
        ), "The provided models are not in the correct order `DPAtomicModel` + `PairTabModel`."
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
        return sorted([model.get_rcut() for model in self.models])

    def get_sel(self) -> int:
        """Get the neighbor selection."""
        return self.get_sels()[-1]

    def get_sels(self) -> int:
        """Get the neighbor selection for each individual models in ascending order."""
        return sorted(
            [
                sum(model.get_sel())
                if isinstance(model.get_sel(), list)
                else model.get_sel()
                for model in self.models
            ]
        )

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
        # the DPAtomicModel sel is always a List or Union[List, int]?
        nlists = build_multiple_neighbor_list(
            extended_coord, nlist, self.get_rcuts(), self.get_sels()
        )

        zbl_nlist = nlists[str(self.zbl_model.rcut) + "_" + str(self.zbl_model.sel)]
        dp_nlist = nlists[
            str(self.dp_model.rcut) + "_" + str(self.dp_model.sel)
        ]  # need to handle sel dtype.

        zbl_nframe, zbl_nloc, zbl_nnei = zbl_nlist.shape
        dp_nframe, dp_nloc, dp_nnei = dp_nlist.shape
        zbl_atype = extended_atype[
            :, :zbl_nloc
        ]  # nframe, nloc should all be the same, only difference is nnei based on rcut and nlist.
        dp_atype = extended_atype[:, :dp_nloc]

        # which rr should I use? this rr should be (nfrmaes, nloc, nnei)
        zbl_weight = self._compute_weight(rr, ra, rb)

        dp_energy = self.dp_model.forward_atomic(extended_coord, dp_atype, nlist)[
            "energy"
        ]
        zbl_energy = self.zbl_model.forward_atomic(extended_coord, zbl_atype, nlist)[
            "energy"
        ]
        fit_ret = (
            zbl_weight * zbl_energy + (1 - zbl_weight) * dp_energy
        )  # (nframes, nloc)
        return fit_ret

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
        if sigma < ra:
            return torch.ones_like(u)
        elif ra <= sigma < rb:
            return -6 * u**5 + 15 * u**4 - 10 * u**3 + 1
        else:
            return torch.zeros_like(u)
