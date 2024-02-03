# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    nlist_distinguish_types,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)


def make_model(T_AtomicModel):
    """Make a model as a derived class of an atomic model.

    The model provide two interfaces.

    1. the `forward_common_lower`, that takes extended coordinates, atyps and neighbor list,
    and outputs the atomic and property and derivatives (if required) on the extended region.

    2. the `forward_common`, that takes coordinates, atypes and cell and predicts
    the atomic and reduced property, and derivatives (if required) on the local region.

    Parameters
    ----------
    T_AtomicModel
        The atomic model.

    Returns
    -------
    CM
        The model.

    """

    class CM(T_AtomicModel):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.fitting_output_def())

        # cannot use the name forward. torch script does not work
        def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, torch.Tensor]:
            """Return model prediction.

            Parameters
            ----------
            coord
                The coordinates of the atoms.
                shape: nf x (nloc x 3)
            atype
                The type of atoms. shape: nf x nloc
            box
                The simulation box. shape: nf x 9
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            nframes, nloc = atype.shape[:2]
            if box is not None:
                coord_normalized = normalize_coord(
                    coord.view(nframes, nloc, 3),
                    box.reshape(nframes, 3, 3),
                )
            else:
                coord_normalized = coord.clone()
            extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
                coord_normalized, atype, box, self.get_rcut()
            )
            nlist = build_neighbor_list(
                extended_coord,
                extended_atype,
                nloc,
                self.get_rcut(),
                self.get_sel(),
                distinguish_types=self.distinguish_types(),
            )
            extended_coord = extended_coord.view(nframes, -1, 3)
            model_predict_lower = self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam,
                aparam=aparam,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

        def forward_common_lower(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ):
            """Return model prediction. Lower interface that takes
            extended atomic coordinates and types, nlist, and mapping
            as input, and returns the predictions on the extended region.
            The predictions are not reduced.

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
            do_atomic_virial
                whether calculate atomic virial

            Returns
            -------
            result_dict
                the result dict, defined by the `FittingOutputDef`.

            """
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.view(nframes, -1, 3)
            nlist = self.format_nlist(extended_coord, extended_atype, nlist)
            atomic_ret = self.forward_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
            )
            model_predict = fit_output_to_model_output(
                atomic_ret,
                self.fitting_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

        def format_nlist(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
        ):
            """Format the neighbor list.

            1. If the number of neighbors in the `nlist` is equal to sum(self.sel),
            it does nothong

            2. If the number of neighbors in the `nlist` is smaller than sum(self.sel),
            the `nlist` is pad with -1.

            3. If the number of neighbors in the `nlist` is larger than sum(self.sel),
            the nearest sum(sel) neighbors will be preseved.

            Known limitations:

            In the case of self.distinguish_types, the nlist is always formatted.
            May have side effact on the efficiency.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x nall x 3
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel

            Returns
            -------
            formated_nlist
                the formated nlist.

            """
            distinguish_types = self.distinguish_types()
            nlist = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if distinguish_types:
                nlist = nlist_distinguish_types(nlist, extended_atype, self.get_sel())
            return nlist

        def _format_nlist(
            self,
            extended_coord: torch.Tensor,
            nlist: torch.Tensor,
            nnei: int,
        ):
            n_nf, n_nloc, n_nnei = nlist.shape
            # nf x nall x 3
            extended_coord = extended_coord.view([n_nf, -1, 3])
            rcut = self.get_rcut()

            if n_nnei < nnei:
                nlist = torch.cat(
                    [
                        nlist,
                        -1
                        * torch.ones(
                            [n_nf, n_nloc, nnei - n_nnei], dtype=nlist.dtype
                        ).to(nlist.device),
                    ],
                    dim=-1,
                )
            elif n_nnei > nnei:
                m_real_nei = nlist >= 0
                nlist = torch.where(m_real_nei, nlist, 0)
                # nf x nloc x 3
                coord0 = extended_coord[:, :n_nloc, :]
                # nf x (nloc x nnei) x 3
                index = nlist.view(n_nf, n_nloc * n_nnei, 1).expand(-1, -1, 3)
                coord1 = torch.gather(extended_coord, 1, index)
                # nf x nloc x nnei x 3
                coord1 = coord1.view(n_nf, n_nloc, n_nnei, 3)
                # nf x nloc x nnei
                rr = torch.linalg.norm(coord0[:, :, None, :] - coord1, dim=-1)
                rr = torch.where(m_real_nei, rr, float("inf"))
                rr, nlist_mapping = torch.sort(rr, dim=-1)
                nlist = torch.gather(nlist, 2, nlist_mapping)
                nlist = torch.where(rr > rcut, -1, nlist)
                nlist = nlist[..., :nnei]
            else:  # n_nnei == nnei:
                pass  # great!
            assert nlist.shape[-1] == nnei
            return nlist

    return CM
