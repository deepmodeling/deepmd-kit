# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import numpy as np

from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.dpmodel.utils import (
    build_neighbor_list,
    extend_coord_with_ghosts,
    nlist_distinguish_types,
    normalize_coord,
)

from .transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)


def make_model(T_AtomicModel):
    """Make a model as a derived class of an atomic model.

    The model provide two interfaces.

    1. the `call_lower`, that takes extended coordinates, atyps and neighbor list,
    and outputs the atomic and property and derivatives (if required) on the extended region.

    2. the `call`, that takes coordinates, atypes and cell and predicts
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

        def call(
            self,
            coord,
            atype,
            box: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            do_atomic_virial: bool = False,
        ) -> Dict[str, np.ndarray]:
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
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,np.ndarray].
                The keys are defined by the `ModelOutputDef`.

            """
            nframes, nloc = atype.shape[:2]
            if box is not None:
                coord_normalized = normalize_coord(
                    coord.reshape(nframes, nloc, 3),
                    box.reshape(nframes, 3, 3),
                )
            else:
                coord_normalized = coord.copy()
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
            extended_coord = extended_coord.reshape(nframes, -1, 3)
            model_predict_lower = self.call_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            return model_predict

        def call_lower(
            self,
            extended_coord: np.ndarray,
            extended_atype: np.ndarray,
            nlist: np.ndarray,
            mapping: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            do_atomic_virial: bool = False,
        ):
            """Return model prediction. Lower interface that takes
            extended atomic coordinates and types, nlist, and mapping
            as input, and returns the predictions on the extended region.
            The predictions are not reduced.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x (nall x 3).
            extended_atype
                atomic type in extended region. nf x nall.
            nlist
                neighbor list. nf x nloc x nsel.
            mapping
                mapps the extended indices to local indices. nf x nall.
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                whether calculate atomic virial

            Returns
            -------
            result_dict
                the result dict, defined by the `FittingOutputDef`.

            """
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.reshape(nframes, -1, 3)
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
            extended_coord: np.ndarray,
            extended_atype: np.ndarray,
            nlist: np.ndarray,
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
            n_nf, n_nloc, n_nnei = nlist.shape
            distinguish_types = self.distinguish_types()
            ret = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if distinguish_types:
                ret = nlist_distinguish_types(ret, extended_atype, self.get_sel())
            return ret

        def _format_nlist(
            self,
            extended_coord: np.ndarray,
            nlist: np.ndarray,
            nnei: int,
        ):
            n_nf, n_nloc, n_nnei = nlist.shape
            extended_coord = extended_coord.reshape([n_nf, -1, 3])
            nall = extended_coord.shape[1]
            rcut = self.get_rcut()

            if n_nnei < nnei:
                # make a copy before revise
                ret = np.concatenate(
                    [
                        nlist,
                        -1 * np.ones([n_nf, n_nloc, nnei - n_nnei], dtype=nlist.dtype),
                    ],
                    axis=-1,
                )
            elif n_nnei > nnei:
                # make a copy before revise
                m_real_nei = nlist >= 0
                ret = np.where(m_real_nei, nlist, 0)
                coord0 = extended_coord[:, :n_nloc, :]
                index = ret.reshape(n_nf, n_nloc * n_nnei, 1).repeat(3, axis=2)
                coord1 = np.take_along_axis(extended_coord, index, axis=1)
                coord1 = coord1.reshape(n_nf, n_nloc, n_nnei, 3)
                rr = np.linalg.norm(coord0[:, :, None, :] - coord1, axis=-1)
                rr = np.where(m_real_nei, rr, float("inf"))
                rr, ret_mapping = np.sort(rr, axis=-1), np.argsort(rr, axis=-1)
                ret = np.take_along_axis(ret, ret_mapping, axis=2)
                ret = np.where(rr > rcut, -1, ret)
                ret = ret[..., :nnei]
            else:  # n_nnei == nnei:
                # copy anyway...
                ret = nlist
            assert ret.shape[-1] == nnei
            return ret

    return CM
