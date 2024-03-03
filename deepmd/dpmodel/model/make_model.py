# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISON_DICT,
    NativeOP,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
    OutputVariableCategory,
    OutputVariableOperation,
    check_operation_applied,
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

    class CM(T_AtomicModel, NativeOP):
        def __init__(
            self,
            *args,
            **kwargs,
        ):
            super().__init__(
                *args,
                **kwargs,
            )
            self.precision_dict = PRECISION_DICT
            self.reverse_precision_dict = RESERVED_PRECISON_DICT
            self.global_np_float_precision = GLOBAL_NP_FLOAT_PRECISION
            self.global_ener_float_precision = GLOBAL_ENER_FLOAT_PRECISION

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        def model_output_type(self) -> List[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            vars = [
                kk
                for kk, vv in var_defs.items()
                if vv.category == OutputVariableCategory.OUT
            ]
            return vars

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
            cc, bb, fp, ap, input_prec = self.input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam
            )
            del coord, box, fparam, aparam
            if bb is not None:
                coord_normalized = normalize_coord(
                    cc.reshape(nframes, nloc, 3),
                    bb.reshape(nframes, 3, 3),
                )
            else:
                coord_normalized = cc.copy()
            extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
                coord_normalized, atype, bb, self.get_rcut()
            )
            nlist = build_neighbor_list(
                extended_coord,
                extended_atype,
                nloc,
                self.get_rcut(),
                self.get_sel(),
                distinguish_types=not self.mixed_types(),
            )
            extended_coord = extended_coord.reshape(nframes, -1, 3)
            model_predict_lower = self.call_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                fparam=fp,
                aparam=ap,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = self.output_type_cast(model_predict, input_prec)
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
            cc_ext, _, fp, ap, input_prec = self.input_type_cast(
                extended_coord, fparam=fparam, aparam=aparam
            )
            del extended_coord, fparam, aparam
            atomic_ret = self.forward_common_atomic(
                cc_ext,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fp,
                aparam=ap,
            )
            model_predict = fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                cc_ext,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = self.output_type_cast(model_predict, input_prec)
            return model_predict

        def input_type_cast(
            self,
            coord: np.ndarray,
            box: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
        ) -> Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            str,
        ]:
            """Cast the input data to global float type."""
            input_prec = self.reverse_precision_dict[
                self.precision_dict[coord.dtype.name]
            ]
            ###
            ### type checking would not pass jit, convert to coord prec anyway
            ###
            _lst: List[Optional[np.ndarray]] = [
                vv.astype(coord.dtype) if vv is not None else None
                for vv in [box, fparam, aparam]
            ]
            box, fparam, aparam = _lst
            if (
                input_prec
                == self.reverse_precision_dict[self.global_np_float_precision]
            ):
                return coord, box, fparam, aparam, input_prec
            else:
                pp = self.global_np_float_precision
                return (
                    coord.astype(pp),
                    box.astype(pp) if box is not None else None,
                    fparam.astype(pp) if fparam is not None else None,
                    aparam.astype(pp) if aparam is not None else None,
                    input_prec,
                )

        def output_type_cast(
            self,
            model_ret: Dict[str, np.ndarray],
            input_prec: str,
        ) -> Dict[str, np.ndarray]:
            """Convert the model output to the input prec."""
            do_cast = (
                input_prec
                != self.reverse_precision_dict[self.global_np_float_precision]
            )
            pp = self.precision_dict[input_prec]
            odef = self.model_output_def()
            for kk in odef.keys():
                if kk not in model_ret.keys():
                    # do not return energy_derv_c if not do_atomic_virial
                    continue
                if check_operation_applied(odef[kk], OutputVariableOperation.REDU):
                    model_ret[kk] = (
                        model_ret[kk].astype(self.global_ener_float_precision)
                        if model_ret[kk] is not None
                        else None
                    )
                elif do_cast:
                    model_ret[kk] = (
                        model_ret[kk].astype(pp) if model_ret[kk] is not None else None
                    )
            return model_ret

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

            In the case of not self.mixed_types, the nlist is always formatted.
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
            mixed_types = self.mixed_types()
            ret = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if not mixed_types:
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
