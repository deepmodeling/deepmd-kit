# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
    OutputVariableOperation,
    check_operation_applied,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_ENER_FLOAT_PRECISION,
    GLOBAL_PT_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISON_DICT,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
    nlist_distinguish_types,
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
            self.precision_dict = PRECISION_DICT
            self.reverse_precision_dict = RESERVED_PRECISON_DICT
            self.global_pt_float_precision = GLOBAL_PT_FLOAT_PRECISION
            self.global_pt_ener_float_precision = GLOBAL_PT_ENER_FLOAT_PRECISION

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        @torch.jit.export
        def model_output_type(self) -> List[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            # jit: Comprehension ifs are not supported yet
            # type hint is critical for JIT
            vars: List[str] = []
            for kk, vv in var_defs.items():
                # .value is critical for JIT
                if vv.category == OutputVariableCategory.OUT.value:
                    vars.append(kk)
            return vars

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
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                If calculate the atomic virial.

            Returns
            -------
            ret_dict
                The result dict of type Dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            cc, bb, fp, ap, input_prec = self.input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam
            )
            del coord, box, fparam, aparam
            (
                extended_coord,
                extended_atype,
                mapping,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                cc,
                atype,
                self.get_rcut(),
                self.get_sel(),
                mixed_types=self.mixed_types(),
                box=bb,
            )
            model_predict_lower = self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                do_atomic_virial=do_atomic_virial,
                fparam=fp,
                aparam=ap,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = self.output_type_cast(model_predict, input_prec)
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
                coodinates in extended region. nf x (nall x 3)
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel.
            mapping
                mapps the extended indices to local indices. nf x nall.
            fparam
                frame parameter. nf x ndf
            aparam
                atomic parameter. nf x nloc x nda
            do_atomic_virial
                whether calculate atomic virial.

            Returns
            -------
            result_dict
                the result dict, defined by the `FittingOutputDef`.

            """
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.view(nframes, -1, 3)
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
            coord: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            str,
        ]:
            """Cast the input data to global float type."""
            input_prec = self.reverse_precision_dict[coord.dtype]
            ###
            ### type checking would not pass jit, convert to coord prec anyway
            ###
            # for vv, kk in zip([fparam, aparam], ["frame", "atomic"]):
            #     if vv is not None and self.reverse_precision_dict[vv.dtype] != input_prec:
            #         log.warning(
            #           f"type of {kk} parameter {self.reverse_precision_dict[vv.dtype]}"
            #           " does not match"
            #           f" that of the coordinate {input_prec}"
            #         )
            _lst: List[Optional[torch.Tensor]] = [
                vv.to(coord.dtype) if vv is not None else None
                for vv in [box, fparam, aparam]
            ]
            box, fparam, aparam = _lst
            if (
                input_prec
                == self.reverse_precision_dict[self.global_pt_float_precision]
            ):
                return coord, box, fparam, aparam, input_prec
            else:
                pp = self.global_pt_float_precision
                return (
                    coord.to(pp),
                    box.to(pp) if box is not None else None,
                    fparam.to(pp) if fparam is not None else None,
                    aparam.to(pp) if aparam is not None else None,
                    input_prec,
                )

        def output_type_cast(
            self,
            model_ret: Dict[str, torch.Tensor],
            input_prec: str,
        ) -> Dict[str, torch.Tensor]:
            """Convert the model output to the input prec."""
            do_cast = (
                input_prec
                != self.reverse_precision_dict[self.global_pt_float_precision]
            )
            pp = self.precision_dict[input_prec]
            odef = self.model_output_def()
            for kk in odef.keys():
                if kk not in model_ret.keys():
                    # do not return energy_derv_c if not do_atomic_virial
                    continue
                if check_operation_applied(odef[kk], OutputVariableOperation.REDU):
                    model_ret[kk] = (
                        model_ret[kk].to(self.global_pt_ener_float_precision)
                        if model_ret[kk] is not None
                        else None
                    )
                elif do_cast:
                    model_ret[kk] = (
                        model_ret[kk].to(pp) if model_ret[kk] is not None else None
                    )
            return model_ret

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
            mixed_types = self.mixed_types()
            nlist = self._format_nlist(extended_coord, nlist, sum(self.get_sel()))
            if not mixed_types:
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
                            [n_nf, n_nloc, nnei - n_nnei],
                            dtype=nlist.dtype,
                            device=nlist.device,
                        ),
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
