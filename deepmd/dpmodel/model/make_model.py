# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Callable,
    Optional,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.dpmodel.common import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
    NativeOP,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
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


def model_call_from_call_lower(
    *,  # enforce keyword-only arguments
    call_lower: Callable[
        [
            np.ndarray,
            np.ndarray,
            np.ndarray,
            Optional[np.ndarray],
            Optional[np.ndarray],
            bool,
        ],
        dict[str, np.ndarray],
    ],
    rcut: float,
    sel: list[int],
    mixed_types: bool,
    model_output_def: ModelOutputDef,
    coord: np.ndarray,
    atype: np.ndarray,
    box: Optional[np.ndarray] = None,
    fparam: Optional[np.ndarray] = None,
    aparam: Optional[np.ndarray] = None,
    do_atomic_virial: bool = False,
):
    """Return model prediction from lower interface.

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
        The result dict of type dict[str,np.ndarray].
        The keys are defined by the `ModelOutputDef`.

    """
    nframes, nloc = atype.shape[:2]
    cc, bb, fp, ap = coord, box, fparam, aparam
    del coord, box, fparam, aparam
    if bb is not None:
        coord_normalized = normalize_coord(
            cc.reshape(nframes, nloc, 3),
            bb.reshape(nframes, 3, 3),
        )
    else:
        coord_normalized = cc.copy()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, bb, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        # types will be distinguished in the lower interface,
        # so it doesn't need to be distinguished here
        distinguish_types=False,
    )
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    model_predict_lower = call_lower(
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
        model_output_def,
        mapping,
        do_atomic_virial=do_atomic_virial,
    )
    return model_predict


def make_model(T_AtomicModel: type[BaseAtomicModel]):
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

    class CM(NativeOP, BaseModel):
        def __init__(
            self,
            *args,
            # underscore to prevent conflict with normal inputs
            atomic_model_: Optional[T_AtomicModel] = None,
            **kwargs,
        ) -> None:
            BaseModel.__init__(self)
            if atomic_model_ is not None:
                self.atomic_model: T_AtomicModel = atomic_model_
            else:
                self.atomic_model: T_AtomicModel = T_AtomicModel(*args, **kwargs)
            self.precision_dict = PRECISION_DICT
            # not supported by flax
            # self.reverse_precision_dict = RESERVED_PRECISION_DICT
            self.global_np_float_precision = GLOBAL_NP_FLOAT_PRECISION
            self.global_ener_float_precision = GLOBAL_ENER_FLOAT_PRECISION

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        def model_output_type(self) -> list[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            vars = [
                kk
                for kk, vv in var_defs.items()
                if vv.category == OutputVariableCategory.OUT
            ]
            return vars

        def enable_compression(
            self,
            table_extrapolate: float = 5,
            table_stride_1: float = 0.01,
            table_stride_2: float = 0.1,
            check_frequency: int = -1,
        ) -> None:
            """Call atomic_model enable_compression().

            Parameters
            ----------
            table_extrapolate
                The scale of model extrapolation
            table_stride_1
                The uniform stride of the first table
            table_stride_2
                The uniform stride of the second table
            check_frequency
                The overflow check frequency
            """
            self.atomic_model.enable_compression(
                self.get_min_nbor_dist(),
                table_extrapolate,
                table_stride_1,
                table_stride_2,
                check_frequency,
            )

        def call(
            self,
            coord,
            atype,
            box: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            do_atomic_virial: bool = False,
        ) -> dict[str, np.ndarray]:
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
                The result dict of type dict[str,np.ndarray].
                The keys are defined by the `ModelOutputDef`.

            """
            cc, bb, fp, ap, input_prec = self.input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam
            )
            del coord, box, fparam, aparam
            model_predict = model_call_from_call_lower(
                call_lower=self.call_lower,
                rcut=self.get_rcut(),
                sel=self.get_sel(),
                mixed_types=self.mixed_types(),
                model_output_def=self.model_output_def(),
                coord=cc,
                atype=atype,
                box=bb,
                fparam=fp,
                aparam=ap,
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
                coordinates in extended region. nf x (nall x 3).
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
            nlist = self.format_nlist(
                extended_coord,
                extended_atype,
                nlist,
                extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            )
            cc_ext, _, fp, ap, input_prec = self.input_type_cast(
                extended_coord, fparam=fparam, aparam=aparam
            )
            del extended_coord, fparam, aparam
            model_predict = self.forward_common_atomic(
                cc_ext,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fp,
                aparam=ap,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = self.output_type_cast(model_predict, input_prec)
            return model_predict

        def forward_common_atomic(
            self,
            extended_coord: np.ndarray,
            extended_atype: np.ndarray,
            nlist: np.ndarray,
            mapping: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
            do_atomic_virial: bool = False,
        ):
            atomic_ret = self.atomic_model.forward_common_atomic(
                extended_coord,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fparam,
                aparam=aparam,
            )
            return fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                extended_coord,
                do_atomic_virial=do_atomic_virial,
            )

        forward_lower = call_lower

        def input_type_cast(
            self,
            coord: np.ndarray,
            box: Optional[np.ndarray] = None,
            fparam: Optional[np.ndarray] = None,
            aparam: Optional[np.ndarray] = None,
        ) -> tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[np.ndarray],
            Optional[np.ndarray],
            str,
        ]:
            """Cast the input data to global float type."""
            input_prec = RESERVED_PRECISION_DICT[self.precision_dict[coord.dtype.name]]
            ###
            ### type checking would not pass jit, convert to coord prec anyway
            ###
            _lst: list[Optional[np.ndarray]] = [
                vv.astype(coord.dtype) if vv is not None else None
                for vv in [box, fparam, aparam]
            ]
            box, fparam, aparam = _lst
            if input_prec == RESERVED_PRECISION_DICT[self.global_np_float_precision]:
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
            model_ret: dict[str, np.ndarray],
            input_prec: str,
        ) -> dict[str, np.ndarray]:
            """Convert the model output to the input prec."""
            do_cast = (
                input_prec != RESERVED_PRECISION_DICT[self.global_np_float_precision]
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
            extra_nlist_sort: bool = False,
        ):
            """Format the neighbor list.

            1. If the number of neighbors in the `nlist` is equal to sum(self.sel),
            it does nothong

            2. If the number of neighbors in the `nlist` is smaller than sum(self.sel),
            the `nlist` is pad with -1.

            3. If the number of neighbors in the `nlist` is larger than sum(self.sel),
            the nearest sum(sel) neighbors will be preserved.

            Known limitations:

            In the case of not self.mixed_types, the nlist is always formatted.
            May have side effact on the efficiency.

            Parameters
            ----------
            extended_coord
                coordinates in extended region. nf x nall x 3
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel
            extra_nlist_sort
                whether to forcibly sort the nlist.

            Returns
            -------
            formatted_nlist
                the formatted nlist.

            """
            n_nf, n_nloc, n_nnei = nlist.shape
            mixed_types = self.mixed_types()
            ret = self._format_nlist(
                extended_coord,
                nlist,
                sum(self.get_sel()),
                extra_nlist_sort=extra_nlist_sort,
            )
            if not mixed_types:
                ret = nlist_distinguish_types(ret, extended_atype, self.get_sel())
            return ret

        def _format_nlist(
            self,
            extended_coord: np.ndarray,
            nlist: np.ndarray,
            nnei: int,
            extra_nlist_sort: bool = False,
        ):
            xp = array_api_compat.array_namespace(extended_coord, nlist)
            n_nf, n_nloc, n_nnei = nlist.shape
            extended_coord = extended_coord.reshape([n_nf, -1, 3])
            nall = extended_coord.shape[1]
            rcut = self.get_rcut()

            if n_nnei < nnei:
                # make a copy before revise
                ret = xp.concat(
                    [
                        nlist,
                        -1 * xp.ones([n_nf, n_nloc, nnei - n_nnei], dtype=nlist.dtype),
                    ],
                    axis=-1,
                )

            if n_nnei > nnei or extra_nlist_sort:
                n_nf, n_nloc, n_nnei = nlist.shape
                # make a copy before revise
                m_real_nei = nlist >= 0
                ret = xp.where(m_real_nei, nlist, 0)
                coord0 = extended_coord[:, :n_nloc, :]
                index = ret.reshape(n_nf, n_nloc * n_nnei, 1).repeat(3, axis=2)
                coord1 = xp.take_along_axis(extended_coord, index, axis=1)
                coord1 = coord1.reshape(n_nf, n_nloc, n_nnei, 3)
                rr = xp.linalg.norm(coord0[:, :, None, :] - coord1, axis=-1)
                rr = xp.where(m_real_nei, rr, float("inf"))
                rr, ret_mapping = xp.sort(rr, axis=-1), xp.argsort(rr, axis=-1)
                ret = xp.take_along_axis(ret, ret_mapping, axis=2)
                ret = xp.where(rr > rcut, -1, ret)
                ret = ret[..., :nnei]
            # not extra_nlist_sort and n_nnei <= nnei:
            elif n_nnei == nnei:
                ret = nlist
            else:
                pass
            assert ret.shape[-1] == nnei
            return ret

        def do_grad_r(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.
            """
            return self.atomic_model.do_grad_r(var_name)

        def do_grad_c(
            self,
            var_name: Optional[str] = None,
        ) -> bool:
            """Tell if the output variable `var_name` is c_differentiable.
            if var_name is None, returns if any of the variable is c_differentiable.
            """
            return self.atomic_model.do_grad_c(var_name)

        def change_type_map(
            self, type_map: list[str], model_with_new_type_stat=None
        ) -> None:
            """Change the type related params to new ones, according to `type_map` and the original one in the model.
            If there are new types in `type_map`, statistics will be updated accordingly to `model_with_new_type_stat` for these new types.
            """
            self.atomic_model.change_type_map(type_map=type_map)

        def serialize(self) -> dict:
            return self.atomic_model.serialize()

        @classmethod
        def deserialize(cls, data) -> "CM":
            return cls(atomic_model_=T_AtomicModel.deserialize(data))

        def set_case_embd(self, case_idx: int):
            self.atomic_model.set_case_embd(case_idx)

        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""
            return self.atomic_model.get_dim_fparam()

        def get_dim_aparam(self) -> int:
            """Get the number (dimension) of atomic parameters of this atomic model."""
            return self.atomic_model.get_dim_aparam()

        def get_sel_type(self) -> list[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """
            return self.atomic_model.get_sel_type()

        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """
            return self.atomic_model.is_aparam_nall()

        def get_rcut(self) -> float:
            """Get the cut-off radius."""
            return self.atomic_model.get_rcut()

        def get_type_map(self) -> list[str]:
            """Get the type map."""
            return self.atomic_model.get_type_map()

        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nsel()

        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nnei()

        def get_sel(self) -> list[int]:
            """Returns the number of selected atoms for each type."""
            return self.atomic_model.get_sel()

        def mixed_types(self) -> bool:
            """If true, the model
            1. assumes total number of atoms aligned across frames;
            2. uses a neighbor list that does not distinguish different atomic types.

            If false, the model
            1. assumes total number of atoms of each atom type aligned across frames;
            2. uses a neighbor list that distinguishes different atomic types.

            """
            return self.atomic_model.mixed_types()

        def has_message_passing(self) -> bool:
            """Returns whether the model has message passing."""
            return self.atomic_model.has_message_passing()

        def need_sorted_nlist_for_lower(self) -> bool:
            """Returns whether the model needs sorted nlist when using `forward_lower`."""
            return self.atomic_model.need_sorted_nlist_for_lower()

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the output def of the atomic model."""
            return self.atomic_model.atomic_output_def()

        def get_ntypes(self) -> int:
            """Get the number of types."""
            return len(self.get_type_map())

    return CM
