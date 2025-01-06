# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableCategory,
    OutputVariableOperation,
    check_operation_applied,
)
from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.env import (
    GLOBAL_PT_ENER_FLOAT_PRECISION,
    GLOBAL_PT_FLOAT_PRECISION,
    PRECISION_DICT,
    RESERVED_PRECISION_DICT,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
    nlist_distinguish_types,
)
from deepmd.utils.path import (
    DPPath,
)


def make_model(T_AtomicModel: type[BaseAtomicModel]):
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

    class CM(BaseModel):
        def __init__(
            self,
            *args,
            # underscore to prevent conflict with normal inputs
            atomic_model_: Optional[T_AtomicModel] = None,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            if atomic_model_ is not None:
                self.atomic_model: T_AtomicModel = atomic_model_
            else:
                self.atomic_model: T_AtomicModel = T_AtomicModel(*args, **kwargs)
            self.precision_dict = PRECISION_DICT
            self.reverse_precision_dict = RESERVED_PRECISION_DICT
            self.global_pt_float_precision = GLOBAL_PT_FLOAT_PRECISION
            self.global_pt_ener_float_precision = GLOBAL_PT_ENER_FLOAT_PRECISION

        def model_output_def(self):
            """Get the output def for the model."""
            return ModelOutputDef(self.atomic_output_def())

        @torch.jit.export
        def model_output_type(self) -> list[str]:
            """Get the output type for the model."""
            output_def = self.model_output_def()
            var_defs = output_def.var_defs
            # jit: Comprehension ifs are not supported yet
            # type hint is critical for JIT
            vars: list[str] = []
            for kk, vv in var_defs.items():
                # .value is critical for JIT
                if vv.category == OutputVariableCategory.OUT.value:
                    vars.append(kk)
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

        # cannot use the name forward. torch script does not work
        def forward_common(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> dict[str, torch.Tensor]:
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
                The result dict of type dict[str,torch.Tensor].
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
                # types will be distinguished in the lower interface,
                # so it doesn't need to be distinguished here
                mixed_types=True,
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

        def get_out_bias(self) -> torch.Tensor:
            return self.atomic_model.get_out_bias()

        def set_out_bias(self, out_bias: torch.Tensor) -> None:
            self.atomic_model.set_out_bias(out_bias)

        def change_out_bias(
            self,
            merged,
            bias_adjust_mode="change-by-statistic",
        ) -> None:
            """Change the output bias of atomic model according to the input data and the pretrained model.

            Parameters
            ----------
            merged : Union[Callable[[], list[dict]], list[dict]]
                - list[dict]: A list of data samples from various data systems.
                    Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                    originating from the `i`-th data system.
                - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                    only when needed. Since the sampling process can be slow and memory-intensive,
                    the lazy function helps by only sampling once.
            bias_adjust_mode : str
                The mode for changing output bias : ['change-by-statistic', 'set-by-statistic']
                'change-by-statistic' : perform predictions on labels of target dataset,
                        and do least square on the errors to obtain the target shift as bias.
                'set-by-statistic' : directly use the statistic output bias in the target dataset.
            """
            self.atomic_model.change_out_bias(
                merged,
                bias_adjust_mode=bias_adjust_mode,
            )

        def forward_common_lower(
            self,
            extended_coord,
            extended_atype,
            nlist,
            mapping: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
            comm_dict: Optional[dict[str, torch.Tensor]] = None,
            extra_nlist_sort: bool = False,
        ):
            """Return model prediction. Lower interface that takes
            extended atomic coordinates and types, nlist, and mapping
            as input, and returns the predictions on the extended region.
            The predictions are not reduced.

            Parameters
            ----------
            extended_coord
                coordinates in extended region. nf x (nall x 3)
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
            comm_dict
                The data needed for communication for parallel inference.
            extra_nlist_sort
                whether to forcibly sort the nlist.

            Returns
            -------
            result_dict
                the result dict, defined by the `FittingOutputDef`.

            """
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.view(nframes, -1, 3)
            nlist = self.format_nlist(
                extended_coord, extended_atype, nlist, extra_nlist_sort=extra_nlist_sort
            )
            cc_ext, _, fp, ap, input_prec = self.input_type_cast(
                extended_coord, fparam=fparam, aparam=aparam
            )
            del extended_coord, fparam, aparam
            atomic_ret = self.atomic_model.forward_common_atomic(
                cc_ext,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fp,
                aparam=ap,
                comm_dict=comm_dict,
            )
            model_predict = fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                cc_ext,
                do_atomic_virial=do_atomic_virial,
                create_graph=self.training,
            )
            model_predict = self.output_type_cast(model_predict, input_prec)
            return model_predict

        def input_type_cast(
            self,
            coord: torch.Tensor,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
        ) -> tuple[
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
            _lst: list[Optional[torch.Tensor]] = [
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
            model_ret: dict[str, torch.Tensor],
            input_prec: str,
        ) -> dict[str, torch.Tensor]:
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
            formated_nlist
                the formatted nlist.

            """
            mixed_types = self.mixed_types()
            nlist = self._format_nlist(
                extended_coord,
                nlist,
                sum(self.get_sel()),
                extra_nlist_sort=extra_nlist_sort,
            )
            if not mixed_types:
                nlist = nlist_distinguish_types(nlist, extended_atype, self.get_sel())
            return nlist

        def _format_nlist(
            self,
            extended_coord: torch.Tensor,
            nlist: torch.Tensor,
            nnei: int,
            extra_nlist_sort: bool = False,
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

            if n_nnei > nnei or extra_nlist_sort:
                n_nf, n_nloc, n_nnei = nlist.shape
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
            else:  # not extra_nlist_sort and n_nnei <= nnei:
                pass  # great!
            assert nlist.shape[-1] == nnei
            return nlist

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
            self.atomic_model.change_type_map(
                type_map=type_map,
                model_with_new_type_stat=model_with_new_type_stat.atomic_model
                if model_with_new_type_stat is not None
                else None,
            )

        def serialize(self) -> dict:
            return self.atomic_model.serialize()

        @classmethod
        def deserialize(cls, data) -> "CM":
            return cls(atomic_model_=T_AtomicModel.deserialize(data))

        def set_case_embd(self, case_idx: int):
            self.atomic_model.set_case_embd(case_idx)

        @torch.jit.export
        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""
            return self.atomic_model.get_dim_fparam()

        @torch.jit.export
        def get_dim_aparam(self) -> int:
            """Get the number (dimension) of atomic parameters of this atomic model."""
            return self.atomic_model.get_dim_aparam()

        @torch.jit.export
        def get_sel_type(self) -> list[int]:
            """Get the selected atom types of this model.

            Only atoms with selected atom types have atomic contribution
            to the result of the model.
            If returning an empty list, all atom types are selected.
            """
            return self.atomic_model.get_sel_type()

        @torch.jit.export
        def is_aparam_nall(self) -> bool:
            """Check whether the shape of atomic parameters is (nframes, nall, ndim).

            If False, the shape is (nframes, nloc, ndim).
            """
            return self.atomic_model.is_aparam_nall()

        @torch.jit.export
        def get_rcut(self) -> float:
            """Get the cut-off radius."""
            return self.atomic_model.get_rcut()

        @torch.jit.export
        def get_type_map(self) -> list[str]:
            """Get the type map."""
            return self.atomic_model.get_type_map()

        @torch.jit.export
        def get_nsel(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nsel()

        @torch.jit.export
        def get_nnei(self) -> int:
            """Returns the total number of selected neighboring atoms in the cut-off radius."""
            return self.atomic_model.get_nnei()

        def atomic_output_def(self) -> FittingOutputDef:
            """Get the output def of the atomic model."""
            return self.atomic_model.atomic_output_def()

        def compute_or_load_stat(
            self,
            sampled_func,
            stat_file_path: Optional[DPPath] = None,
        ):
            """Compute or load the statistics."""
            return self.atomic_model.compute_or_load_stat(sampled_func, stat_file_path)

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

        @torch.jit.export
        def has_message_passing(self) -> bool:
            """Returns whether the model has message passing."""
            return self.atomic_model.has_message_passing()

        def need_sorted_nlist_for_lower(self) -> bool:
            """Returns whether the model needs sorted nlist when using `forward_lower`."""
            return self.atomic_model.need_sorted_nlist_for_lower()

        def forward(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
        ) -> dict[str, torch.Tensor]:
            # directly call the forward_common method when no specific transform rule
            return self.forward_common(
                coord,
                atype,
                box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
            )

    return CM
