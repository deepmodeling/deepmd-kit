# SPDX-License-Identifier: LGPL-3.0-or-later
from collections.abc import (
    Callable,
)
from typing import (
    Any,
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


def make_model(T_AtomicModel: type[BaseAtomicModel]) -> type:
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
            *args: Any,
            # underscore to prevent conflict with normal inputs
            atomic_model_: T_AtomicModel | None = None,  # type: ignore
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if atomic_model_ is not None:
                self.atomic_model: T_AtomicModel = atomic_model_  # type: ignore
            else:
                self.atomic_model: T_AtomicModel = T_AtomicModel(*args, **kwargs)  # type: ignore
            self.precision_dict = PRECISION_DICT
            self.reverse_precision_dict = RESERVED_PRECISION_DICT
            self.global_pt_float_precision = GLOBAL_PT_FLOAT_PRECISION
            self.global_pt_ener_float_precision = GLOBAL_PT_ENER_FLOAT_PRECISION

        def model_output_def(self) -> ModelOutputDef:
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
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            coord_corr_for_virial: torch.Tensor | None = None,
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
            coord_corr_for_virial
                The coordinates correction of the atoms for virial.
                shape: nf x (nloc x 3)

            Returns
            -------
            ret_dict
                The result dict of type dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            cc, bb, fp, ap, input_prec = self._input_type_cast(
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
            if coord_corr_for_virial is not None:
                coord_corr_for_virial = coord_corr_for_virial.to(cc.dtype)
                extended_coord_corr = torch.gather(
                    coord_corr_for_virial, 1, mapping.unsqueeze(-1).expand(-1, -1, 3)
                )
            else:
                extended_coord_corr = None

            model_predict_lower = self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                mapping,
                do_atomic_virial=do_atomic_virial,
                fparam=fp,
                aparam=ap,
                extended_coord_corr=extended_coord_corr,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

        def forward_common_flat_native(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            batch: torch.Tensor,
            ptr: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            extended_atype: torch.Tensor | None = None,
            extended_batch: torch.Tensor | None = None,
            extended_image: torch.Tensor | None = None,
            extended_ptr: torch.Tensor | None = None,
            mapping: torch.Tensor | None = None,
            central_ext_index: torch.Tensor | None = None,
            nlist: torch.Tensor | None = None,
            nlist_ext: torch.Tensor | None = None,
            a_nlist: torch.Tensor | None = None,
            a_nlist_ext: torch.Tensor | None = None,
            nlist_mask: torch.Tensor | None = None,
            a_nlist_mask: torch.Tensor | None = None,
            edge_index: torch.Tensor | None = None,
            angle_index: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Forward pass for mixed-nloc batches with a precomputed flat graph.

            This path consumes graph tensors prepared by the LMDB collate function
            and keeps atom-wise values flattened across frames.

            Parameters
            ----------
            coord
                Flattened atomic coordinates with shape [total_atoms, 3].
            atype
                Flattened atomic types with shape [total_atoms].
            batch
                Atom-to-frame assignment with shape [total_atoms].
            ptr
                Frame boundaries with shape [nframes + 1].
            box
                Simulation boxes with shape [nframes, 9].
            fparam
                Frame parameters with shape [nframes, ndf].
            aparam
                Flattened atomic parameters with shape [total_atoms, nda].
            do_atomic_virial
                Whether to calculate atomic virial.

            Returns
            -------
            model_predict : dict[str, torch.Tensor]
                Model predictions with flat format:
                - atomwise outputs: [total_atoms, ...]
                - frame-wise outputs: [nframes, ...]

            Notes
            -----
            The precomputed graph fields are required for this path; missing
            fields are treated as a data pipeline error.
            """
            # Enable gradient tracking for coord and box if needed
            if self.do_grad_r("energy"):
                coord = coord.clone().detach().requires_grad_(True)
            if self.do_grad_c("energy") and box is not None:
                box = box.clone().detach().requires_grad_(True)
            if (
                extended_atype is not None
                and extended_batch is not None
                and extended_image is not None
                and mapping is not None
                and nlist is not None
                and nlist_ext is not None
                and a_nlist is not None
                and a_nlist_ext is not None
                and nlist_mask is not None
                and a_nlist_mask is not None
                and central_ext_index is not None
            ):
                from deepmd.pt.utils.nlist import rebuild_extended_coord_from_flat_graph

                extended_coord = rebuild_extended_coord_from_flat_graph(
                    coord,
                    box,
                    mapping,
                    extended_batch,
                    extended_image,
                )
            else:
                raise RuntimeError(
                    "Flat mixed-batch forward requires precomputed graph fields from "
                    "the LMDB collate_fn."
                )
            # Pass flat extended coordinates directly to the atomic model.
            assert extended_atype is not None
            assert extended_batch is not None
            assert mapping is not None
            assert nlist is not None
            model_predict_lower = self.forward_common_lower_flat(
                extended_coord,
                extended_atype,
                extended_batch,
                nlist,
                mapping,
                batch,
                ptr,
                do_atomic_virial=do_atomic_virial,
                fparam=fparam,
                aparam=aparam,
                extended_ptr=extended_ptr,
                central_ext_index=central_ext_index,
                nlist_ext=nlist_ext,
                a_nlist=a_nlist,
                a_nlist_ext=a_nlist_ext,
                nlist_mask=nlist_mask,
                a_nlist_mask=a_nlist_mask,
                edge_index=edge_index,
                angle_index=angle_index,
            )

            # Compute derivatives if needed
            if self.do_grad_r("energy") or self.do_grad_c("energy"):
                model_predict_lower = self._compute_derivatives_flat(
                    model_predict_lower,
                    extended_coord,
                    extended_atype,
                    extended_batch,
                    coord,
                    atype,
                    batch,
                    ptr,
                    box,
                    do_atomic_virial,
                )

            return model_predict_lower

        def forward_common_lower_flat(
            self,
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            extended_batch: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor,
            batch: torch.Tensor,
            ptr: torch.Tensor,
            do_atomic_virial: bool = False,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            extended_ptr: torch.Tensor | None = None,
            central_ext_index: torch.Tensor | None = None,
            nlist_ext: torch.Tensor | None = None,
            a_nlist: torch.Tensor | None = None,
            a_nlist_ext: torch.Tensor | None = None,
            nlist_mask: torch.Tensor | None = None,
            a_nlist_mask: torch.Tensor | None = None,
            edge_index: torch.Tensor | None = None,
            angle_index: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Lower interface for flat batch format.

            Parameters
            ----------
            extended_coord : torch.Tensor
                Extended coordinates [total_extended_atoms, 3].
            extended_atype : torch.Tensor
                Extended atom types [total_extended_atoms].
            extended_batch : torch.Tensor
                Frame assignment for extended atoms [total_extended_atoms].
            nlist : torch.Tensor
                Neighbor list [total_atoms, nnei].
            mapping : torch.Tensor
                Extended atom -> local flat index mapping [total_extended_atoms].
            batch : torch.Tensor
                Frame assignment for local atoms [total_atoms].
            ptr : torch.Tensor
                Frame boundaries [nframes + 1].
            do_atomic_virial : bool
                Whether to compute atomic virial.
            fparam : torch.Tensor | None
                Frame parameters [nframes, ndf].
            aparam : torch.Tensor | None
                Atomic parameters [total_atoms, nda].

            Returns
            -------
            model_predict : dict[str, torch.Tensor]
                Model predictions in flat format.
            """
            # The atomic model keeps atom-wise outputs in flat format.
            model_ret = self.atomic_model.forward_common_atomic_flat(
                extended_coord,
                extended_atype,
                extended_batch,
                nlist,
                mapping,
                batch,
                ptr,
                fparam=fparam,
                aparam=aparam,
                extended_ptr=extended_ptr,
                central_ext_index=central_ext_index,
                nlist_ext=nlist_ext,
                a_nlist=a_nlist,
                a_nlist_ext=a_nlist_ext,
                nlist_mask=nlist_mask,
                a_nlist_mask=a_nlist_mask,
                edge_index=edge_index,
                angle_index=angle_index,
            )

            # Reduce atom-wise energy to frame-wise energy.
            nframes = ptr.numel() - 1
            if "energy" in model_ret:
                energy_atomic = model_ret["energy"]  # [total_atoms, 1]
                energy_redu = energy_atomic.new_zeros((nframes, energy_atomic.shape[-1]))
                energy_redu.index_add_(0, batch, energy_atomic)
                model_ret["energy_redu"] = energy_redu

            return model_ret

        def _compute_derivatives_flat(
            self,
            fit_ret: dict[str, torch.Tensor],
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            extended_batch: torch.Tensor,
            coord: torch.Tensor,
            atype: torch.Tensor,
            batch: torch.Tensor,
            ptr: torch.Tensor,
            box: torch.Tensor | None,
            do_atomic_virial: bool,
        ) -> dict[str, torch.Tensor]:
            """Compute force and virial derivatives for flat batch format.

            Parameters
            ----------
            fit_ret : dict[str, torch.Tensor]
                Fitting network output with "energy" key [total_atoms, 1].
            extended_coord : torch.Tensor
                Extended coordinates [total_extended_atoms, 3].
            extended_atype : torch.Tensor
                Extended atom types [total_extended_atoms].
            extended_batch : torch.Tensor
                Frame assignment for extended atoms [total_extended_atoms].
            coord : torch.Tensor
                Original coordinates [total_atoms, 3].
            atype : torch.Tensor
                Original atom types [total_atoms].
            batch : torch.Tensor
                Frame assignment for original atoms [total_atoms].
            ptr : torch.Tensor
                Frame boundaries [nframes + 1].
            box : torch.Tensor | None
                Simulation boxes [nframes, 9].
            do_atomic_virial : bool
                Whether to compute atomic virial.

            Returns
            -------
            model_predict : dict[str, torch.Tensor]
                Model predictions with derivatives in flat format.
            """
            # Force is the negative gradient of the total atomic energy.
            if self.do_grad_r("energy"):
                energy_atomic = fit_ret["energy"]  # [total_atoms, 1]

                energy_derv_r = torch.autograd.grad(
                    outputs=energy_atomic.sum(),
                    inputs=coord,
                    create_graph=True,
                    retain_graph=True,
                )[0]  # [total_atoms, 3]

                fit_ret["energy_derv_r"] = -energy_derv_r.unsqueeze(-2)  # [total_atoms, 1, 3]
                # Also provide dforce field for compatibility with EnergyModel.forward()
                fit_ret["dforce"] = -energy_derv_r  # [total_atoms, 3]

            # Compute virial: dE/dh
            if self.do_grad_c("energy"):
                nframes = ptr.numel() - 1
                energy_redu = fit_ret["energy_redu"]  # [nframes, 1]

                if box is not None:
                    energy_derv_c_redu = torch.autograd.grad(
                        outputs=energy_redu.sum(),
                        inputs=box,
                        create_graph=True,
                        retain_graph=True,
                    )[0]  # [nframes, 9]

                    fit_ret["energy_derv_c_redu"] = energy_derv_c_redu.unsqueeze(
                        1
                    )  # [nframes, 1, 9]

                    # Preserve the current flat-path behavior: reduced virial is
                    # available, atomic virial is not populated yet.
                    if do_atomic_virial:
                        pass  # Not yet implemented for flat format

            return fit_ret

        def forward_common_flat(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            batch: torch.Tensor,
            ptr: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            extended_atype: torch.Tensor | None = None,
            extended_batch: torch.Tensor | None = None,
            extended_image: torch.Tensor | None = None,
            extended_ptr: torch.Tensor | None = None,
            mapping: torch.Tensor | None = None,
            central_ext_index: torch.Tensor | None = None,
            nlist: torch.Tensor | None = None,
            nlist_ext: torch.Tensor | None = None,
            a_nlist: torch.Tensor | None = None,
            a_nlist_ext: torch.Tensor | None = None,
            nlist_mask: torch.Tensor | None = None,
            a_nlist_mask: torch.Tensor | None = None,
            edge_index: torch.Tensor | None = None,
            angle_index: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Forward pass for flat mixed-nloc batch.

            This method consumes the precomputed flat graph produced by LMDB
            collation and returns the same output keys as the regular path.

            Parameters
            ----------
            coord
                Flattened atomic coordinates with shape [total_atoms, 3].
            atype
                Flattened atomic types with shape [total_atoms].
            batch
                Atom-to-frame assignment with shape [total_atoms].
            ptr
                Frame boundaries with shape [nframes + 1].
            box
                Simulation boxes with shape [nframes, 9].
            fparam
                Frame parameters with shape [nframes, ndf].
            aparam
                Flattened atomic parameters with shape [total_atoms, nda].
            do_atomic_virial
                Whether to calculate atomic virial.

            Returns
            -------
            model_predict : dict[str, torch.Tensor]
                Model predictions with flat format:
                - atomwise outputs: [total_atoms, ...]
                - frame-wise outputs: [nframes, ...]
            """
            return self.forward_common_flat_native(
                coord,
                atype,
                batch,
                ptr,
                box,
                fparam,
                aparam,
                do_atomic_virial,
                extended_atype=extended_atype,
                extended_batch=extended_batch,
                extended_image=extended_image,
                extended_ptr=extended_ptr,
                mapping=mapping,
                central_ext_index=central_ext_index,
                nlist=nlist,
                nlist_ext=nlist_ext,
                a_nlist=a_nlist,
                a_nlist_ext=a_nlist_ext,
                nlist_mask=nlist_mask,
                a_nlist_mask=a_nlist_mask,
                edge_index=edge_index,
                angle_index=angle_index,
            )

        def get_out_bias(self) -> torch.Tensor:
            return self.atomic_model.get_out_bias()

        def set_out_bias(self, out_bias: torch.Tensor) -> None:
            self.atomic_model.set_out_bias(out_bias)

        def change_out_bias(
            self,
            merged: Any,
            bias_adjust_mode: str = "change-by-statistic",
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
            extended_coord: torch.Tensor,
            extended_atype: torch.Tensor,
            nlist: torch.Tensor,
            mapping: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            comm_dict: dict[str, torch.Tensor] | None = None,
            extra_nlist_sort: bool = False,
            extended_coord_corr: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
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
            extended_coord_corr
                coordinates correction for virial in extended region. nf x (nall x 3)

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
            cc_ext, _, fp, ap, input_prec = self._input_type_cast(
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
                mask=atomic_ret["mask"] if "mask" in atomic_ret else None,
                extended_coord_corr=extended_coord_corr,
            )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

        def _input_type_cast(
            self,
            coord: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
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
            _lst: list[torch.Tensor | None] = [
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

        def _output_type_cast(
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
        ) -> torch.Tensor:
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
        ) -> torch.Tensor:
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
            var_name: str | None = None,
        ) -> bool:
            """Tell if the output variable `var_name` is r_differentiable.
            if var_name is None, returns if any of the variable is r_differentiable.
            """
            return self.atomic_model.do_grad_r(var_name)

        def do_grad_c(
            self,
            var_name: str | None = None,
        ) -> bool:
            """Tell if the output variable `var_name` is c_differentiable.
            if var_name is None, returns if any of the variable is c_differentiable.
            """
            return self.atomic_model.do_grad_c(var_name)

        def change_type_map(
            self, type_map: list[str], model_with_new_type_stat: Any | None = None
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
        def deserialize(cls, data: Any) -> "CM":
            return cls(atomic_model_=T_AtomicModel.deserialize(data))

        def set_case_embd(self, case_idx: int) -> None:
            self.atomic_model.set_case_embd(case_idx)

        @torch.jit.export
        def get_dim_fparam(self) -> int:
            """Get the number (dimension) of frame parameters of this atomic model."""
            return self.atomic_model.get_dim_fparam()

        @torch.jit.export
        def has_default_fparam(self) -> bool:
            """Check if the model has default frame parameters."""
            return self.atomic_model.has_default_fparam()

        def get_default_fparam(self) -> torch.Tensor | None:
            return self.atomic_model.get_default_fparam()

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
            sampled_func: Callable[[], Any],
            stat_file_path: DPPath | None = None,
            preset_observed_type: list[str] | None = None,
        ) -> None:
            """Compute or load the statistics."""
            return self.atomic_model.compute_or_load_stat(
                sampled_func,
                stat_file_path,
                preset_observed_type=preset_observed_type,
            )

        @torch.jit.export
        def get_observed_type_list(self) -> list[str]:
            """Get observed types (elements) of the model during data statistics.

            Returns
            -------
            observed_type_list: a list of the observed types in this model.
            """
            type_map = self.get_type_map()
            out_bias = self.atomic_model.get_out_bias()[0]

            assert out_bias is not None, "No out_bias found in the model."
            assert out_bias.dim() == 2, "The supported out_bias should be a 2D tensor."
            assert out_bias.size(0) == len(type_map), (
                "The out_bias shape does not match the type_map length."
            )
            bias_mask = (
                torch.gt(torch.abs(out_bias), 1e-6).any(dim=-1).detach().cpu()
            )  # 1e-6 for stability

            observed_type_list: list[str] = []
            for i in range(len(type_map)):
                if bias_mask[i]:
                    observed_type_list.append(type_map[i])
            return observed_type_list

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
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
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
