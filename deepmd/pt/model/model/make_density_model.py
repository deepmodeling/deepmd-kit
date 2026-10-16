# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import torch

from deepmd.pt.model.atomic_model.base_atomic_model import (
    BaseAtomicModel,
)
from deepmd.pt.model.model.make_model import (
    make_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.model.transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)
from deepmd.pt.utils.nlist import (
    build_directional_neighbor_list,
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)

log = logging.getLogger(__name__)


def make_density_model(T_AtomicModel: type[BaseAtomicModel]) -> type[BaseModel]:
    """Make a density model as a derived class of an atomic model.

    The model predicts a scalar density on grid points instead of atomic
    energies. It is built by subclassing the standard model made by
    `make_model` and overriding only the interfaces that differ:

    1. the `forward_common_lower`, that takes extended coordinates, atypes,
    neighbor list and the grid (with its type and directional neighbor list),
    and outputs the density on the grid points;

    2. the `forward_common`, that takes coordinates, atypes, cell and grid,
    and predicts the density on the grid points.

    Parameters
    ----------
    T_AtomicModel
        The atomic model.

    Returns
    -------
    CM
        The model.

    """

    class CM(make_model(T_AtomicModel)):
        def forward_common(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            coord_corr_for_virial: torch.Tensor | None = None,
            charge_spin: torch.Tensor | None = None,
            grid: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Return model prediction.

            The base parameters keep the base positional order (including
            ``coord_corr_for_virial``, which is accepted and ignored since
            density has no virial correction); the density-specific
            ``grid`` is appended at the end and is required for this model.

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
            grid
                The coordinates of the grids.
                shape: nf x (ngrid x 3)

            Returns
            -------
            ret_dict
                The result dict of type dict[str,torch.Tensor].
                The keys are defined by the `ModelOutputDef`.

            """
            del coord_corr_for_virial
            assert grid is not None
            cc, gg, bb, fp, ap, input_prec = self._input_type_cast(
                coord, box=box, fparam=fparam, aparam=aparam, grid=grid
            )
            del coord, grid, box, fparam, aparam
            gg = gg.view(gg.shape[0], -1, 3)
            if bb is not None:
                # wrap grid points into the primary cell: the atomic
                # coordinates are normalized and extended with ghosts, so
                # periodically equivalent grids outside the cell would
                # otherwise lose their neighbors
                gg = normalize_coord(gg, bb.to(gg.device).view(bb.shape[0], 3, 3))
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
            grid_type = torch.full(
                (gg.shape[0], gg.shape[1]),
                self.atomic_model.descriptor.get_ntypes() - 1,
                device=gg.device,
                dtype=atype.dtype,
            )
            grid_nlist = build_directional_neighbor_list(
                gg,
                grid_type,
                extended_coord,
                extended_atype,
                self.get_rcut(),
                self.get_sel(),
                distinguish_types=(not self.mixed_types()),
            )
            model_predict_lower = self.forward_common_lower(
                extended_coord,
                extended_atype,
                nlist,
                grid=gg,
                grid_type=grid_type,
                grid_nlist=grid_nlist,
                mapping=mapping,
                do_atomic_virial=do_atomic_virial,
                fparam=fp,
                aparam=ap,
                charge_spin=charge_spin,
            )
            model_predict = communicate_extended_output(
                model_predict_lower,
                self.model_output_def(),
                mapping,
                do_atomic_virial=do_atomic_virial,
            )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

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
            charge_spin: torch.Tensor | None = None,
            grid: torch.Tensor | None = None,
            grid_type: torch.Tensor | None = None,
            grid_nlist: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            """Return model prediction. Lower interface that takes
            extended atomic coordinates and types, nlist, mapping, and the
            grid inputs as input, and returns the predictions on the grid
            points. The predictions are not reduced.

            The base parameters keep the base positional order (including
            ``extended_coord_corr``, which is accepted and ignored); the
            density-specific grid parameters are appended at the end and
            are required for this model.

            Parameters
            ----------
            extended_coord
                coodinates in extended region. nf x (nall x 3)
            extended_atype
                atomic type in extended region. nf x nall
            nlist
                neighbor list. nf x nloc x nsel.
            grid
                grid coordinates. nf x ngrid x 3
            grid_type
                type of the grid points. nf x ngrid
            grid_nlist
                directional neighbor list from grid points to atoms.
                nf x ngrid x nsel
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
            del extended_coord_corr
            nframes, nall = extended_atype.shape[:2]
            extended_coord = extended_coord.view(nframes, -1, 3)
            nlist = self.format_nlist(
                extended_coord, extended_atype, nlist, extra_nlist_sort=extra_nlist_sort
            )
            assert grid is not None
            assert grid_type is not None
            assert grid_nlist is not None
            cc_ext, gg, _, fp, ap, input_prec = self._input_type_cast(
                extended_coord, grid=grid, fparam=fparam, aparam=aparam
            )
            del extended_coord, grid, fparam, aparam
            atomic_ret = self.atomic_model.forward_common_atomic(
                cc_ext,
                extended_atype,
                nlist,
                mapping=mapping,
                fparam=fp,
                aparam=ap,
                comm_dict=comm_dict,
                grid=gg,
                grid_type=grid_type,
                grid_nlist=grid_nlist,
                charge_spin=charge_spin,
            )
            model_predict = fit_output_to_model_output(
                atomic_ret,
                self.atomic_output_def(),
                cc_ext,
                do_atomic_virial=do_atomic_virial,
                create_graph=self.training,
            )
            model_predict = self._output_type_cast(model_predict, input_prec)
            return model_predict

        def _input_type_cast(
            self,
            coord: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            grid: torch.Tensor | None = None,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            str,
        ]:
            """Cast the input data to global float type."""
            input_prec = self.reverse_precision_dict[coord.dtype]
            # dtype mismatch warnings are not emitted here: type checking
            # would not pass jit, so inputs are converted to coord prec anyway.
            _lst: list[torch.Tensor | None] = [
                vv.to(coord.dtype) if vv is not None else None
                for vv in [grid, box, fparam, aparam]
            ]
            grid, box, fparam, aparam = _lst
            assert grid is not None
            if (
                input_prec
                == self.reverse_precision_dict[self.global_pt_float_precision]
            ):
                return coord, grid, box, fparam, aparam, input_prec
            else:
                pp = self.global_pt_float_precision
                return (
                    coord.to(pp),
                    grid.to(pp),
                    box.to(pp) if box is not None else None,
                    fparam.to(pp) if fparam is not None else None,
                    aparam.to(pp) if aparam is not None else None,
                    input_prec,
                )

        def forward(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            do_atomic_virial: bool = False,
            charge_spin: torch.Tensor | None = None,
            grid: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            # directly call the forward_common method when no specific transform rule
            assert grid is not None
            return self.forward_common(
                coord,
                atype,
                box=box,
                fparam=fparam,
                aparam=aparam,
                do_atomic_virial=do_atomic_virial,
                charge_spin=charge_spin,
                grid=grid,
            )

        @torch.jit.export
        def forward_embedding(
            self,
            coord: torch.Tensor,
            atype: torch.Tensor,
            box: torch.Tensor | None = None,
            fparam: torch.Tensor | None = None,
            aparam: torch.Tensor | None = None,
            charge_spin: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            # the base implementation would call the grid-aware caster without
            # a grid; density models do not support embedding extraction
            raise NotImplementedError(
                "forward_embedding is not supported for density models."
            )

        def change_out_bias(
            self,
            merged: Any,
            bias_adjust_mode: str = "change-by-statistic",
        ) -> None:
            """Change the output bias according to the input data.

            Not supported for density models: the output is defined on grid
            points rather than atoms, so the standard bias adjustment (which
            would run a grid-less forward through the stat wrapper) does not
            apply. Overridden at the model level because the default
            ``change-by-statistic`` mode never reaches the atomic-level
            no-op.
            """
            log.warning(
                "change_out_bias is not supported for density models; skipping."
            )

    return CM
