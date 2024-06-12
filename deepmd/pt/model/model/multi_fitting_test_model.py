# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    ModelOutputDef,
    OutputVariableDef,
    get_deriv_name,
    get_reduce_name,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

from .dp_multi_fitting_model import (
    DPMultiFittingModel,
)
from .transform_output import (
    communicate_extended_output,
    fit_output_to_model_output,
)


class MultiFittingTestModel(DPMultiFittingModel):
    model_type = "multi_fitting_test"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def model_output_def(self):
        """Get the output def for the model.

        Default output is defined in deepmd.pt.model.model.make_multi_fitting_model.make_multi_fitting_model as:
            ModelOutputDef(self.atomic_output_def())
        Here we are interested in the sum of the components, so the redefinition of the model output is necessary.
        """
        outdef = FittingOutputDef(
            [
                OutputVariableDef(
                    "energy",
                    [1],
                    reduciable=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
            ]
        )
        return ModelOutputDef(outdef)

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if "energy_derv_r" in model_ret:
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if "energy_derv_c_redu" in model_ret:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord,
        extended_atype,
        nlist,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if "energy_derv_r" in model_ret:
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if "energy_derv_c_redu" in model_ret:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -3
                )
        return model_predict

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
        comm_dict: Optional[Dict[str, torch.Tensor]] = None,
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
        comm_dict
            The data needed for communication for parallel inference.

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
        atomic_ret = self.atomic_model.forward_common_atomic(
            cc_ext,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fp,
            aparam=ap,
            comm_dict=comm_dict,
        )
        model_predict = self.fit_output_to_model_output(
            atomic_ret,
            self.atomic_output_def(),
            cc_ext,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = self.output_type_cast(model_predict, input_prec)
        return model_predict

    def fit_output_to_model_output(
        self,
        fit_ret: Dict[str, torch.Tensor],
        fit_output_def: FittingOutputDef,
        coord_ext: torch.Tensor,
        do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        raw_model_ret = fit_output_to_model_output(
            fit_ret,
            fit_output_def,
            coord_ext,
            do_atomic_virial=do_atomic_virial,
        )
        l_fit_ret = []
        l_fit_ret_redu = []
        l_fit_ret_derv_r = []
        l_fit_ret_derv_c = []
        l_fit_ret_derv_c_redu = []
        for kk in fit_ret.keys():
            if kk != "mask":
                kk_redu = get_reduce_name(kk)
                kk_derv_r, kk_derv_c = get_deriv_name(kk)
                # kk_derv_c + "_redu"
                if kk in raw_model_ret:
                    l_fit_ret.append(raw_model_ret[kk])
                if kk_redu in raw_model_ret:
                    l_fit_ret_redu.append(raw_model_ret[kk_redu])
                if kk_derv_r in raw_model_ret:
                    l_fit_ret_derv_r.append(raw_model_ret[kk_derv_r])
                if kk_derv_c in raw_model_ret:
                    l_fit_ret_derv_c.append(raw_model_ret[kk_derv_c])
                if kk_derv_c + "_redu" in raw_model_ret:
                    l_fit_ret_derv_c_redu.append(raw_model_ret[kk_derv_c + "_redu"])
        # energy energy_redu energy_derv_r energy_derv_c energy_derv_c_redu
        model_output_def = "energy"
        model_output_def_redu = get_reduce_name(model_output_def)
        model_output_def_derv_r, model_output_def_derv_c = get_deriv_name(
            model_output_def
        )
        model_ret = {}
        if len(l_fit_ret) > 0:
            model_ret[model_output_def] = torch.sum(torch.stack(l_fit_ret), dim=0)
        if len(l_fit_ret_redu) > 0:
            model_ret[model_output_def_redu] = torch.sum(
                torch.stack(l_fit_ret_redu), dim=0
            )
        if len(l_fit_ret_derv_r) > 0:
            model_ret[model_output_def_derv_r] = torch.sum(
                torch.stack(l_fit_ret_derv_r), dim=0
            )
        if len(l_fit_ret_derv_c) > 0:
            model_ret[model_output_def_derv_c] = torch.sum(
                torch.stack(l_fit_ret_derv_c), dim=0
            )
        if len(l_fit_ret_derv_c_redu) > 0:
            model_ret[model_output_def_derv_c + "_redu"] = torch.sum(
                torch.stack(l_fit_ret_derv_c_redu), dim=0
            )
        return model_ret
