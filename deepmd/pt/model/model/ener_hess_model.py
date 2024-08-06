# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Dict,
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)
from .make_hessian_model import (
    make_hessian_model,
)  # anchor added
import numpy as np  # anchor added


DPEnergyModel_ = make_model(DPEnergyAtomicModel)
DPEnergyModel_ = make_hessian_model(DPEnergyModel_)  # anchor added


@BaseModel.register("ener_hess")
class EnergyHessianModel(DPModelCommon, DPEnergyModel_):  # anchor created
    model_type = "ener_hess"

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)

    def translated_output_def(self):
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": deepcopy(out_def_data["energy"]),
            "energy": deepcopy(out_def_data["energy_redu"]),
        }
        if self.do_grad_r("energy"):
            output_def["force"] = deepcopy(out_def_data["energy_derv_r"])
            output_def["force"].squeeze(-2)
            # print(f"output_def['force'] in translated_output_def: {output_def['force']}")  # anchor added
        if self.do_grad_c("energy"):
            output_def["virial"] = deepcopy(out_def_data["energy_derv_c_redu"])
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = deepcopy(out_def_data["energy_derv_c"])
            output_def["atom_virial"].squeeze(-3)
        output_def["hessian"] = deepcopy(out_def_data["energy_derv_r_derv_r"])  # anchor added
        # output_def["hessian"].squeeze(-2)  # anchor added
        if "mask" in out_def_data:
            output_def["mask"] = deepcopy(out_def_data["mask"])
        return output_def

    def forward(
            self,
            coord,
            atype,
            box: Optional[torch.Tensor] = None,
            fparam: Optional[torch.Tensor] = None,
            aparam: Optional[torch.Tensor] = None,
            do_atomic_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self.requires_hessian("energy")  # anchor added
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            # ener_tmp = model_predict["energy"]  # anchor added
            # print(f"energy of {ener_tmp.size()} and grad {ener_tmp.requires_grad} in forward: {ener_tmp}")  # anchor added
            # coord_flat = coord.view(-1)  # anchor added
            # coord_flat.requires_grad_(True)  # anchor added
            # print(f"coord of {coord_flat.size()} and grad {coord_flat.requires_grad} in forward: {coord_flat}")  # anchor added
            # grad_tmp = torch.autograd.grad(ener_tmp, coord_flat, create_graph=True)[0]  # anchor added
            # print(f"grad of {grad_tmp.size()} and grad {grad_tmp.requires_grad} in forward: {grad_tmp}")  # anchor added
            if self.do_grad_r("energy"):
                model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
                # force_tmp = torch.reshape(model_predict["force"], (-1,))  # anchor added
                # print(f"model_predict['force'] of {force_tmp.size()} and {force_tmp.requires_grad} in forward: {force_tmp}")  # anchor added
                # print(f"grad of force in forward: {model_predict['force'].grad}")
                # n_hess_elem = force_tmp.size()[0]  # anchor added
                # hess_tmp = torch.zeros(n_hess_elem, n_hess_elem, device=force_tmp.device, dtype=force_tmp.dtype)  # anchor added
                # coord_tmp = torch.reshape(coord, (-1,))  # anchor added
                # coord_tmp.requires_grad_(True)  # anchor added
                # print(f"coord of {coord_tmp.size()} and {coord_tmp.requires_grad} in forward: {coord_tmp}")  # anchor added
                # for i in range(n_hess_elem):  # anchor added
                #     grad2 = torch.autograd.grad(force_tmp[i], coord_tmp, retain_graph=True, allow_unused=True)[0]
                #     # if grad2 is None:
                #     #     grad2 = torch.zeros_like(coord_tmp)
                #     hess_tmp[i] = grad2
                # print(f"hess_tmp of {hess_tmp.size()} in EnergyHessianModel.forward: {hess_tmp}")  # anchor added
                # anchor: tensor.squeeze(dim) remove -dim if its 1, or keep dim ((2,1,3)-->(2,3), (1,2,3)-->(1,2,3))
            # if self.do_grad_r_hess("energy"):  # tbd; anchor created
            #     model_predict["hessian"] = model_ret["kk"]  # kk is m_h_m value
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(
                        -3
                    )
            else:
                model_predict["force"] = model_ret["dforce"]
            if "mask" in model_ret:
                model_predict["mask"] = model_ret["mask"]
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"]  # anchor added squeeze?
            # print(f"model_predict['hessian'] before squeeze: size of {model_predict['hessian'].size()}, "
            #       f"nan of {np.sum(np.isnan(model_predict['hessian'].cpu().detach().numpy()))}")  # anchor added
            # if np.sum(np.isnan(model_predict['hessian'].cpu().detach().numpy())) == 135:  # anchor added
            #     np.save("./hess_45x45.npy", model_predict['hessian'].cpu().detach().numpy())
            #     print("hess_45x45.npy is saved")
            # if np.sum(np.isnan(model_predict['hessian'].cpu().detach().numpy())) == 162:  # anchor added
            #     np.save("./hess_54x54.npy", model_predict['hessian'].cpu().detach().numpy())
            #     print("hess_54x54.npy is saved")
            model_predict["hessian"].squeeze(-2)  # anchor added: no need to squeeze
            # print(f"model_predict['hessian'] after squeeze: size of {model_predict['hessian'].size()}, "
            #       f"nan of {np.sum(np.isnan(model_predict['hessian'].cpu().detach().numpy()))}")  # anchor added
        else:
            model_predict = model_ret
            model_predict["updated_coord"] += coord
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
        if self.get_fitting_net() is not None:
            model_predict = {}
            model_predict["atom_energy"] = model_ret["energy"]
            model_predict["energy"] = model_ret["energy_redu"]
            if self.do_grad_r("energy"):
                model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
            if self.do_grad_c("energy"):
                model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
                if do_atomic_virial:
                    model_predict["extended_virial"] = model_ret[
                        "energy_derv_c"
                    ].squeeze(-3)
            else:
                assert model_ret["dforce"] is not None
                model_predict["dforce"] = model_ret["dforce"]
        else:
            model_predict = model_ret
        return model_predict

