# SPDX-License-Identifier: LGPL-3.0-or-later
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

DPEnergyModel_ = make_model(DPEnergyAtomicModel)


@BaseModel.register("ener")
class EnergyModel(DPModelCommon, DPEnergyModel_):
    r"""Energy model that predicts total energy and derived quantities.

    The model takes atomic energies from the atomic model and computes
    global properties by reduction and differentiation:

    **Reduction** (total energy):

    .. math::
        E = \sum_{i=1}^{N} E^i,

    where :math:`E^i` is the atomic energy from the atomic model.

    **Differentiation** (forces and virials):

    .. math::
        \mathbf{F}_i = -\frac{\partial E}{\partial \mathbf{r}_i},

    .. math::
        \boldsymbol{\Xi} = -\sum_{i=1}^{N} \frac{\partial E}{\partial \mathbf{r}_i} \otimes \mathbf{r}_i
        = \sum_{i=1}^{N} \mathbf{r}_i \otimes \mathbf{F}_i,

    where :math:`\mathbf{F}_i` is the force on atom :math:`i` and
    :math:`\boldsymbol{\Xi}` is the virial tensor.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        DPModelCommon.__init__(self)
        DPEnergyModel_.__init__(self, *args, **kwargs)
        self._enable_hessian = False
        self.hess_fitting_def = None

    def enable_hessian(self) -> None:
        self.hess_fitting_def = deepcopy(self.atomic_output_def())
        self.hess_fitting_def["energy"].r_hessian = True
        self._enable_hessian = True

    def atomic_output_def(self) -> FittingOutputDef:
        if self._enable_hessian:
            return self.hess_fitting_def
        return super().atomic_output_def()

    def call(
        self,
        coord: Array,
        atype: Array,
        box: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
        model_ret = self.call_common(
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
        if self.do_grad_r("energy") and model_ret["energy_derv_r"] is not None:
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy") and model_ret["energy_derv_c_redu"] is not None:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret["energy_derv_c"] is not None:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        if self._enable_hessian and model_ret.get("energy_derv_r_derv_r") is not None:
            model_predict["hessian"] = model_ret["energy_derv_r_derv_r"].squeeze(-3)
        return model_predict

    def call_lower(
        self,
        extended_coord: Array,
        extended_atype: Array,
        nlist: Array,
        mapping: Array | None = None,
        fparam: Array | None = None,
        aparam: Array | None = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, Array]:
        model_ret = self.call_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
        )
        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy") and model_ret.get("energy_derv_r") is not None:
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy") and model_ret.get("energy_derv_c_redu") is not None:
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial and model_ret.get("energy_derv_c") is not None:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -2
                )
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def translated_output_def(self) -> dict[str, Any]:
        """Get the translated output definition.

        Maps internal output names to user-facing names, e.g.
        ``energy_redu`` -> ``energy``, ``energy_derv_r`` -> ``force``.
        """
        out_def_data = self.model_output_def().get_data()
        output_def = {
            "atom_energy": out_def_data["energy"],
            "energy": out_def_data["energy_redu"],
        }
        if self.do_grad_r("energy"):
            output_def["force"] = out_def_data["energy_derv_r"]
            output_def["force"].squeeze(-2)
        if self.do_grad_c("energy"):
            output_def["virial"] = out_def_data["energy_derv_c_redu"]
            output_def["virial"].squeeze(-2)
            output_def["atom_virial"] = out_def_data["energy_derv_c"]
            output_def["atom_virial"].squeeze(-2)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        if self._enable_hessian:
            output_def["hessian"] = out_def_data["energy_derv_r_derv_r"]
        return output_def
