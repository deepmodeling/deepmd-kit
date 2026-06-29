# SPDX-License-Identifier: LGPL-3.0-or-later
"""SeZM model variant for invariant property prediction."""

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)

from .sezm_model import (
    SeZMModel,
)


@BaseModel.register("SeZMProperty")
@BaseModel.register("sezm_property")
@BaseModel.register("DPA4Property")
@BaseModel.register("dpa4_property")
class SeZMPropertyModel(SeZMModel):
    """SeZM sparse-edge model for invariant property fitting.

    The descriptor path, sparse edge construction, compile cache, and type
    handling are inherited from :class:`SeZMModel`. The property variant only
    changes the readout contract: fitting outputs are reduced by their
    ``OutputVariableDef`` and no conservative force or virial derivative is
    constructed.
    """

    model_type = "SeZMProperty"

    def __init__(
        self,
        *args: Any,
        bridging_method: str = "none",
        **kwargs: Any,
    ) -> None:
        if str(bridging_method).upper() != "NONE":
            raise ValueError(
                "SeZM property fitting does not support analytical bridging "
                "potentials; set `bridging_method` to `none`."
            )
        super().__init__(*args, bridging_method=bridging_method, **kwargs)

    def _translate_property_output(
        self,
        model_ret: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Translate lower property keys to public prediction keys."""
        var_name = self.get_var_name()
        model_predict = {
            f"atom_{var_name}": model_ret[var_name],
            var_name: model_ret[f"{var_name}_redu"],
        }
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        force_input: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run invariant property prediction from coordinates."""
        del do_atomic_virial, force_input, noise_mask
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
        )
        return self._translate_property_output(model_ret)

    def forward_lower(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_scatter_index: torch.Tensor,
        edge_mask: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extended_atype: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run lower-interface invariant property prediction."""
        del do_atomic_virial
        model_ret = self.forward_common_lower(
            coord,
            atype,
            edge_index,
            edge_vec,
            edge_scatter_index,
            edge_mask,
            fparam=fparam,
            aparam=aparam,
            comm_dict=comm_dict,
            extended_atype=extended_atype,
            charge_spin=charge_spin,
        )
        return self._translate_property_output(model_ret)

    def core_compute(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_scatter_index: torch.Tensor,
        edge_mask: torch.Tensor,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
        extended_atype: torch.Tensor | None = None,
        extended_coord_corr: torch.Tensor | None = None,
        embedding_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute property outputs through the SeZM forward-only graph."""
        return super().core_compute(
            coord,
            atype,
            edge_index,
            edge_vec,
            edge_scatter_index,
            edge_mask,
            fparam=fparam,
            aparam=aparam,
            charge_spin=charge_spin,
            comm_dict=comm_dict,
            extended_atype=extended_atype,
            extended_coord_corr=extended_coord_corr,
            embedding_only=embedding_only,
            conservative=False,
        )

    def _inductor_compile_options(self) -> dict[str, Any]:
        """Augment the shared Inductor options for the property compile path.

        The non-conservative property backward graph triggers a TorchInductor
        CPU codegen bug: a scalar ``where``/blendv is emitted as
        ``decltype(scalar)::blendv(...)``, which the host C++ compiler rejects.
        Forcing scalar CPU codegen (``cpp.simdlen = 0``) selects the path that
        never emits the vectorized blendv. ``cpp.*`` options affect only the CPU
        backend, so CUDA/Triton lowering -- the actual ``use_compile`` deployment
        target -- is unchanged.
        """
        options = super()._inductor_compile_options()
        options["cpp.simdlen"] = 0
        return options

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        """Return public output definitions for property prediction."""
        out_def_data = self.model_output_def().get_data()
        var_name = self.get_var_name()
        output_def = {
            f"atom_{var_name}": out_def_data[var_name],
            var_name: out_def_data[f"{var_name}_redu"],
        }
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def get_task_dim(self) -> int:
        """Return the property output dimension."""
        return int(self.get_fitting_net().dim_out)

    def get_intensive(self) -> bool:
        """Return whether the reduced property is intensive."""
        return bool(self.model_output_def()[self.get_var_name()].intensive)

    def get_var_name(self) -> str:
        """Return the fitted property name."""
        return str(self.get_fitting_net().var_name)
