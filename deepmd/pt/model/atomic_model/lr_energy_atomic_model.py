# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)
from collections.abc import Iterable

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
    EnergyFittingNetDirect,
    InvarFitting,
)
from deepmd.pt.model.task.property import (
    PropertyFittingNet,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


@BaseAtomicModel.register("energy_lr")
class LREnergyAtomicModel(BaseAtomicModel):
    """Energy model with an auxiliary property-driven correction.

    This model shares one descriptor with two fitting nets:
    - ``energy_fitting`` predicts the primary atomic energy/force term.
    - ``property_fitting`` predicts an auxiliary per-atom property ``q``.
    The property is then passed through a small trainable correction head
    to generate an additive energy (and resulting force) correction.
    """

    def __init__(
        self,
        descriptor: BaseDescriptor,
        energy_fitting: InvarFitting,
        property_fitting: PropertyFittingNet,
        type_map: list[str],
        correction_hidden: Iterable[int] | None = None,
        correction_activation: str = "tanh",
        **kwargs: Any,
    ) -> None:
        super().__init__(type_map, **kwargs)
        if not (
            isinstance(energy_fitting, EnergyFittingNet)
            or isinstance(energy_fitting, EnergyFittingNetDirect)
            or isinstance(energy_fitting, InvarFitting)
        ):
            raise TypeError(
                "energy_fitting must be an energy-like InvarFitting for LREnergyAtomicModel"
            )
        if not isinstance(property_fitting, PropertyFittingNet):
            raise TypeError(
                "property_fitting must be an instance of PropertyFittingNet for LREnergyAtomicModel"
            )

        if energy_fitting.get_dim_fparam() != property_fitting.get_dim_fparam():
            raise ValueError(
                "energy_fitting and property_fitting must share the same dim_fparam"
            )
        if energy_fitting.get_dim_aparam() != property_fitting.get_dim_aparam():
            raise ValueError(
                "energy_fitting and property_fitting must share the same dim_aparam"
            )

        self.descriptor = descriptor
        self.energy_fitting = energy_fitting
        self.property_fitting = property_fitting
        self.property_name = property_fitting.var_name
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()
        self.correction_activation = correction_activation
        hidden = (
            list(correction_hidden)
            if correction_hidden is not None
            else [property_fitting.dim_out]
        )
        self.correction_hidden = hidden
        self.correction_head = self._build_correction_head(
            property_fitting.dim_out, hidden, correction_activation
        )
        super().init_out_stat()

        self.enable_eval_descriptor_hook = False
        self.enable_eval_fitting_last_layer_hook = False
        self.eval_descriptor_list: list[torch.Tensor] = []
        self.eval_fitting_last_layer_list: list[torch.Tensor] = []

    @staticmethod
    def _build_correction_head(
        input_dim: int, hidden: list[int], activation: str
    ) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        last = input_dim
        act_factory = getattr(torch.nn, activation.capitalize(), torch.nn.Tanh)
        for width in hidden:
            layers.append(torch.nn.Linear(last, width))
            layers.append(act_factory())
            last = width
        layers.append(torch.nn.Linear(last, 1))
        return torch.nn.Sequential(*layers)

    def fitting_output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    name="energy",
                    shape=[1],
                    reducible=True,
                    r_differentiable=True,
                    c_differentiable=True,
                ),
                OutputVariableDef(
                    name=self.property_name,
                    shape=[self.property_fitting.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=self.property_fitting.get_intensive(),
                ),
            ]
        )

    def get_rcut(self) -> float:
        return self.rcut

    def get_sel(self) -> list[int]:
        return self.sel

    def mixed_types(self) -> bool:
        return self.descriptor.mixed_types()

    def has_message_passing(self) -> bool:
        return self.descriptor.has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        return self.descriptor.need_sorted_nlist_for_lower()

    def set_case_embd(self, case_idx: int) -> None:
        self.energy_fitting.set_case_embd(case_idx)
        self.property_fitting.set_case_embd(case_idx)

    def get_dim_fparam(self) -> int:
        return self.energy_fitting.get_dim_fparam()

    def has_default_fparam(self) -> bool:
        return self.energy_fitting.has_default_fparam()

    def get_default_fparam(self) -> torch.Tensor | None:
        return self.energy_fitting.get_default_fparam()

    def get_dim_aparam(self) -> int:
        return self.energy_fitting.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        return self.energy_fitting.get_sel_type()

    def is_aparam_nall(self) -> bool:
        return False

    def set_eval_descriptor_hook(self, enable: bool) -> None:
        self.enable_eval_descriptor_hook = enable
        self.eval_descriptor_list.clear()

    def eval_descriptor(self) -> torch.Tensor:
        return torch.concat(self.eval_descriptor_list)

    def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
        self.enable_eval_fitting_last_layer_hook = enable
        self.energy_fitting.set_return_middle_output(enable)
        self.eval_fitting_last_layer_list.clear()

    def eval_fitting_last_layer(self) -> torch.Tensor:
        return torch.concat(self.eval_fitting_last_layer_list)

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        comm_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        nframes, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)
        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=comm_dict,
        )
        assert descriptor is not None
        if self.enable_eval_descriptor_hook:
            self.eval_descriptor_list.append(descriptor.detach())

        energy_ret = self.energy_fitting(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )
        prop_ret = self.property_fitting(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
        )

        if self.enable_eval_fitting_last_layer_hook and "middle_output" in energy_ret:
            self.eval_fitting_last_layer_list.append(
                energy_ret["middle_output"].detach()
            )

        q_val = prop_ret[self.property_name]
        corr_energy = self.correction_head(q_val)
        total_energy = energy_ret["energy"] + corr_energy

        return {
            "energy": total_energy,
            self.property_name: q_val,
        }

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        for kk in self.bias_keys:
            if kk == self.property_name:
                ret[kk] = ret[kk] * out_std[kk][atype] + out_bias[kk][atype]
            else:
                ret[kk] = ret[kk] + out_bias[kk][atype]
        return ret

    def compute_or_load_stat(
        self,
        sampled_func: Any,
        stat_file_path: Any | None = None,
        compute_or_load_out_stat: bool = True,
    ) -> None:
        if stat_file_path is not None and self.type_map is not None:
            stat_file_path /= " ".join(self.type_map)

        def wrapped_sampler() -> list[dict]:
            sampled = sampled_func()
            if self.pair_excl is not None:
                pair_exclude_types = self.pair_excl.get_exclude_types()
                for sample in sampled:
                    sample["pair_exclude_types"] = list(pair_exclude_types)
            if self.atom_excl is not None:
                atom_exclude_types = self.atom_excl.get_exclude_types()
                for sample in sampled:
                    sample["atom_exclude_types"] = list(atom_exclude_types)
            if (
                "find_fparam" not in sampled[0]
                and "fparam" not in sampled[0]
                and self.has_default_fparam()
            ):
                default_fparam = self.get_default_fparam()
                for sample in sampled:
                    nframe = sample["atype"].shape[0]
                    sample["fparam"] = default_fparam.repeat(nframe, 1)
            return sampled

        self.descriptor.compute_input_stats(wrapped_sampler, stat_file_path)
        self.compute_fitting_input_stat(wrapped_sampler, stat_file_path)
        if compute_or_load_out_stat:
            self.compute_or_load_out_stat(wrapped_sampler, stat_file_path)

    def compute_fitting_input_stat(
        self,
        sample_merged: Any,
        stat_file_path: Any | None = None,
    ) -> None:
        self.energy_fitting.compute_input_stats(
            sample_merged,
            protection=self.data_stat_protect,
            stat_file_path=stat_file_path,
        )
        self.property_fitting.compute_input_stats(
            sample_merged,
            protection=self.data_stat_protect,
            stat_file_path=stat_file_path,
        )

    def serialize(self) -> dict:
        dd = BaseAtomicModel.serialize(self)
        dd.update(
            {
                "@class": "Model",
                "@version": 1,
                "type": "energy_q_aug",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "energy_fitting": self.energy_fitting.serialize(),
                "property_fitting": self.property_fitting.serialize(),
                "correction_hidden": self.correction_hidden,
                "correction_activation": self.correction_activation,
                "@variables": {
                    **dd.get("@variables", {}),
                    "correction_head": {
                        k: to_numpy_array(v)
                        for k, v in self.correction_head.state_dict().items()
                    },
                },
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "LREnergyAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        variables = data.pop("@variables", {})
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        energy_fitting_obj = InvarFitting.deserialize(data.pop("energy_fitting"))
        property_fitting_obj = PropertyFittingNet.deserialize(
            data.pop("property_fitting")
        )
        correction_hidden = data.pop("correction_hidden", None)
        correction_activation = data.pop("correction_activation", "tanh")
        obj = cls(
            descriptor_obj,
            energy_fitting_obj,
            property_fitting_obj,
            correction_hidden=correction_hidden,
            correction_activation=correction_activation,
            **data,
        )
        correction_state = variables.get("correction_head", None)
        if correction_state is not None:
            obj.correction_head.load_state_dict(
                {k: to_torch_tensor(v) for k, v in correction_state.items()}
            )
        return obj
