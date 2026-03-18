# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Any, Optional

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.descriptor.base_descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.task.sog_energy_fitting import (
    SOGEnergyFittingNet,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

from .base_atomic_model import (
    BaseAtomicModel,
)


@BaseAtomicModel.register("energy_sog")
class SOGEnergyAtomicModel(BaseAtomicModel):
    """Energy model using a dedicated SOG energy fitting net.

    The SOG energy fitting net combines a short-range invariant fitting
    and a long-range correction derived from another invariant fitting.
    This avoids requiring a user-defined property name in the dataset.
    """

    def __init__(
        self,
        descriptor: Any,
        type_map: list[str],
        sog_energy_fitting: Optional[SOGEnergyFittingNet] = None,
        fitting: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(type_map, **kwargs)
        if sog_energy_fitting is None:
            sog_energy_fitting = fitting
        if not isinstance(sog_energy_fitting, SOGEnergyFittingNet):
            raise TypeError(
                "sog_energy_fitting must be an instance of SOGEnergyFittingNet"
            )

        self.descriptor = descriptor
        self.fitting_net = sog_energy_fitting
        # self.sog_energy_fitting = self.fitting_net
        self.type_map = type_map
        self.ntypes = len(type_map)
        self.rcut = self.descriptor.get_rcut()
        self.sel = self.descriptor.get_sel()

        super().init_out_stat()

        self.enable_eval_descriptor_hook = False
        self.enable_eval_fitting_last_layer_hook = False
        self.eval_descriptor_list: list[torch.Tensor] = []
        self.eval_fitting_last_layer_list: list[torch.Tensor] = []

    @torch.jit.export
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
        self.fitting_net.set_case_embd(case_idx)

    def get_dim_fparam(self) -> int:
        return self.fitting_net.get_dim_fparam()

    def has_default_fparam(self) -> bool:
        return self.fitting_net.has_default_fparam()

    def get_default_fparam(self) -> Optional[torch.Tensor]:
        return self.fitting_net.get_default_fparam()

    def get_dim_aparam(self) -> int:
        return self.fitting_net.get_dim_aparam()

    def get_sel_type(self) -> list[int]:
        return self.fitting_net.get_sel_type()

    def is_aparam_nall(self) -> bool:
        return False

    def set_eval_descriptor_hook(self, enable: bool) -> None:
        self.enable_eval_descriptor_hook = enable
        self.eval_descriptor_list.clear()

    def eval_descriptor(self) -> torch.Tensor:
        return torch.concat(self.eval_descriptor_list)

    def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
        self.enable_eval_fitting_last_layer_hook = enable
        self.fitting_net.set_return_middle_output(enable)
        self.eval_fitting_last_layer_list.clear()

    def eval_fitting_last_layer(self) -> torch.Tensor:
        return torch.concat(self.eval_fitting_last_layer_list)

    def forward_atomic(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        nframes, nloc, _ = nlist.shape
        atype = extended_atype[:, :nloc]
        if self.do_grad_r() or self.do_grad_c():
            extended_coord.requires_grad_(True)

        descriptor_comm_dict = comm_dict
        if comm_dict is not None and "send_list" not in comm_dict:
            descriptor_comm_dict = None

        descriptor, rot_mat, g2, h2, sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            comm_dict=descriptor_comm_dict,
        )
        assert descriptor is not None
        if self.enable_eval_descriptor_hook:
            self.eval_descriptor_list.append(descriptor.detach())

        energy_ret = self.fitting_net(
            descriptor,
            atype,
            gr=rot_mat,
            g2=g2,
            h2=h2,
            fparam=fparam,
            aparam=aparam,
            coord=extended_coord[:, :nloc, :],
            box=(
                comm_dict["box"].view(nframes, 3, 3)
                if comm_dict is not None and "box" in comm_dict
                else None
            ),
        )

        if self.enable_eval_fitting_last_layer_hook and "middle_output" in energy_ret:
            self.eval_fitting_last_layer_list.append(
                energy_ret["middle_output"].detach()
            )

        ret = {
            "energy": energy_ret["energy"],
        }
        if "middle_output" in energy_ret:
            ret["middle_output"] = energy_ret["middle_output"]
        return ret

    def apply_out_stat(
        self,
        ret: dict[str, torch.Tensor],
        atype: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out_bias, out_std = self._fetch_out_stat(self.bias_keys)
        for kk in self.bias_keys:
            ret[kk] = ret[kk] + out_bias[kk][atype]
        return ret

    def compute_or_load_stat(
        self,
        sampled_func: Any,
        stat_file_path: Optional[Any] = None,
        compute_or_load_out_stat: bool = True,
        preset_observed_type: Optional[list[str]] = None,
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

        self._collect_and_set_observed_type(
            wrapped_sampler, stat_file_path, preset_observed_type
        )

    def compute_fitting_input_stat(
        self,
        sample_merged: Any,
        stat_file_path: Optional[Any] = None,
    ) -> None:
        self.fitting_net.compute_input_stats(
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
                "type": "energy_sog",
                "type_map": self.type_map,
                "descriptor": self.descriptor.serialize(),
                "sog_energy_fitting": self.fitting_net.serialize(),
            }
        )
        return dd

    @classmethod
    def deserialize(cls, data: dict) -> "SOGEnergyAtomicModel":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        descriptor_obj = BaseDescriptor.deserialize(data.pop("descriptor"))
        sog_energy_fitting_obj = SOGEnergyFittingNet.deserialize(
            data.pop("sog_energy_fitting")
        )
        obj = cls(
            descriptor=descriptor_obj,
            sog_energy_fitting=sog_energy_fitting_obj,
            **data,
        )
        return obj
