# SPDX-License-Identifier: LGPL-3.0-or-later
"""Group-level property model for end-to-end grouped embeddings."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable

import torch

from deepmd.dpmodel import (
    ModelOutputDef,
)
from deepmd.pt.model.model.dp_model import (
    DPModelCommon,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task.group_property import (
    GroupPropertyFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.grouped import (
    GROUP_ID_KEY,
    GROUP_WEIGHT_KEY,
    POOL_MASK_KEY,
    normalize_group_id_tensor,
    normalize_pool_mask_tensor,
    normalize_weight_tensor,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.utils.path import (
    DPPath,
)


@BaseModel.register("group_property")
class GroupPropertyModel(DPModelCommon, BaseModel):
    """Predict one property per weighted group of frames."""

    model_type = "group_property"

    def __init__(
        self,
        descriptor: Any,
        fitting: Any,
        type_map: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        BaseModel.__init__(self)
        if not isinstance(fitting, GroupPropertyFittingNet):
            raise TypeError(
                "fitting must be a GroupPropertyFittingNet for GroupPropertyModel"
            )
        self.descriptor = descriptor
        self.fitting_net = fitting
        self.type_map = list(type_map or [])
        self.atomic_model = SimpleNamespace(
            descriptor=self.descriptor,
            fitting_net=self.fitting_net,
            get_dim_fparam=self.fitting_net.get_dim_fparam,
            has_default_fparam=self.fitting_net.has_default_fparam,
            get_default_fparam=self.fitting_net.get_default_fparam,
            get_dim_aparam=self.fitting_net.get_dim_aparam,
        )

    def model_output_def(self) -> ModelOutputDef:
        return ModelOutputDef(self.fitting_net.output_def())

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
        return self.model_output_def().get_data()

    def compute_or_load_stat(
        self,
        sampled_func: Callable[[], Any],
        stat_file_path: DPPath | None = None,
        preset_observed_type: list[str] | None = None,
    ) -> None:
        if stat_file_path is not None and self.type_map is not None:
            stat_file_path /= " ".join(self.type_map)
        self.descriptor.compute_input_stats(sampled_func, stat_file_path)

    @torch.jit.export
    def get_observed_type_list(self) -> list[str]:
        return self.type_map

    def get_descriptor(self):  # noqa: ANN201
        return self.descriptor

    def get_fitting_net(self):  # noqa: ANN201
        return self.fitting_net

    @torch.jit.export
    def get_task_dim(self) -> int:
        return self.fitting_net.task_dim

    @torch.jit.export
    def get_var_name(self) -> str:
        return self.fitting_net.var_name

    def get_rcut(self) -> float:
        return self.descriptor.get_rcut()

    def get_sel(self) -> list[int]:
        return self.descriptor.get_sel()

    def mixed_types(self) -> bool:
        mixed_types = getattr(self.descriptor, "mixed_types", None)
        return True if mixed_types is None else mixed_types()

    def get_type_map(self) -> list[str]:
        return self.type_map

    def get_dim_fparam(self) -> int:
        return self.fitting_net.get_dim_fparam()

    def has_default_fparam(self) -> bool:
        return self.fitting_net.has_default_fparam()

    def get_default_fparam(self) -> torch.Tensor | None:
        return self.fitting_net.get_default_fparam()

    def get_dim_aparam(self) -> int:
        return self.fitting_net.get_dim_aparam()

    @torch.jit.export
    def has_chg_spin_ebd(self) -> bool:
        return bool(getattr(self.descriptor, "add_chg_spin_ebd", False))

    @torch.jit.export
    def has_default_chg_spin(self) -> bool:
        if not self.has_chg_spin_ebd():
            return False
        return self.descriptor.has_default_chg_spin()

    @torch.jit.export
    def get_default_chg_spin(self) -> torch.Tensor | None:
        if self.has_default_chg_spin():
            return self.descriptor.get_default_chg_spin()
        return None

    @torch.jit.export
    def has_message_passing(self) -> bool:
        has_message_passing = getattr(self.descriptor, "has_message_passing", None)
        return False if has_message_passing is None else has_message_passing()

    def need_sorted_nlist_for_lower(self) -> bool:
        need_sorted = getattr(self.descriptor, "need_sorted_nlist_for_lower", None)
        return False if need_sorted is None else need_sorted()

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
        group_id: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        pool_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del fparam, aparam, do_atomic_virial
        coord = coord.to(env.GLOBAL_PT_FLOAT_PRECISION)
        if coord.dim() == 2:
            coord = coord.view(coord.shape[0], -1, 3)
        atype = atype.long()
        box = None if box is None else box.to(env.GLOBAL_PT_FLOAT_PRECISION)
        nframes, natoms = atype.shape

        extended_coord, extended_atype, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            atype,
            self.get_rcut(),
            self.get_sel(),
            mixed_types=True,
            box=box,
        )
        descriptor, _rot_mat, _g2, _h2, _sw = self.descriptor(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            charge_spin=charge_spin,
        )
        if descriptor is None:
            raise RuntimeError("Descriptor returned None for group_property.")
        descriptor = descriptor[:, :natoms, :]

        if pool_mask is None:
            pool_mask = torch.ones(
                (nframes, natoms), dtype=descriptor.dtype, device=descriptor.device
            )
        else:
            pool_mask = normalize_pool_mask_tensor(pool_mask, nframes, natoms).to(
                descriptor.device, descriptor.dtype
            )
        denom = pool_mask.sum(dim=1).clamp_min(1.0)
        frame_embedding = (descriptor * pool_mask[:, :, None]).sum(dim=1) / denom[:, None]

        if group_id is None:
            group_id = torch.arange(nframes, dtype=torch.long, device=descriptor.device)
        else:
            group_id = normalize_group_id_tensor(group_id, nframes).to(descriptor.device)
        if weight is None:
            weight = torch.ones(nframes, dtype=descriptor.dtype, device=descriptor.device)
        else:
            weight = normalize_weight_tensor(weight, nframes).to(
                descriptor.device, descriptor.dtype
            )
        weight = weight.detach()

        group_order, inverse = torch.unique(group_id, sorted=True, return_inverse=True)
        group_embedding = torch.zeros(
            (group_order.shape[0], frame_embedding.shape[1]),
            dtype=frame_embedding.dtype,
            device=frame_embedding.device,
        )
        group_embedding.index_add_(0, inverse, frame_embedding * weight[:, None])
        prediction = self.fitting_net(group_embedding)
        return {
            self.get_var_name(): prediction,
            "group_id": group_order,
            "frame_group_id": group_id,
            "group_inverse": inverse,
            "frame_embedding": frame_embedding,
            GROUP_WEIGHT_KEY: weight,
            POOL_MASK_KEY: pool_mask,
        }
