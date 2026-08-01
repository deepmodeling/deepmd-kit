# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from copy import (
    deepcopy,
)
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.output_def import (
    OutputVariableDef,
)
from deepmd.pt.model.atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

from .dp_model import (
    DPModelCommon,
)
from .make_model import (
    make_model,
)

log = logging.getLogger(__name__)

DPLinearModel_ = make_model(LinearEnergyAtomicModel)


def _get_linear_model_index(model_key: str) -> int:
    if not model_key.startswith("model_"):
        raise RuntimeError(f"Unknown linear model key {model_key}!")
    return int(model_key.removeprefix("model_"))


def normalize_linear_model_type_map(model_params: dict[str, Any]) -> None:
    """Fill the linear model type_map from sub-models when needed."""
    if "type_map" in model_params:
        return
    for idx, sub_model_params in enumerate(model_params["models"]):
        if "type_map" not in sub_model_params:
            raise ValueError(
                f"Linear sub-model {idx} must define type_map when "
                "linear_ener has no top-level type_map."
            )
    first_type_map = model_params["models"][0]["type_map"]
    for idx, sub_model_params in enumerate(model_params["models"][1:], start=1):
        if sub_model_params["type_map"] != first_type_map:
            raise ValueError(
                f"Linear sub-model {idx} type_map differs from sub-model 0. "
                "All type_map values must be identical when linear_ener "
                "has no top-level type_map."
            )
    model_params["type_map"] = deepcopy(first_type_map)


def validate_linear_shared_descriptor_type_maps(
    models: list[dict[str, Any]],
    shared_links: dict[str, Any] | None,
) -> None:
    """Reject descriptor sharing across incompatible linear sub-model type maps."""
    if not shared_links:
        return
    for shared_key, shared_item in shared_links.items():
        descriptor_links = [
            link for link in shared_item["links"] if "descriptor" in link["shared_type"]
        ]
        if len(descriptor_links) < 2:
            continue
        base_link = descriptor_links[0]
        base_index = _get_linear_model_index(base_link["model_key"])
        base_type_map = models[base_index]["type_map"]
        for link_item in descriptor_links[1:]:
            model_index = _get_linear_model_index(link_item["model_key"])
            model_type_map = models[model_index]["type_map"]
            if model_type_map != base_type_map:
                raise ValueError(
                    f"Linear sub-model {model_index} type_map {model_type_map} "
                    f"is incompatible with sub-model {base_index} type_map "
                    f"{base_type_map} for shared descriptor {shared_key!r}. "
                    "Shared descriptor links require identical type_map values."
                )


@BaseModel.register("linear_ener")
class LinearEnergyModel(DPLinearModel_):
    model_type = "linear_ener"

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

    def share_params(
        self,
        shared_links: dict[str, Any],
        model_key_prob_map: dict[str, float] | None = None,
        data_stat_protect: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """Share parameters between linear sub-models.

        ``shared_links`` follows the same structure as the multi-task
        preprocessor. Linear sub-model keys are named ``model_0``, ``model_1``,
        ... by ``get_linear_model``.
        """

        def get_sub_model(model_key: str):  # noqa: ANN202
            model_index = _get_linear_model_index(model_key)
            return self.atomic_model.models[model_index]

        def get_descriptor_class(model_key: str, shared_type: str):  # noqa: ANN202
            sub_model = get_sub_model(model_key)
            if shared_type == "descriptor":
                return sub_model.descriptor
            if "hybrid" in shared_type:
                hybrid_index = int(shared_type.split("_")[-1])
                return sub_model.descriptor.descrpt_list[hybrid_index]
            raise RuntimeError(f"Unknown class_type {shared_type}!")

        for shared_item in shared_links:
            shared_base = shared_links[shared_item]["links"][0]
            class_type_base = shared_base["shared_type"]
            model_key_base = shared_base["model_key"]
            shared_level_base = int(shared_base["shared_level"])
            previous_shared_level = shared_level_base
            if "descriptor" in class_type_base:
                base_class = get_descriptor_class(model_key_base, class_type_base)
                for link_item in shared_links[shared_item]["links"][1:]:
                    class_type_link = link_item["shared_type"]
                    model_key_link = link_item["model_key"]
                    shared_level_link = int(link_item["shared_level"])
                    if shared_level_link < previous_shared_level:
                        raise ValueError(
                            "The shared_links must be sorted by shared_level!"
                        )
                    previous_shared_level = shared_level_link
                    if "descriptor" not in class_type_link:
                        raise ValueError(
                            f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                        )
                    link_class = get_descriptor_class(model_key_link, class_type_link)
                    link_class.share_params(
                        base_class, shared_level_link, resume=resume
                    )
                    log.warning(
                        "Shared params of %s.%s and %s.%s!",
                        model_key_base,
                        class_type_base,
                        model_key_link,
                        class_type_link,
                    )
            else:
                base_model = get_sub_model(model_key_base)
                if hasattr(base_model, class_type_base):
                    base_class = getattr(base_model, class_type_base)
                    for link_item in shared_links[shared_item]["links"][1:]:
                        class_type_link = link_item["shared_type"]
                        model_key_link = link_item["model_key"]
                        shared_level_link = int(link_item["shared_level"])
                        if shared_level_link < previous_shared_level:
                            raise ValueError(
                                "The shared_links must be sorted by shared_level!"
                            )
                        previous_shared_level = shared_level_link
                        if class_type_base != class_type_link:
                            raise ValueError(
                                f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                            )
                        link_model = get_sub_model(model_key_link)
                        link_class = getattr(link_model, class_type_link)
                        if model_key_prob_map is None:
                            frac_prob = 1.0
                        else:
                            frac_prob = (
                                model_key_prob_map[model_key_link]
                                / model_key_prob_map[model_key_base]
                            )
                        link_class.share_params(
                            base_class,
                            shared_level_link,
                            model_prob=frac_prob,
                            protection=data_stat_protect,
                            resume=resume,
                        )
                        log.warning(
                            "Shared params of %s.%s and %s.%s!",
                            model_key_base,
                            class_type_base,
                            model_key_link,
                            class_type_link,
                        )

    def translated_output_def(self) -> dict[str, OutputVariableDef]:
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
        return output_def

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common(
            coord,
            atype,
            box,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            charge_spin=charge_spin,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-2)
        else:
            model_predict["force"] = model_ret["dforce"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
        return model_predict

    @torch.jit.export
    def forward_lower(
        self,
        extended_coord: torch.Tensor,
        extended_atype: torch.Tensor,
        nlist: torch.Tensor,
        mapping: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        do_atomic_virial: bool = False,
        comm_dict: dict[str, torch.Tensor] | None = None,
        charge_spin: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model_ret = self.forward_common_lower(
            extended_coord,
            extended_atype,
            nlist,
            mapping=mapping,
            fparam=fparam,
            aparam=aparam,
            do_atomic_virial=do_atomic_virial,
            comm_dict=comm_dict,
            extra_nlist_sort=self.need_sorted_nlist_for_lower(),
            charge_spin=charge_spin,
        )

        model_predict = {}
        model_predict["atom_energy"] = model_ret["energy"]
        model_predict["energy"] = model_ret["energy_redu"]
        if self.do_grad_r("energy"):
            model_predict["extended_force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["extended_virial"] = model_ret["energy_derv_c"].squeeze(
                    -3
                )
        else:
            assert model_ret["dforce"] is not None
            model_predict["dforce"] = model_ret["dforce"]
        return model_predict

    @classmethod
    def update_sel(
        cls,
        train_data: DeepmdDataSystem,
        type_map: list[str] | None,
        local_jdata: dict,
    ) -> tuple[dict, float | None]:
        """Update the selection and perform neighbor statistics.

        Parameters
        ----------
        train_data : DeepmdDataSystem
            data used to do neighbor statistics
        type_map : list[str], optional
            The name of each type of atoms
        local_jdata : dict
            The local data refer to the current class

        Returns
        -------
        dict
            The updated local data
        float
            The minimum distance between two atoms
        """
        local_jdata_cpy = deepcopy(local_jdata)
        original_models = deepcopy(local_jdata_cpy["models"])
        has_shared_dict = "shared_dict" in local_jdata_cpy
        if has_shared_dict:
            shared_config = {
                "model_dict": {
                    f"model_{idx}": sub_model
                    for idx, sub_model in enumerate(local_jdata_cpy["models"])
                },
                "shared_dict": local_jdata_cpy.get("shared_dict", {}),
            }
            if "type_map" in local_jdata_cpy:
                shared_config["type_map"] = deepcopy(local_jdata_cpy["type_map"])
            shared_config, shared_links = preprocess_shared_params(
                shared_config,
                require_shared_type_map=False,
            )
            local_jdata_cpy["models"] = list(shared_config["model_dict"].values())
            normalize_linear_model_type_map(local_jdata_cpy)
            validate_linear_shared_descriptor_type_maps(
                local_jdata_cpy["models"],
                shared_links,
            )
        type_map = local_jdata_cpy["type_map"]
        min_nbor_dist = None
        for idx, sub_model in enumerate(local_jdata_cpy["models"]):
            if "tab_file" not in sub_model:
                sub_type_map = sub_model.get("type_map", type_map)
                local_jdata_cpy["models"][idx], temp_min = DPModelCommon.update_sel(
                    train_data, sub_type_map, sub_model
                )
                if min_nbor_dist is None or temp_min <= min_nbor_dist:
                    min_nbor_dist = temp_min
        if not has_shared_dict:
            return local_jdata_cpy, min_nbor_dist

        def get_shared_key(shared_ref: str) -> str:
            return shared_ref.split(":", maxsplit=1)[0]

        ret_jdata = deepcopy(local_jdata)
        ret_jdata["models"] = original_models
        if "type_map" not in ret_jdata:
            ret_jdata["type_map"] = deepcopy(type_map)
        for idx, original_sub_model in enumerate(original_models):
            if "tab_file" in original_sub_model:
                continue
            updated_sub_model = local_jdata_cpy["models"][idx]
            descriptor_ref = original_sub_model.get("descriptor")
            if isinstance(descriptor_ref, str):
                ret_jdata["shared_dict"][get_shared_key(descriptor_ref)] = (
                    updated_sub_model["descriptor"]
                )
            elif (
                isinstance(descriptor_ref, dict)
                and descriptor_ref.get("type") == "hybrid"
            ):
                updated_descriptor = updated_sub_model["descriptor"]
                for hybrid_idx, hybrid_ref in enumerate(descriptor_ref["list"]):
                    if isinstance(hybrid_ref, str):
                        ret_jdata["shared_dict"][get_shared_key(hybrid_ref)] = (
                            updated_descriptor["list"][hybrid_idx]
                        )
                    else:
                        ret_jdata["models"][idx]["descriptor"]["list"][hybrid_idx] = (
                            updated_descriptor["list"][hybrid_idx]
                        )
            else:
                ret_jdata["models"][idx]["descriptor"] = updated_sub_model["descriptor"]
        return ret_jdata, min_nbor_dist
