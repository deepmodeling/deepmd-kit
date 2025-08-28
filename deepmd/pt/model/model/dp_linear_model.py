# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    LinearEnergyAtomicModel,
)
from deepmd.pt.model.descriptor import (
    BaseDescriptor,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)
from deepmd.pt.model.task import (
    BaseFitting,
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

DPLinearModel_ = make_model(LinearEnergyAtomicModel)


@BaseModel.register("linear_ener")
class LinearEnergyModel(DPLinearModel_):
    model_type = "linear_ener"

    def __init__(
        self,
        models: Optional[list] = None,
        shared_dict: Optional[dict] = None,
        weights: Optional[str] = "mean",
        type_map: Optional[list[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        # Handle shared_dict preprocessing if provided
        if shared_dict is not None and models is not None:
            # Convert to multi-task format for preprocessing
            model_config = {
                "model_dict": {f"model_{i}": model for i, model in enumerate(models)},
                "shared_dict": shared_dict,
            }
            processed_config, shared_links = preprocess_shared_params(model_config)
            # Extract processed models
            processed_models = [
                processed_config["model_dict"][f"model_{i}"] for i in range(len(models))
            ]

            # Build individual atomic models from processed config
            sub_models = []
            for model_config in processed_models:
                # Create descriptor and fitting net from the processed config
                descriptor_config = model_config["descriptor"]
                fitting_config = model_config["fitting_net"]
                model_type_map = model_config["type_map"]

                # Add ntypes and type_map to descriptor config if needed
                descriptor_config = descriptor_config.copy()
                if "ntypes" not in descriptor_config:
                    descriptor_config["ntypes"] = len(model_type_map)
                if "type_map" not in descriptor_config:
                    descriptor_config["type_map"] = model_type_map

                descriptor = BaseDescriptor.get_class_by_type(
                    descriptor_config["type"]
                )(**descriptor_config)

                # Add required parameters to fitting config
                fitting_config = fitting_config.copy()
                if "ntypes" not in fitting_config:
                    fitting_config["ntypes"] = len(model_type_map)
                if "dim_descrpt" not in fitting_config:
                    fitting_config["dim_descrpt"] = descriptor.get_dim_out()
                if "type_map" not in fitting_config:
                    fitting_config["type_map"] = model_type_map
                # Add embedding_width for dipole fitting if needed
                if (
                    fitting_config.get("type") == "dipole"
                    and "embedding_width" not in fitting_config
                ):
                    fitting_config["embedding_width"] = descriptor.get_dim_emb()

                fitting_net = BaseFitting.get_class_by_type(
                    fitting_config.get("type", "ener")
                )(**fitting_config)

                # Create DPAtomicModel from the processed config
                sub_model = DPAtomicModel(
                    descriptor=descriptor,
                    fitting=fitting_net,
                    type_map=model_type_map,
                )
                sub_models.append(sub_model)

            # Create LinearEnergyAtomicModel with the sub-models
            atomic_model = LinearEnergyAtomicModel(
                models=sub_models,
                type_map=type_map or sub_models[0].get_type_map(),
                weights=weights,
            )

            # Store shared_links for parameter sharing
            self.shared_links = shared_links

            super().__init__(atomic_model_=atomic_model, **kwargs)

            # Apply parameter sharing
            if hasattr(self, "shared_links") and self.shared_links:
                self._share_params()
        elif models is not None:
            # Handle traditional linear model with model configurations
            # Convert model configs to atomic models if they are dictionaries
            if all(isinstance(m, dict) for m in models):
                sub_models = []
                for model_config in models:
                    # Create descriptor and fitting net from config
                    descriptor_config = model_config["descriptor"]
                    fitting_config = model_config["fitting_net"]
                    model_type_map = model_config.get("type_map", type_map)

                    # Add ntypes and type_map to descriptor config if needed
                    descriptor_config = descriptor_config.copy()
                    if "ntypes" not in descriptor_config:
                        descriptor_config["ntypes"] = len(model_type_map)
                    if "type_map" not in descriptor_config:
                        descriptor_config["type_map"] = model_type_map

                    descriptor = BaseDescriptor.get_class_by_type(
                        descriptor_config["type"]
                    )(**descriptor_config)

                    # Add required parameters to fitting config
                    fitting_config = fitting_config.copy()
                    if "ntypes" not in fitting_config:
                        fitting_config["ntypes"] = len(model_type_map)
                    if "dim_descrpt" not in fitting_config:
                        fitting_config["dim_descrpt"] = descriptor.get_dim_out()
                    if "type_map" not in fitting_config:
                        fitting_config["type_map"] = model_type_map
                    # Add embedding_width for dipole fitting if needed
                    if (
                        fitting_config.get("type") == "dipole"
                        and "embedding_width" not in fitting_config
                    ):
                        fitting_config["embedding_width"] = descriptor.get_dim_emb()

                    fitting_net = BaseFitting.get_class_by_type(
                        fitting_config.get("type", "ener")
                    )(**fitting_config)

                    # Create DPAtomicModel
                    sub_model = DPAtomicModel(
                        descriptor=descriptor,
                        fitting=fitting_net,
                        type_map=model_type_map,
                    )
                    sub_models.append(sub_model)

                # Create LinearEnergyAtomicModel with the sub-models
                atomic_model = LinearEnergyAtomicModel(
                    models=sub_models,
                    type_map=type_map,
                    weights=weights,
                )
                super().__init__(atomic_model_=atomic_model, **kwargs)
            else:
                # Models are already atomic model objects
                atomic_model = LinearEnergyAtomicModel(
                    models=models,
                    type_map=type_map,
                    weights=weights,
                )
                super().__init__(atomic_model_=atomic_model, **kwargs)
            self.shared_links = None
        else:
            # Standard initialization without models
            super().__init__(*args, **kwargs)
            self.shared_links = None

    def _share_params(self, resume=False) -> None:
        """Share the parameters between sub-models based on shared_links.

        Adapted from deepmd.pt.train.wrapper.ModelWrapper.share_params.
        """
        if not hasattr(self, "shared_links") or not self.shared_links:
            return

        supported_types = ["descriptor", "fitting_net"]
        for shared_item in self.shared_links:
            class_name = self.shared_links[shared_item]["type"]
            shared_base = self.shared_links[shared_item]["links"][0]
            class_type_base = shared_base["shared_type"]
            model_key_base = shared_base["model_key"]
            shared_level_base = shared_base["shared_level"]

            # Get model index from model_key (format: "model_X")
            base_idx = int(model_key_base.split("_")[1])
            base_model = self.atomic_model.models[base_idx]

            if "descriptor" in class_type_base:
                if class_type_base == "descriptor":
                    base_class = base_model.descriptor
                else:
                    raise RuntimeError(f"Unknown class_type {class_type_base}!")

                for link_item in self.shared_links[shared_item]["links"][1:]:
                    class_type_link = link_item["shared_type"]
                    model_key_link = link_item["model_key"]
                    shared_level_link = int(link_item["shared_level"])

                    # Get model index from model_key
                    link_idx = int(model_key_link.split("_")[1])
                    link_model = self.atomic_model.models[link_idx]

                    assert shared_level_link >= shared_level_base, (
                        "The shared_links must be sorted by shared_level!"
                    )
                    assert "descriptor" in class_type_link, (
                        f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                    )

                    if class_type_link == "descriptor":
                        link_class = link_model.descriptor
                    else:
                        raise RuntimeError(f"Unknown class_type {class_type_link}!")

                    link_class.share_params(
                        base_class, shared_level_link, resume=resume
                    )
            else:
                # Handle fitting_net sharing
                if hasattr(base_model, class_type_base):
                    base_class = getattr(base_model, class_type_base)
                    for link_item in self.shared_links[shared_item]["links"][1:]:
                        class_type_link = link_item["shared_type"]
                        model_key_link = link_item["model_key"]
                        shared_level_link = int(link_item["shared_level"])

                        # Get model index from model_key
                        link_idx = int(model_key_link.split("_")[1])
                        link_model = self.atomic_model.models[link_idx]

                        assert shared_level_link >= shared_level_base, (
                            "The shared_links must be sorted by shared_level!"
                        )
                        assert class_type_base == class_type_link, (
                            f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                        )

                        link_class = getattr(link_model, class_type_link)
                        link_class.share_params(
                            base_class, shared_level_link, resume=resume
                        )

    def translated_output_def(self):
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
            output_def["atom_virial"].squeeze(-3)
        if "mask" in out_def_data:
            output_def["mask"] = out_def_data["mask"]
        return output_def

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
        do_atomic_virial: bool = False,
    ) -> dict[str, torch.Tensor]:
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
        if self.do_grad_r("energy"):
            model_predict["force"] = model_ret["energy_derv_r"].squeeze(-2)
        if self.do_grad_c("energy"):
            model_predict["virial"] = model_ret["energy_derv_c_redu"].squeeze(-2)
            if do_atomic_virial:
                model_predict["atom_virial"] = model_ret["energy_derv_c"].squeeze(-3)
        else:
            model_predict["force"] = model_ret["dforce"]
        if "mask" in model_ret:
            model_predict["mask"] = model_ret["mask"]
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
        comm_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
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
        type_map: Optional[list[str]],
        local_jdata: dict,
    ) -> tuple[dict, Optional[float]]:
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
        local_jdata_cpy = local_jdata.copy()
        type_map = local_jdata_cpy["type_map"]
        min_nbor_dist = None
        for idx, sub_model in enumerate(local_jdata_cpy["models"]):
            if "tab_file" not in sub_model:
                sub_model, temp_min = DPModelCommon.update_sel(
                    train_data, type_map, local_jdata["models"][idx]
                )
                if min_nbor_dist is None or temp_min <= min_nbor_dist:
                    min_nbor_dist = temp_min
        return local_jdata_cpy, min_nbor_dist
