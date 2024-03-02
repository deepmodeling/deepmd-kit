# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Dict,
    Optional,
    Union,
)

import torch

if torch.__version__.startswith("2"):
    import torch._dynamo


log = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: Union[torch.nn.Module, Dict],
        loss: Union[torch.nn.Module, Dict] = None,
        model_params=None,
        shared_links=None,
    ):
        """Construct a DeePMD model wrapper.

        Args:
        - config: The Dict-like configuration with training options.
        """
        super().__init__()
        self.model_params = model_params if model_params is not None else {}
        self.train_infos = {
            "lr": 0,
            "step": 0,
        }
        self.multi_task = False
        self.model = torch.nn.ModuleDict()
        # Model
        if isinstance(model, torch.nn.Module):
            self.model["Default"] = model
        elif isinstance(model, dict):
            self.multi_task = True
            for task_key in model:
                assert isinstance(
                    model[task_key], torch.nn.Module
                ), f"{task_key} in model_dict is not a torch.nn.Module!"
                self.model[task_key] = model[task_key]
        # Loss
        self.loss = None
        if loss is not None:
            self.loss = torch.nn.ModuleDict()
            if isinstance(loss, torch.nn.Module):
                self.loss["Default"] = loss
            elif isinstance(loss, dict):
                for task_key in loss:
                    assert isinstance(
                        loss[task_key], torch.nn.Module
                    ), f"{task_key} in loss_dict is not a torch.nn.Module!"
                    self.loss[task_key] = loss[task_key]
        self.inference_only = self.loss is None

    def share_params(self, shared_links, resume=False):
        """
        Share the parameters of classes following rules defined in shared_links during multitask training.
        If not start from checkpoint (resume is False),
        some seperated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        supported_types = ["descriptor", "fitting_net"]
        for shared_item in shared_links:
            class_name = shared_links[shared_item]["type"]
            shared_base = shared_links[shared_item]["links"][0]
            class_type_base = shared_base["shared_type"]
            model_key_base = shared_base["model_key"]
            shared_level_base = shared_base["shared_level"]
            if "descriptor" in class_type_base:
                if class_type_base == "descriptor":
                    base_class = self.model[model_key_base].__getattr__("descriptor")
                elif "hybrid" in class_type_base:
                    hybrid_index = int(class_type_base.split("_")[-1])
                    base_class = (
                        self.model[model_key_base]
                        .__getattr__("descriptor")
                        .descriptor_list[hybrid_index]
                    )
                else:
                    raise RuntimeError(f"Unknown class_type {class_type_base}!")
                for link_item in shared_links[shared_item]["links"][1:]:
                    class_type_link = link_item["shared_type"]
                    model_key_link = link_item["model_key"]
                    shared_level_link = int(link_item["shared_level"])
                    assert (
                        shared_level_link >= shared_level_base
                    ), "The shared_links must be sorted by shared_level!"
                    assert (
                        "descriptor" in class_type_link
                    ), f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                    if class_type_link == "descriptor":
                        link_class = self.model[model_key_link].__getattr__(
                            "descriptor"
                        )
                    elif "hybrid" in class_type_link:
                        hybrid_index = int(class_type_link.split("_")[-1])
                        link_class = (
                            self.model[model_key_link]
                            .__getattr__("descriptor")
                            .descriptor_list[hybrid_index]
                        )
                    else:
                        raise RuntimeError(f"Unknown class_type {class_type_link}!")
                    link_class.share_params(
                        base_class, shared_level_link, resume=resume
                    )
                    log.warning(
                        f"Shared params of {model_key_base}.{class_type_base} and {model_key_link}.{class_type_link}!"
                    )
            else:
                if hasattr(self.model[model_key_base], class_type_base):
                    base_class = self.model[model_key_base].__getattr__(class_type_base)
                    for link_item in shared_links[shared_item]["links"][1:]:
                        class_type_link = link_item["shared_type"]
                        model_key_link = link_item["model_key"]
                        shared_level_link = int(link_item["shared_level"])
                        assert (
                            shared_level_link >= shared_level_base
                        ), "The shared_links must be sorted by shared_level!"
                        assert (
                            class_type_base == class_type_link
                        ), f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                        link_class = self.model[model_key_link].__getattr__(
                            class_type_link
                        )
                        link_class.share_params(
                            base_class, shared_level_link, resume=resume
                        )
                        log.warning(
                            f"Shared params of {model_key_base}.{class_type_base} and {model_key_link}.{class_type_link}!"
                        )

    def forward(
        self,
        coord,
        atype,
        box: Optional[torch.Tensor] = None,
        spin: Optional[torch.Tensor] = None,
        cur_lr: Optional[torch.Tensor] = None,
        label: Optional[torch.Tensor] = None,
        task_key: Optional[torch.Tensor] = None,
        inference_only=False,
        do_atomic_virial=False,
        fparam: Optional[torch.Tensor] = None,
        aparam: Optional[torch.Tensor] = None,
    ):
        if not self.multi_task:
            task_key = "Default"
        else:
            assert (
                task_key is not None
            ), f"Multitask model must specify the inference task! Supported tasks are {list(self.model.keys())}."
        model_pred = self.model[task_key](
            coord,
            atype,
            box=box,
            do_atomic_virial=do_atomic_virial,
            fparam=fparam,
            aparam=aparam,
        )
        natoms = atype.shape[-1]
        if not self.inference_only and not inference_only:
            loss, more_loss = self.loss[task_key](
                model_pred, label, natoms=natoms, learning_rate=cur_lr
            )
            return model_pred, loss, more_loss
        else:
            return model_pred, None, None

    def set_extra_state(self, state: Dict):
        self.model_params = state["model_params"]
        self.train_infos = state["train_infos"]
        return None

    def get_extra_state(self) -> Dict:
        state = {
            "model_params": self.model_params,
            "train_infos": self.train_infos,
        }
        return state
