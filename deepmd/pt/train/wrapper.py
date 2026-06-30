# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from collections.abc import (
    Generator,
)
from contextlib import (
    contextmanager,
)
from typing import (
    Any,
)

import torch

if torch.__version__.startswith("2"):
    import torch._dynamo


log = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module | dict,
        loss: torch.nn.Module | dict = None,
        model_params: dict[str, Any] | None = None,
        shared_links: dict[str, Any] | None = None,
        modifier: torch.nn.Module | None = None,
    ) -> None:
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
                assert isinstance(model[task_key], torch.nn.Module), (
                    f"{task_key} in model_dict is not a torch.nn.Module!"
                )
                self.model[task_key] = model[task_key]
        # Loss
        self.loss = None
        if loss is not None:
            self.loss = torch.nn.ModuleDict()
            if isinstance(loss, torch.nn.Module):
                self.loss["Default"] = loss
            elif isinstance(loss, dict):
                for task_key in loss:
                    assert isinstance(loss[task_key], torch.nn.Module), (
                        f"{task_key} in loss_dict is not a torch.nn.Module!"
                    )
                    self.loss[task_key] = loss[task_key]
        self.inference_only = self.loss is None
        # Modifier
        self.modifier = modifier

    def share_params(
        self,
        shared_links: dict[str, Any],
        model_key_prob_map: dict,
        data_stat_protect: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """
        Share the parameters of classes following rules defined in shared_links during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
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
                    base_class = self.model[model_key_base].get_descriptor()
                elif "hybrid" in class_type_base:
                    hybrid_index = int(class_type_base.split("_")[-1])
                    base_class = (
                        self.model[model_key_base]
                        .get_descriptor()
                        .descriptor_list[hybrid_index]
                    )
                else:
                    raise RuntimeError(f"Unknown class_type {class_type_base}!")
                for link_item in shared_links[shared_item]["links"][1:]:
                    class_type_link = link_item["shared_type"]
                    model_key_link = link_item["model_key"]
                    shared_level_link = int(link_item["shared_level"])
                    assert shared_level_link >= shared_level_base, (
                        "The shared_links must be sorted by shared_level!"
                    )
                    assert "descriptor" in class_type_link, (
                        f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                    )
                    if class_type_link == "descriptor":
                        link_class = self.model[model_key_link].get_descriptor()
                    elif "hybrid" in class_type_link:
                        hybrid_index = int(class_type_link.split("_")[-1])
                        link_class = (
                            self.model[model_key_link]
                            .get_descriptor()
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
                if hasattr(self.model[model_key_base].atomic_model, class_type_base):
                    base_class = self.model[model_key_base].atomic_model.__getattr__(
                        class_type_base
                    )
                    for link_item in shared_links[shared_item]["links"][1:]:
                        class_type_link = link_item["shared_type"]
                        model_key_link = link_item["model_key"]
                        shared_level_link = int(link_item["shared_level"])
                        assert shared_level_link >= shared_level_base, (
                            "The shared_links must be sorted by shared_level!"
                        )
                        assert class_type_base == class_type_link, (
                            f"Class type mismatched: {class_type_base} vs {class_type_link}!"
                        )
                        link_class = self.model[
                            model_key_link
                        ].atomic_model.__getattr__(class_type_link)
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
                            f"Shared params of {model_key_base}.{class_type_base} and {model_key_link}.{class_type_link}!"
                        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        spin: torch.Tensor | None = None,
        box: torch.Tensor | None = None,
        cur_lr: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        task_key: torch.Tensor | None = None,
        skip_loss: bool = False,
        do_atomic_virial: bool = False,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        charge_spin: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        extended_atype: torch.Tensor | None = None,
        extended_batch: torch.Tensor | None = None,
        extended_image: torch.Tensor | None = None,
        extended_ptr: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        central_ext_index: torch.Tensor | None = None,
        nlist: torch.Tensor | None = None,
        nlist_ext: torch.Tensor | None = None,
        a_nlist: torch.Tensor | None = None,
        a_nlist_ext: torch.Tensor | None = None,
        nlist_mask: torch.Tensor | None = None,
        a_nlist_mask: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        angle_index: torch.Tensor | None = None,
    ) -> tuple[Any, Any, Any]:
        if not self.multi_task:
            task_key = "Default"
        else:
            assert task_key is not None, (
                f"Multitask model must specify the inference task! Supported tasks are {list(self.model.keys())}."
            )
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": box,
            "do_atomic_virial": do_atomic_virial,
            "fparam": fparam,
            "aparam": aparam,
            "charge_spin": charge_spin,
        }
        if batch is not None and ptr is not None:
            input_dict["mixed_batch"] = {
                "batch": batch,
                "ptr": ptr,
                "extended_atype": extended_atype,
                "extended_batch": extended_batch,
                "extended_image": extended_image,
                "extended_ptr": extended_ptr,
                "mapping": mapping,
                "central_ext_index": central_ext_index,
                "nlist": nlist,
                "nlist_ext": nlist_ext,
                "a_nlist": a_nlist,
                "a_nlist_ext": a_nlist_ext,
                "nlist_mask": nlist_mask,
                "a_nlist_mask": a_nlist_mask,
                "edge_index": edge_index,
                "angle_index": angle_index,
            }
        has_spin = getattr(self.model[task_key], "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if has_spin:
            input_dict["spin"] = spin

        # A loss-free wrapper is a pure inference object, so parameters can be
        # treated as constants while coordinate gradients remain enabled.
        if self.inference_only:
            with self._frozen_parameter_context():
                model_pred = self._forward_without_loss(task_key, input_dict)
            return model_pred, None, None
        # Training wrappers may request predictions without loss construction
        # and still backpropagate those predictions into model parameters
        # (for example, KFWrapper updates).
        if skip_loss:
            model_pred = self._forward_without_loss(task_key, input_dict)
            return model_pred, None, None

        natoms = atype.shape[-1] if atype.dim() > 1 else atype.shape[0]
        model_pred, loss, more_loss = self.loss[task_key](
            input_dict,
            self.model[task_key],
            label,
            natoms=natoms,
            learning_rate=cur_lr,
        )
        return model_pred, loss, more_loss

    @contextmanager
    def _frozen_parameter_context(self) -> Generator[None, None, None]:
        """
        Freeze model parameters during pure inference.

        Conservative inference still differentiates model outputs with respect
        to coordinates to obtain forces and virials. Parameter gradients are not
        part of that contract, so disabling them trims the autograd graph while
        leaving the coordinate-gradient path intact.
        """
        params = tuple(self.parameters())
        requires_grad = tuple(param.requires_grad for param in params)
        if not any(requires_grad):
            yield
            return
        for param in params:
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param, flag in zip(params, requires_grad, strict=True):
                param.requires_grad_(flag)

    def _forward_without_loss(
        self,
        task_key: str,
        input_dict: dict[str, Any],
    ) -> Any:
        """Return predictions without constructing a loss."""
        model_pred = self.model[task_key](**input_dict)
        if self.modifier is not None:
            modifier_pred = self.modifier(**input_dict)
            for key, value in modifier_pred.items():
                model_pred[key] = model_pred[key] + value
        return model_pred

    def set_extra_state(self, state: dict) -> None:
        self.model_params = state["model_params"]
        self.train_infos = state["train_infos"]
        return None

    def get_extra_state(self) -> dict:
        state = {
            "model_params": self.model_params,
            "train_infos": self.train_infos,
        }
        return state
