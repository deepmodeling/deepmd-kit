# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import torch

log = logging.getLogger(__name__)


class ModelWrapper(torch.nn.Module):
    """Model wrapper that bundles model(s) and loss(es).

    Supports both single-task and multi-task training.

    Parameters
    ----------
    model : torch.nn.Module or dict
        Single model or dict of models keyed by task name.
    loss : torch.nn.Module or dict or None
        Single loss or dict of losses keyed by task name.
    model_params : dict, optional
        Model parameters to store as extra state.
    """

    def __init__(
        self,
        model: torch.nn.Module | dict,
        loss: torch.nn.Module | dict | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_params = model_params if model_params is not None else {}
        self.train_infos: dict[str, Any] = {
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
        # Loss — dpmodel losses are not nn.Module, so store in a plain dict.
        self.loss: dict[str, Any] | None = None
        if loss is not None:
            if isinstance(loss, dict):
                self.loss = dict(loss)
            else:
                self.loss = {"Default": loss}
        self.inference_only = self.loss is None

    def share_params(
        self,
        shared_links: dict[str, Any],
        model_key_prob_map: dict,
        data_stat_protect: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """Share parameters between models following rules in shared_links.

        Parameters
        ----------
        shared_links : dict
            Sharing rules from ``preprocess_shared_params``.
        model_key_prob_map : dict
            Probability map for each model key (for fitting_net stat weighting).
        data_stat_protect : float
            Protection value for standard deviation computation.
        resume : bool
            Whether resuming from checkpoint.
        """
        for shared_item in shared_links:
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
                        .descrpt_list[hybrid_index]
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
                            .descrpt_list[hybrid_index]
                        )
                    else:
                        raise RuntimeError(f"Unknown class_type {class_type_link}!")
                    frac_prob = (
                        model_key_prob_map[model_key_link]
                        / model_key_prob_map[model_key_base]
                    )
                    link_class.share_params(
                        base_class,
                        shared_level_link,
                        model_prob=frac_prob,
                        resume=resume,
                    )
                    log.warning(
                        f"Shared params of {model_key_base}.{class_type_base} "
                        f"and {model_key_link}.{class_type_link}!"
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
                            f"Shared params of {model_key_base}.{class_type_base} "
                            f"and {model_key_link}.{class_type_link}!"
                        )

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
        cur_lr: float | torch.Tensor | None = None,
        label: dict[str, torch.Tensor] | None = None,
        task_key: str | None = None,
        do_atomic_virial: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict | None]:
        if not self.multi_task:
            task_key = "Default"
        else:
            assert task_key is not None, (
                f"Multitask model must specify the inference task! "
                f"Supported tasks are {list(self.model.keys())}."
            )
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": box,
            "do_atomic_virial": do_atomic_virial,
            "fparam": fparam,
            "aparam": aparam,
        }

        model_pred = self.model[task_key](**input_dict)

        if self.inference_only or label is None:
            return model_pred, None, None
        else:
            natoms = atype.shape[-1]
            loss, more_loss = self.loss[task_key](
                cur_lr,
                natoms,
                model_pred,
                label,
            )
            return model_pred, loss, more_loss

    def set_extra_state(self, state: dict) -> None:
        self.model_params = state.get("model_params", {})
        self.train_infos = state.get("train_infos", {"lr": 0, "step": 0})

    def get_extra_state(self) -> dict:
        return {
            "model_params": self.model_params,
            "train_infos": self.train_infos,
        }
