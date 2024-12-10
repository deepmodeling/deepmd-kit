# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

import logging
from collections import (
    OrderedDict,
)
from typing import (
    Union,
)

import paddle

_StateDict = Union[dict[str, paddle.Tensor], OrderedDict[str, paddle.Tensor]]


log = logging.getLogger(__name__)


class ModelWrapper(paddle.nn.Layer):
    def __init__(
        self,
        model: paddle.nn.Layer | dict,
        loss: paddle.nn.Layer | dict = None,
        model_params=None,
        shared_links=None,
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
        self.model = paddle.nn.LayerDict()
        # Model
        if isinstance(model, paddle.nn.Layer):
            self.model["Default"] = model
        elif isinstance(model, dict):
            self.multi_task = True
            for task_key in model:
                assert isinstance(
                    model[task_key], paddle.nn.Layer
                ), f"{task_key} in model_dict is not a paddle.nn.Layer!"
                self.model[task_key] = model[task_key]
        # Loss
        self.loss = None
        if loss is not None:
            self.loss = paddle.nn.LayerDict()
            if isinstance(loss, paddle.nn.Layer):
                self.loss["Default"] = loss
            elif isinstance(loss, dict):
                for task_key in loss:
                    assert isinstance(
                        loss[task_key], paddle.nn.Layer
                    ), f"{task_key} in loss_dict is not a paddle.nn.Layer!"
                    self.loss[task_key] = loss[task_key]
        self.inference_only = self.loss is None

    def share_params(self, shared_links, resume=False) -> None:
        """
        Share the parameters of classes following rules defined in shared_links during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        raise NotImplementedError("share_params is not implemented yet")

    def forward(
        self,
        coord,
        atype,
        spin: paddle.Tensor | None = None,
        box: paddle.Tensor | None = None,
        cur_lr: paddle.Tensor | None = None,
        label: paddle.Tensor | None = None,
        task_key: paddle.Tensor | None = None,
        inference_only=False,
        do_atomic_virial=False,
        fparam: paddle.Tensor | None = None,
        aparam: paddle.Tensor | None = None,
    ):
        if not self.multi_task:
            task_key = "Default"
        else:
            assert (
                task_key is not None
            ), f"Multitask model must specify the inference task! Supported tasks are {list(self.model.keys())}."
        input_dict = {
            "coord": coord,
            "atype": atype,
            "box": box,
            "do_atomic_virial": do_atomic_virial,
            "fparam": fparam,
            "aparam": aparam,
        }
        has_spin = getattr(self.model[task_key], "has_spin", False)
        if callable(has_spin):
            has_spin = has_spin()
        if has_spin:
            input_dict["spin"] = spin

        if self.inference_only or inference_only:
            model_pred = self.model[task_key](**input_dict)
            return model_pred, None, None
        else:
            natoms = atype.shape[-1]
            model_pred, loss, more_loss = self.loss[task_key](
                input_dict,
                self.model[task_key],
                label,
                natoms=natoms,
                learning_rate=cur_lr,
            )
            return model_pred, loss, more_loss

    def load_state_dict(
        self,
        state_dict: _StateDict,
    ) -> tuple[list[str], list[str]]:
        self.set_extra_state(state_dict.pop("_extra_state"))
        return super().set_state_dict(state_dict)

    def set_state_dict(
        self,
        state_dict: _StateDict,
    ) -> tuple[list[str], list[str]]:
        return self.load_state_dict(state_dict)

    def state_dict(self):
        state_dict = super().state_dict()
        extra_state = self.get_extra_state()
        state_dict.update({"_extra_state": extra_state})
        return state_dict

    def set_extra_state(self, extra_state: dict):
        self.model_params = extra_state["model_params"]
        self.train_infos = extra_state["train_infos"]
        return None

    def get_extra_state(self) -> dict:
        extra_state = {
            "model_params": self.model_params,
            "train_infos": self.train_infos,
        }
        return extra_state
