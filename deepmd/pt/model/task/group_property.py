# SPDX-License-Identifier: LGPL-3.0-or-later
"""Group-level property fitting network."""

from __future__ import annotations

from typing import Any

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    PRECISION_DICT,
)


def _activation(name: str) -> torch.nn.Module:
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "gelu":
        return torch.nn.GELU()
    if name in {"linear", "none"}:
        return torch.nn.Identity()
    raise ValueError(f"Unsupported activation_function: {name!r}")


@Fitting.register("group_property")
class GroupPropertyFittingNet(Fitting):
    """MLP that maps aggregated group embeddings to group properties."""

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        property_name: str,
        task_dim: int = 1,
        neuron: list[int] | None = None,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        trainable: bool | list[bool] = True,
        type_map: list[str] | None = None,
        numb_fparam: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.var_name = property_name
        self.task_dim = task_dim
        self.dim_out = task_dim
        self.neuron = list(neuron or [128, 128])
        self.activation_function = activation_function
        self.precision = precision
        self.prec = PRECISION_DICT[self.precision]
        self.type_map = list(type_map or [])
        self.trainable = (
            all(trainable) if isinstance(trainable, list) else bool(trainable)
        )
        self.numb_fparam = int(numb_fparam)

        # Per-group side features (fparam) are concatenated to the aggregated
        # group embedding, so the fitting input widens by ``numb_fparam``.
        dims = [dim_descrpt + self.numb_fparam, *self.neuron, task_dim]
        layers: list[torch.nn.Module] = []
        for ii in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[ii], dims[ii + 1], dtype=self.prec))
            if ii < len(dims) - 2:
                layers.append(_activation(self.activation_function))
        self.network = torch.nn.Sequential(*layers).to(env.DEVICE)
        for param in self.parameters():
            param.requires_grad = self.trainable

    def forward(self, group_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(group_embedding.to(self.prec))

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.task_dim],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=True,
                ),
            ]
        )

    def get_dim_fparam(self) -> int:
        return self.numb_fparam

    def has_default_fparam(self) -> bool:
        return False

    def get_default_fparam(self) -> torch.Tensor | None:
        return None

    def get_dim_aparam(self) -> int:
        return 0

    def get_type_map(self) -> list[str]:
        return self.type_map

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: Any | None = None
    ) -> None:
        self.type_map = list(type_map)

    def compute_input_stats(self, *args: Any, **kwargs: Any) -> None:
        return None

    def serialize(self) -> dict[str, Any]:
        return {
            "type": "group_property",
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "property_name": self.var_name,
            "task_dim": self.task_dim,
            "numb_fparam": self.numb_fparam,
            "neuron": self.neuron,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "type_map": self.type_map,
        }
