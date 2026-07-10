# SPDX-License-Identifier: LGPL-3.0-or-later
"""Group-level property fitting network."""

from __future__ import (
    annotations,
)

import logging
from typing import (
    Any,
)

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

_log = logging.getLogger(__name__)


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
        # GELU (not tanh): the group head consumes an un-normalized
        # frame/group embedding concatenated with fparam.  tanh saturates at
        # that input scale, so the head collapses to a constant (bias-only)
        # prediction; GELU keeps gradients flowing and learns per-group signal.
        activation_function: str = "gelu",
        precision: str = DEFAULT_PRECISION,
        trainable: bool | list[bool] = True,
        type_map: list[str] | None = None,
        numb_fparam: int = 0,
        fparam_neuron: list[int] | None = None,
        dim_case_embd: int = 0,
        group_reduce: str = "mean",
        seed: int | None = None,
        # Injected unconditionally by the generic model-building path
        # (deepmd.pt.model.model._get_standard_model_components) for every
        # fitting type; "type" selects this class via the Fitting registry
        # and "mixed_types" is the descriptor's own property (read via
        # GroupPropertyModel.mixed_types()), so neither is used here. Kept
        # as explicit, named, no-op parameters -- not **kwargs -- so any
        # other unrecognized field fails loudly instead of being silently
        # accepted and ignored.
        type: str = "group_property",
        mixed_types: bool = True,
    ) -> None:
        del type, mixed_types
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
        self.seed = seed
        self.numb_fparam = int(numb_fparam)
        self.fparam_neuron = list(fparam_neuron or [])
        if self.fparam_neuron and self.numb_fparam <= 0:
            raise ValueError("fparam_neuron requires numb_fparam > 0.")
        self.dim_case_embd = int(dim_case_embd)
        if self.dim_case_embd > 0:
            # One-hot branch identifier for multi-task training, where several
            # fitting nets (e.g. an MFT aux ener head and this group_property
            # head) share one descriptor and need to stay distinguishable; see
            # set_case_embd. Concatenated onto the group embedding in
            # forward(), so it counts toward the first Linear layer's input
            # width below.
            self.register_buffer(
                "case_embd",
                torch.zeros(self.dim_case_embd, dtype=self.prec, device=env.DEVICE),
            )
        else:
            self.case_embd = None
        if group_reduce not in ("mean", "sum"):
            raise ValueError(
                f"group_reduce must be 'mean' or 'sum'; got {group_reduce!r}."
            )
        # frame->group reduction (executed in GroupPropertyModel.forward):
        #   "mean": group_emb = index_add(w_i*e_i) / index_add(w_i) -- weighted
        #           mean; embedding scale independent of group size. (default)
        #   "sum" : group_emb = index_add(w_i*e_i) -- norm grows with group size;
        #           for additive-property semantics only.
        self.group_reduce = group_reduce

        # Per-group side features (fparam) can either be concatenated directly
        # to the aggregated group embedding (default/backward-compatible path),
        # or first encoded through a small fparam-only branch.  The branch helps
        # when low-dimensional experimental conditions (salt/concentration/pH)
        # would otherwise be numerically drowned by a high-dimensional structural
        # embedding in the first fused Linear layer.
        fparam_out_dim = (
            self.fparam_neuron[-1] if self.fparam_neuron else self.numb_fparam
        )
        dims = [
            dim_descrpt + fparam_out_dim + self.dim_case_embd,
            *self.neuron,
            task_dim,
        ]

        def _build_mlp(
            dims: list[int], *, activate_last: bool = False
        ) -> list[torch.nn.Module]:
            layers: list[torch.nn.Module] = []
            for ii in range(len(dims) - 1):
                layers.append(torch.nn.Linear(dims[ii], dims[ii + 1], dtype=self.prec))
                if activate_last or ii < len(dims) - 2:
                    layers.append(_activation(self.activation_function))
            return layers

        def _build_layers() -> tuple[list[torch.nn.Module], list[torch.nn.Module]]:
            fparam_layers = (
                _build_mlp([self.numb_fparam, *self.fparam_neuron], activate_last=True)
                if self.fparam_neuron
                else []
            )
            return fparam_layers, _build_mlp(dims)

        if seed is None:
            # No seed requested: draw from (and advance) the caller's global
            # RNG stream exactly as an unseeded torch.nn.Linear always does.
            fparam_layers, layers = _build_layers()
        else:
            # Seed only this net's own initialization, without disturbing the
            # caller's global RNG state (torch.nn.Linear.reset_parameters()
            # has no generator= argument, so fork-then-seed is the way to
            # scope a seed to a plain nn.Linear-based net).
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                fparam_layers, layers = _build_layers()
        self.fparam_network = torch.nn.Sequential(*fparam_layers).to(env.DEVICE)
        self.network = torch.nn.Sequential(*layers).to(env.DEVICE)
        # Task E (route 3): group_property does NOT init the output bias from
        # label statistics.  The property path's frame-level stat would count each
        # group's replicated label once per component (biasing toward large groups)
        # and is computed at frame level while this net operates at group level.
        # Zero-init the output bias explicitly; training learns the offset.
        last = self.network[-1]
        if isinstance(last, torch.nn.Linear) and last.bias is not None:
            torch.nn.init.zeros_(last.bias)
            _log.warning(
                "group_property output bias is zero-initialized (label-stat bias "
                "init is disabled for grouped labels); training may need more steps."
            )

        linear_layers = [m for m in self.network if isinstance(m, torch.nn.Linear)]
        if isinstance(trainable, list):
            if len(trainable) != len(linear_layers):
                raise ValueError(
                    f"trainable has {len(trainable)} entries; expected "
                    f"{len(linear_layers)} (one per Linear layer, i.e. "
                    "len(neuron)+1)."
                )
            self.trainable = trainable
            for layer, layer_trainable in zip(linear_layers, trainable, strict=True):
                for param in layer.parameters():
                    param.requires_grad = bool(layer_trainable)
        else:
            self.trainable = bool(trainable)
            for param in self.parameters():
                param.requires_grad = self.trainable

    def forward(self, group_embedding: torch.Tensor) -> torch.Tensor:
        group_embedding = group_embedding.to(self.prec)
        if self.fparam_neuron:
            descrpt = group_embedding[:, : self.dim_descrpt]
            fparam = group_embedding[:, self.dim_descrpt :]
            if fparam.shape[-1] != self.numb_fparam:
                raise ValueError(
                    f"expected {self.numb_fparam} fparam columns, got {fparam.shape[-1]}"
                )
            group_embedding = torch.cat([descrpt, self.fparam_network(fparam)], dim=-1)
        if self.dim_case_embd > 0:
            case_embd = self.case_embd.to(
                dtype=group_embedding.dtype, device=group_embedding.device
            )
            case_embd = case_embd.unsqueeze(0).expand(group_embedding.shape[0], -1)
            group_embedding = torch.cat([group_embedding, case_embd], dim=-1)
        return self.network(group_embedding)

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

    def set_case_embd(self, case_idx: int) -> None:
        """
        Set the case (branch) embedding of this fitting net by the given
        case_idx, concatenated with the aggregated group embedding and fed
        into the fitting net. Used to keep multiple fitting nets
        distinguishable when they share one descriptor in multi-task
        training.
        """
        self.case_embd = torch.eye(
            self.dim_case_embd, dtype=self.prec, device=env.DEVICE
        )[case_idx]

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
            "fparam_neuron": self.fparam_neuron,
            "dim_case_embd": self.dim_case_embd,
            "group_reduce": self.group_reduce,
            "neuron": self.neuron,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "type_map": self.type_map,
        }
