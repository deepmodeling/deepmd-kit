# SPDX-License-Identifier: LGPL-3.0-or-later
"""The Uni-Mol v1 pretraining head set, as a deepmd fitting.

Ported from Uni-Mol (https://github.com/deepmodeling/Uni-Mol) at commit 90f52c4,
MIT licensed:

    Copyright (c) DP Technology
    This source code is licensed under the MIT license found in the LICENSE
    file in the root directory of that source tree.

Three heads sit on the Uni-Mol backbone: the element head reads the node
representation, the coordinate head reads the pair delta, and the distance head
reads the pair representation. None of the three is reducible to a frame total
and none is differentiated with respect to the coordinates, because this task
denoises structures rather than predicting a potential energy surface.

The distance head predicts a full row per atom, and upstream's objective counts
the two virtual tokens among the columns, so the output is padded to
``max_atoms + 2`` columns and the loss masks it back down.
"""

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    NativeOP,
)
from deepmd.dpmodel.descriptor.unimol_nn import (
    DistanceHead,
    MaskLMHead,
    NonLinearHead,
    coord_update,
)
from deepmd.dpmodel.fitting.base_fitting import (
    BaseFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@BaseFitting.register("unimol_pretrain")
class UniMolPretrainFitting(NativeOP, BaseFitting):
    r"""The three self-supervised heads of Uni-Mol v1.

    Parameters
    ----------
    ntypes : int
        Number of element types of the model.
    dim_descrpt : int
        Width of the node representation produced by the backbone.
    n_token : int
        Size of the Uni-Mol vocabulary, which is the width of the element head.
    attention_heads : int
        Width of the pair channel, which both pair-reading heads consume.
    max_atoms : int
        Largest molecule accepted, which fixes the column count of the distance
        head at ``max_atoms + 2``.
    activation_function : str
        Activation of the heads; Uni-Mol uses the exact GELU.
    mask_token_head : bool
        Build the element head.
    coord_head : bool
        Build the coordinate head.
    dist_head : bool
        Build the distance head.
    precision : str
        Floating-point precision of the parameters.
    seed : int, optional
        Random seed for initialization.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        n_token: int = 31,
        attention_heads: int = 64,
        max_atoms: int = 256,
        activation_function: str = "gelu_erf",
        mask_token_head: bool = True,
        coord_head: bool = True,
        dist_head: bool = True,
        precision: str = "float64",
        seed: int | list[int] | None = None,
        type_map: list[str] | None = None,
        **kwargs: float | str | bool,
    ) -> None:
        del kwargs
        self.type_map = list(type_map) if type_map is not None else None
        self.ntypes = ntypes
        self.dim_descrpt = dim_descrpt
        self.n_token = n_token
        self.attention_heads = attention_heads
        self.max_atoms = max_atoms
        self.activation_function = activation_function
        self.precision = precision
        self.lm_head = (
            MaskLMHead(dim_descrpt, n_token, activation_function, precision, seed)
            if mask_token_head
            else None
        )
        self.pair2coord_proj = (
            NonLinearHead(
                attention_heads,
                1,
                activation_function,
                hidden=attention_heads,
                precision=precision,
                seed=seed,
            )
            if coord_head
            else None
        )
        self.dist_head = (
            DistanceHead(attention_heads, activation_function, precision, seed)
            if dist_head
            else None
        )

    def get_type_map(self) -> list[str]:
        """Element names, as configured on the model."""
        return self.type_map if self.type_map is not None else []

    def change_type_map(
        self, type_map: list[str], model_with_new_type_stat: object = None
    ) -> None:
        """Adopt a new element list.

        The heads are indexed by Uni-Mol token, not by deepmd type, so nothing
        but the recorded names changes.
        """
        del model_with_new_type_stat
        self.type_map = list(type_map)
        self.ntypes = len(type_map)

    def output_def(self) -> FittingOutputDef:
        """Declare the three head outputs.

        All three are per-atom, none reduces to a frame total, and none is
        differentiated with respect to coordinates or cell.
        """
        variables = []
        if self.lm_head is not None:
            variables.append(
                OutputVariableDef(
                    "token_logits",
                    [self.n_token],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        if self.pair2coord_proj is not None:
            variables.append(
                OutputVariableDef(
                    "coord_update",
                    [3],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        if self.dist_head is not None:
            variables.append(
                OutputVariableDef(
                    "pair_dist",
                    [self.max_atoms + 2],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        # The two regularisers are frame scalars, but only per-atom variables
        # survive the atomic-output machinery, so each is broadcast over the
        # local atoms and the loss averages it back with the real-atom mask.
        for name in ("x_norm", "delta_pair_norm"):
            variables.append(
                OutputVariableDef(
                    name,
                    [1],
                    reducible=False,
                    r_differentiable=False,
                    c_differentiable=False,
                )
            )
        return FittingOutputDef(variables)

    def call_tokens(self, backbone: dict[str, Array]) -> dict[str, Array]:
        """Run the heads on the token-resolution backbone output.

        Parameters
        ----------
        backbone
            The dictionary returned by
            :meth:`deepmd.dpmodel.descriptor.unimol.DescrptUniMol.forward_tokens`.

        Returns
        -------
        dict
            ``token_logits`` and ``coord_update`` carry one row per local atom,
            with the two virtual tokens dropped; ``pair_dist`` keeps the virtual
            columns, padded out to ``max_atoms + 2``; the two norm regularisers
            are passed through for the loss.
        """
        xp = array_api_compat.array_namespace(backbone["node_ebd"])
        node = backbone["node_ebd"]
        nf, nt = node.shape[0], node.shape[1]
        nloc = nt - 2
        out = {}
        for name in ("x_norm", "delta_pair_norm"):
            value = xp.astype(backbone[name], node.dtype)
            out[name] = xp.full((nf, nloc, 1), value, dtype=node.dtype)
        if self.lm_head is not None:
            logits = self.lm_head(node)
            out["token_logits"] = logits[:, 1 : nloc + 1, :]
        if self.pair2coord_proj is not None:
            updated = coord_update(
                backbone["coord"],
                backbone["delta_pair_rep"],
                backbone["padding_mask"],
                self.pair2coord_proj,
            )
            out["coord_update"] = updated[:, 1 : nloc + 1, :]
        if self.dist_head is not None:
            dist = self.dist_head(backbone["pair_rep"])[:, 1 : nloc + 1, :]
            width = self.max_atoms + 2
            if dist.shape[-1] < width:
                pad = xp.zeros((nf, nloc, width - dist.shape[-1]), dtype=dist.dtype)
                dist = xp.concat([dist, pad], axis=-1)
            out["pair_dist"] = dist
        return out

    def call(self, descriptor: Array, atype: Array, **kwargs) -> dict[str, Array]:  # noqa: ANN003
        """Not reachable: the heads need token-resolution inputs.

        The pretraining path goes through :meth:`call_tokens`, driven by the
        Uni-Mol atomic model, because the two virtual tokens and the pair
        channel do not fit the descriptor's five-tuple.
        """
        raise NotImplementedError(
            "unimol_pretrain reads token-resolution backbone output; it is driven "
            "through call_tokens by the unimol atomic model"
        )

    def serialize(self) -> dict:
        """Serialize the fitting."""
        return {
            "@class": "Fitting",
            "type": "unimol_pretrain",
            "@version": 1,
            "ntypes": self.ntypes,
            "type_map": self.type_map,
            "dim_descrpt": self.dim_descrpt,
            "n_token": self.n_token,
            "attention_heads": self.attention_heads,
            "max_atoms": self.max_atoms,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mask_token_head": self.lm_head is not None,
            "coord_head": self.pair2coord_proj is not None,
            "dist_head": self.dist_head is not None,
            "lm_head": None if self.lm_head is None else self.lm_head.serialize(),
            "pair2coord_proj": None
            if self.pair2coord_proj is None
            else self.pair2coord_proj.serialize(),
            "dist_head_net": None
            if self.dist_head is None
            else self.dist_head.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "UniMolPretrainFitting":
        """Deserialize the fitting."""
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class", None)
        data.pop("type", None)
        parts = {
            k: data.pop(k) for k in ("lm_head", "pair2coord_proj", "dist_head_net")
        }
        obj = cls(**data)
        if parts["lm_head"] is not None:
            obj.lm_head = MaskLMHead.deserialize(parts["lm_head"])
        if parts["pair2coord_proj"] is not None:
            obj.pair2coord_proj = NonLinearHead.deserialize(parts["pair2coord_proj"])
        if parts["dist_head_net"] is not None:
            obj.dist_head = DistanceHead.deserialize(parts["dist_head_net"])
        return obj
